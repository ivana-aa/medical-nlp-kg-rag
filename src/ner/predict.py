"""Inference helper for Chinese medical NER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.ner.paths import resolve_ner_model_dir


class MedicalNERPredictor:
    """Load a saved NER checkpoint and extract entities from raw text."""

    def __init__(
        self,
        model_dir: str | Path = "outputs/ner_model",
        device: str = "auto",
        entity_lexicon: dict[str, Any] | None = None,
        lexicon_min_count: int = 5,
        lexicon_add_missing: bool = False,
    ) -> None:
        self.model_dir = resolve_ner_model_dir(model_dir)
        self.entity_lexicon = entity_lexicon
        self.lexicon_min_count = lexicon_min_count
        self.lexicon_add_missing = lexicon_add_missing
        label_map_path = self.model_dir / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError("NER model is missing. Train it with `python scripts/train_ner.py`.")

        with label_map_path.open("r", encoding="utf-8") as f:
            label_map = json.load(f)

        self.id2label = {int(k): v for k, v in label_map["id2label"].items()}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str, max_length: int = 128) -> list[dict[str, object]]:
        """Return entities with type, text, and character offsets."""
        chars = list(text.strip())
        if not chars:
            return []

        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        model_inputs = {key: value.to(self.device) for key, value in encoding.items()}
        logits = self.model(**model_inputs).logits
        pred_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

        char_labels: list[str] = ["O"] * len(chars)
        seen_word_ids: set[int] = set()
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_ids or word_id >= len(chars):
                continue
            seen_word_ids.add(word_id)
            char_labels[word_id] = self.id2label[int(pred_ids[token_idx])]

        entities = bio_to_entities(chars, char_labels)
        return self._postprocess_entities(chars, entities)

    def _postprocess_entities(
        self,
        chars: list[str],
        entities: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Apply optional training-lexicon post-processing to model entities."""
        if not self.entity_lexicon:
            return entities

        from src.ner.lexicon_postprocess import apply_lexicon_postprocess

        return apply_lexicon_postprocess(
            chars,
            entities,
            self.entity_lexicon,
            min_count=int(self.lexicon_min_count),
            add_missing=bool(self.lexicon_add_missing),
        )


def bio_to_entities(chars: list[str], labels: list[str]) -> list[dict[str, object]]:
    """Convert BIO labels to entity dictionaries."""
    entities: list[dict[str, object]] = []
    start = None
    current_type = None

    for idx, label in enumerate(labels + ["O"]):
        if label.startswith("B-"):
            if current_type is not None and start is not None:
                entities.append(
                    {
                        "text": "".join(chars[start:idx]),
                        "type": current_type,
                        "start": start,
                        "end": idx,
                    }
                )
            start = idx
            current_type = label[2:]
        elif label.startswith("I-") and current_type == label[2:]:
            continue
        else:
            if current_type is not None and start is not None:
                entities.append(
                    {
                        "text": "".join(chars[start:idx]),
                        "type": current_type,
                        "start": start,
                        "end": idx,
                    }
                )
            start = None
            current_type = None

    return entities
