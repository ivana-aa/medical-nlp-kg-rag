"""Optional lexicon-based post-processing for NER predictions."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.ner.dataset import MedicalNERDataset, read_bio_file
from src.ner.error_analysis import extract_bio_entities
from src.ner.paths import resolve_ner_model_dir
from src.ner.predict import bio_to_entities
from src.ner.train import get_device, load_config
from src.utils.metrics import align_predictions


@dataclass(frozen=True)
class LexiconEntry:
    """One entity text with label frequency statistics from training data."""

    text: str
    label: str
    count: int
    label_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_entity_lexicon(
    tokens: Sequence[Sequence[str]],
    labels: Sequence[Sequence[str]],
    min_text_length: int = 2,
) -> dict[str, LexiconEntry]:
    """Build an entity-text lexicon from BIO training labels."""
    counts: dict[str, Counter[str]] = {}
    for sentence_tokens, sentence_labels in zip(tokens, labels):
        for entity in extract_bio_entities(sentence_tokens, sentence_labels):
            if len(entity.text) < min_text_length:
                continue
            counts.setdefault(entity.text, Counter())[entity.label] += 1

    lexicon: dict[str, LexiconEntry] = {}
    for text, label_counts in counts.items():
        label, count = label_counts.most_common(1)[0]
        lexicon[text] = LexiconEntry(
            text=text,
            label=label,
            count=sum(label_counts.values()),
            label_counts=dict(label_counts),
        )
    return lexicon


def apply_lexicon_postprocess(
    chars: Sequence[str],
    entities: list[dict[str, object]],
    lexicon: dict[str, LexiconEntry],
    min_count: int = 3,
    add_missing: bool = False,
) -> list[dict[str, object]]:
    """Correct labels and optionally add missing entities using a training lexicon."""
    processed = [_normalize_entity(entity) for entity in entities]
    for entity in processed:
        entry = lexicon.get(str(entity["text"]))
        if entry is not None and entry.count >= min_count:
            entity["type"] = entry.label

    if add_missing:
        processed = _add_missing_entities(chars, processed, lexicon, min_count)
    return sorted(processed, key=lambda item: (int(item["start"]), int(item["end"])))


def entities_to_bio_labels(chars: Sequence[str], entities: list[dict[str, object]]) -> list[str]:
    """Convert non-overlapping entity dictionaries back to BIO labels."""
    labels = ["O"] * len(chars)
    for entity in sorted(entities, key=lambda item: (int(item["start"]), int(item["end"]))):
        start = int(entity["start"])
        end = int(entity["end"])
        entity_type = str(entity["type"])
        if start < 0 or end > len(chars) or start >= end:
            continue
        if any(label != "O" for label in labels[start:end]):
            continue
        labels[start] = f"B-{entity_type}"
        for idx in range(start + 1, end):
            labels[idx] = f"I-{entity_type}"
    return labels


@torch.no_grad()
def evaluate_saved_model_with_lexicon(
    config_path: str | Path = "configs/config.yaml",
    split: str = "test",
    model_dir: str | Path | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
    min_count: int = 3,
    add_missing: bool = False,
) -> dict[str, object]:
    """Evaluate optional lexicon post-processing against a saved NER model."""
    config = load_config(config_path)
    ner_config = dict(config["ner"])
    if batch_size is not None:
        ner_config["batch_size"] = batch_size
    if max_length is not None:
        ner_config["max_length"] = max_length
    project_config = config["project"]

    output_dir = Path(model_dir) if model_dir is not None else resolve_ner_model_dir(ner_config["output_dir"])
    label_map_path = output_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError("Saved model not found. Run `python scripts/train_ner.py` first.")

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    label2id = {key: int(value) for key, value in label_map["label2id"].items()}
    id2label = {int(key): value for key, value in label_map["id2label"].items()}

    data_dir = Path(ner_config["data_dir"])
    train_tokens, train_labels = read_bio_file(data_dir / ner_config["train_file"])
    eval_tokens, eval_labels = read_bio_file(data_dir / ner_config[f"{split}_file"])
    lexicon = build_entity_lexicon(train_tokens, train_labels)

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForTokenClassification.from_pretrained(output_dir)
    device = get_device(project_config.get("device", "auto"))
    model.to(device)
    model.eval()

    dataset = MedicalNERDataset(
        eval_tokens,
        eval_labels,
        tokenizer,
        label2id,
        max_length=int(ner_config["max_length"]),
    )
    dataloader = DataLoader(dataset, batch_size=int(ner_config["batch_size"]), shuffle=False)

    raw_predictions: list[list[int]] = []
    gold_ids: list[list[int]] = []
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        raw_predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        gold_ids.extend(batch["labels"].cpu().tolist())

    raw_pred_labels, gold_labels = align_predictions(raw_predictions, gold_ids, id2label)
    aligned_tokens = [list(sentence_tokens)[: len(sentence_labels)] for sentence_tokens, sentence_labels in zip(eval_tokens, gold_labels)]

    postprocessed_labels: list[list[str]] = []
    corrected_entity_count = 0
    added_entity_count = 0
    for chars, sentence_pred_labels in zip(aligned_tokens, raw_pred_labels):
        raw_entities = bio_to_entities(list(chars), list(sentence_pred_labels))
        processed_entities = apply_lexicon_postprocess(
            chars,
            raw_entities,
            lexicon,
            min_count=min_count,
            add_missing=add_missing,
        )
        corrected_entity_count += _count_label_changes(raw_entities, processed_entities)
        added_entity_count += max(0, len(processed_entities) - len(raw_entities))
        postprocessed_labels.append(entities_to_bio_labels(chars, processed_entities))

    return {
        "split": split,
        "model_dir": str(output_dir),
        "lexicon_size": len(lexicon),
        "min_count": min_count,
        "add_missing": add_missing,
        "corrected_entity_count": corrected_entity_count,
        "added_entity_count": added_entity_count,
        "raw": _label_metric_report(gold_labels, raw_pred_labels),
        "postprocessed": _label_metric_report(gold_labels, postprocessed_labels),
    }


def _normalize_entity(entity: dict[str, object]) -> dict[str, object]:
    return {
        "text": str(entity.get("text", "")),
        "type": str(entity.get("type", "")),
        "start": int(entity.get("start", 0)),
        "end": int(entity.get("end", 0)),
    }


def _add_missing_entities(
    chars: Sequence[str],
    entities: list[dict[str, object]],
    lexicon: dict[str, LexiconEntry],
    min_count: int,
) -> list[dict[str, object]]:
    sentence = "".join(chars)
    occupied = {
        idx
        for entity in entities
        for idx in range(int(entity["start"]), int(entity["end"]))
    }
    result = list(entities)
    candidates = [
        entry
        for entry in lexicon.values()
        if entry.count >= min_count and len(entry.text) >= 2
    ]
    for entry in sorted(candidates, key=lambda item: (-len(item.text), item.text)):
        start = sentence.find(entry.text)
        while start != -1:
            end = start + len(entry.text)
            if not any(idx in occupied for idx in range(start, end)):
                result.append(
                    {
                        "text": entry.text,
                        "type": entry.label,
                        "start": start,
                        "end": end,
                    }
                )
                occupied.update(range(start, end))
            start = sentence.find(entry.text, end)
    return result


def _count_label_changes(raw_entities: list[dict[str, object]], processed_entities: list[dict[str, object]]) -> int:
    processed_by_span = {
        (int(entity["start"]), int(entity["end"])): str(entity["type"])
        for entity in processed_entities
    }
    changes = 0
    for entity in raw_entities:
        key = (int(entity["start"]), int(entity["end"]))
        if key in processed_by_span and processed_by_span[key] != str(entity["type"]):
            changes += 1
    return changes


def _label_metric_report(true_labels: list[list[str]], predicted_labels: list[list[str]]) -> dict[str, object]:
    return {
        "precision": precision_score(true_labels, predicted_labels),
        "recall": recall_score(true_labels, predicted_labels),
        "f1": f1_score(true_labels, predicted_labels),
        "report": classification_report(true_labels, predicted_labels, digits=4),
    }
