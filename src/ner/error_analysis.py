"""Error analysis utilities for Chinese medical NER."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.ner.dataset import MedicalNERDataset, read_bio_file
from src.ner.decoding import apply_entity_confidence_threshold
from src.ner.paths import resolve_ner_model_dir
from src.ner.train import get_device, load_config
from src.utils.metrics import align_predictions, compute_ner_metrics


@dataclass(frozen=True)
class EntitySpan:
    """One entity span decoded from BIO labels."""

    text: str
    label: str
    start: int
    end: int

    def key(self) -> tuple[str, int, int]:
        return (self.label, self.start, self.end)

    def span_key(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_bio_entities(tokens: Sequence[str], labels: Sequence[str]) -> list[EntitySpan]:
    """Extract entity spans from one BIO label sequence."""
    entities: list[EntitySpan] = []
    current_label = ""
    current_start: int | None = None
    length = min(len(tokens), len(labels))

    def close_entity(end: int) -> None:
        nonlocal current_label, current_start
        if current_start is not None:
            entities.append(
                EntitySpan(
                    text="".join(tokens[current_start:end]),
                    label=current_label,
                    start=current_start,
                    end=end,
                )
            )
        current_label = ""
        current_start = None

    for idx in range(length):
        raw_label = str(labels[idx]).strip()
        if not raw_label or raw_label == "O":
            close_entity(idx)
            continue

        prefix, entity_label = _split_bio_label(raw_label)
        starts_new = (
            prefix == "B"
            or current_start is None
            or current_label != entity_label
            or prefix not in {"I"}
        )
        if starts_new:
            close_entity(idx)
            current_start = idx
            current_label = entity_label

    close_entity(length)
    return entities


def analyze_ner_sequences(
    tokens: Sequence[Sequence[str]],
    true_labels: Sequence[Sequence[str]],
    pred_labels: Sequence[Sequence[str]],
    max_examples: int = 30,
    top_k: int = 20,
) -> dict[str, object]:
    """Analyze exact-match, false-positive, false-negative, and overlap errors."""
    per_label: dict[str, Counter[str]] = {}
    false_positive_counter: Counter[tuple[str, str]] = Counter()
    false_negative_counter: Counter[tuple[str, str]] = Counter()
    false_positive_examples: list[dict[str, object]] = []
    false_negative_examples: list[dict[str, object]] = []
    boundary_errors: list[dict[str, object]] = []
    label_confusions: list[dict[str, object]] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for sentence_index, (sent_tokens, sent_true, sent_pred) in enumerate(zip(tokens, true_labels, pred_labels)):
        sentence = "".join(sent_tokens)
        gold_entities = extract_bio_entities(sent_tokens, sent_true)
        pred_entities = extract_bio_entities(sent_tokens, sent_pred)
        gold_by_key = {entity.key(): entity for entity in gold_entities}
        pred_by_key = {entity.key(): entity for entity in pred_entities}

        true_positive_keys = set(gold_by_key) & set(pred_by_key)
        false_positives = [entity for key, entity in pred_by_key.items() if key not in gold_by_key]
        false_negatives = [entity for key, entity in gold_by_key.items() if key not in pred_by_key]

        for key in true_positive_keys:
            label = key[0]
            _label_counts(per_label, label)["tp"] += 1
            total_tp += 1

        for entity in false_positives:
            _label_counts(per_label, entity.label)["fp"] += 1
            false_positive_counter[(entity.label, entity.text)] += 1
            total_fp += 1
            if len(false_positive_examples) < max_examples:
                false_positive_examples.append(_example(sentence_index, sentence, entity))

        for entity in false_negatives:
            _label_counts(per_label, entity.label)["fn"] += 1
            false_negative_counter[(entity.label, entity.text)] += 1
            total_fn += 1
            if len(false_negative_examples) < max_examples:
                false_negative_examples.append(_example(sentence_index, sentence, entity))

        if len(label_confusions) < max_examples:
            label_confusions.extend(
                _find_label_confusions(sentence_index, sentence, gold_entities, pred_entities, max_examples - len(label_confusions))
            )
        if len(boundary_errors) < max_examples:
            boundary_errors.extend(
                _find_boundary_errors(sentence_index, sentence, gold_entities, pred_entities, max_examples - len(boundary_errors))
            )

    return {
        "totals": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
        "overall_scores": _scores(total_tp, total_fp, total_fn),
        "per_label": _format_per_label(per_label),
        "top_false_positive_entities": _format_top_entities(false_positive_counter, top_k),
        "top_false_negative_entities": _format_top_entities(false_negative_counter, top_k),
        "false_positive_examples": false_positive_examples,
        "false_negative_examples": false_negative_examples,
        "boundary_errors": boundary_errors[:max_examples],
        "label_confusions": label_confusions[:max_examples],
    }


@torch.no_grad()
def analyze_saved_model_errors(
    config_path: str | Path = "configs/config.yaml",
    split: str = "test",
    model_dir: str | Path | None = None,
    ner_overrides: dict[str, object] | None = None,
    max_examples: int = 30,
    top_k: int = 20,
) -> dict[str, object]:
    """Run a saved NER model and return a structured error-analysis report."""
    config = load_config(config_path)
    ner_config = dict(config["ner"])
    if ner_overrides:
        ner_config.update({key: value for key, value in ner_overrides.items() if value is not None})
    project_config = config["project"]

    output_dir = Path(model_dir) if model_dir is not None else resolve_ner_model_dir(ner_config["output_dir"])
    label_map_path = output_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError("Saved model not found. Run `python scripts/train_ner.py` first.")

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    label2id = {key: int(value) for key, value in label_map["label2id"].items()}
    id2label = {int(key): value for key, value in label_map["id2label"].items()}

    file_name = {
        "train": ner_config["train_file"],
        "dev": ner_config["dev_file"],
        "test": ner_config["test_file"],
    }[split]
    data_path = Path(ner_config["data_dir"]) / file_name
    tokens, labels = read_bio_file(data_path)

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForTokenClassification.from_pretrained(output_dir)
    device = get_device(project_config.get("device", "auto"))
    model.to(device)
    model.eval()

    dataset = MedicalNERDataset(
        tokens,
        labels,
        tokenizer,
        label2id,
        max_length=int(ner_config["max_length"]),
    )
    dataloader = DataLoader(dataset, batch_size=int(ner_config["batch_size"]), shuffle=False)

    predictions: list[list[int]] = []
    gold_label_ids: list[list[int]] = []
    entity_threshold = float(ner_config.get("entity_confidence_threshold", 0.0))
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        if entity_threshold > 0:
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidences, batch_predictions = torch.max(probabilities, dim=-1)
            batch_predictions = apply_entity_confidence_threshold(
                batch_predictions,
                confidences,
                id2label,
                threshold=entity_threshold,
            )
        else:
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(batch_predictions.cpu().tolist())
        gold_label_ids.extend(batch["labels"].cpu().tolist())

    metrics = compute_ner_metrics(predictions, gold_label_ids, id2label)
    pred_label_names, true_label_names = align_predictions(predictions, gold_label_ids, id2label)
    aligned_tokens = [list(sentence_tokens)[: len(sentence_labels)] for sentence_tokens, sentence_labels in zip(tokens, true_label_names)]
    analysis = analyze_ner_sequences(
        aligned_tokens,
        true_label_names,
        pred_label_names,
        max_examples=max_examples,
        top_k=top_k,
    )
    analysis.update(
        {
            "split": split,
            "data_path": str(data_path),
            "model_dir": str(output_dir),
            "entity_threshold": entity_threshold,
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "classification_report": metrics["report"],
        }
    )
    return analysis


def _split_bio_label(label: str) -> tuple[str, str]:
    if "-" not in label:
        return "B", label
    prefix, entity_label = label.split("-", 1)
    return prefix, entity_label


def _label_counts(per_label: dict[str, Counter[str]], label: str) -> Counter[str]:
    if label not in per_label:
        per_label[label] = Counter({"tp": 0, "fp": 0, "fn": 0})
    return per_label[label]


def _scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _format_per_label(per_label: dict[str, Counter[str]]) -> dict[str, dict[str, object]]:
    formatted: dict[str, dict[str, object]] = {}
    for label in sorted(per_label):
        counts = per_label[label]
        scores = _scores(counts["tp"], counts["fp"], counts["fn"])
        formatted[label] = {
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            **scores,
        }
    return formatted


def _format_top_entities(counter: Counter[tuple[str, str]], top_k: int) -> list[dict[str, object]]:
    rows = []
    for (label, text), count in counter.most_common(top_k):
        rows.append({"label": label, "text": text, "count": count})
    return rows


def _example(sentence_index: int, sentence: str, entity: EntitySpan) -> dict[str, object]:
    return {
        "sentence_index": sentence_index,
        "sentence": sentence,
        "entity": entity.to_dict(),
    }


def _find_label_confusions(
    sentence_index: int,
    sentence: str,
    gold_entities: list[EntitySpan],
    pred_entities: list[EntitySpan],
    limit: int,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    pred_by_span = {entity.span_key(): entity for entity in pred_entities}
    for gold in gold_entities:
        pred = pred_by_span.get(gold.span_key())
        if pred is None or pred.label == gold.label:
            continue
        examples.append(
            {
                "sentence_index": sentence_index,
                "sentence": sentence,
                "text": gold.text,
                "gold_label": gold.label,
                "predicted_label": pred.label,
                "gold": gold.to_dict(),
                "predicted": pred.to_dict(),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _find_boundary_errors(
    sentence_index: int,
    sentence: str,
    gold_entities: list[EntitySpan],
    pred_entities: list[EntitySpan],
    limit: int,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for gold in gold_entities:
        for pred in pred_entities:
            if gold.label != pred.label or gold.span_key() == pred.span_key():
                continue
            if not _overlaps(gold, pred):
                continue
            examples.append(
                {
                    "sentence_index": sentence_index,
                    "sentence": sentence,
                    "gold": gold.to_dict(),
                    "predicted": pred.to_dict(),
                }
            )
            break
        if len(examples) >= limit:
            break
    return examples


def _overlaps(left: EntitySpan, right: EntitySpan) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)
