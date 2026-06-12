"""Metric utilities for Chinese medical NER."""

from __future__ import annotations

from typing import Iterable

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def align_predictions(
    predictions: Iterable[Iterable[int]],
    labels: Iterable[Iterable[int]],
    id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert token ids to label strings and ignore padding/special tokens.

    Hugging Face token classification uses ``-100`` labels for positions that
    should not contribute to the loss. The same positions must be ignored when
    computing seqeval metrics.
    """
    true_predictions: list[list[str]] = []
    true_labels: list[list[str]] = []

    for pred_seq, label_seq in zip(predictions, labels):
        sent_preds: list[str] = []
        sent_labels: list[str] = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if int(label_id) == -100:
                continue
            sent_preds.append(id2label[int(pred_id)])
            sent_labels.append(id2label[int(label_id)])
        true_predictions.append(sent_preds)
        true_labels.append(sent_labels)

    return true_predictions, true_labels


def compute_ner_metrics(
    predictions: list[list[int]],
    labels: list[list[int]],
    id2label: dict[int, str],
) -> dict[str, object]:
    """Return precision, recall, F1, and a readable classification report."""
    true_predictions, true_labels = align_predictions(predictions, labels, id2label)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions, digits=4),
    }
