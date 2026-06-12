"""Model construction for the NER token classifier."""

from __future__ import annotations

from transformers import AutoModelForTokenClassification


def build_token_classifier(
    model_name: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
):
    """Load a Chinese pretrained model for token classification."""
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
