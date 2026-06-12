"""Loss helpers for NER training experiments."""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Sequence

import torch
import torch.nn.functional as F


def build_label_loss_weights(
    strategy: str,
    label2id: dict[str, int],
    train_labels: Sequence[Sequence[str]],
    entity_weight: float = 1.0,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
) -> torch.Tensor | None:
    """Build per-label CE weights.

    Supported strategies:
    - ``none``: return ``None`` and use the model's standard loss.
    - ``entity``: assign one configurable weight to all non-``O`` labels.
    - ``inverse_sqrt``: down-weight frequent labels and up-weight rare labels
      using sqrt inverse frequency with clamps.
    """
    normalized_strategy = strategy.lower().strip()
    if normalized_strategy == "none":
        return None

    weights = torch.ones(len(label2id), dtype=torch.float)

    if normalized_strategy == "entity":
        for label, idx in label2id.items():
            if label != "O":
                weights[idx] = float(entity_weight)
        return weights

    if normalized_strategy != "inverse_sqrt":
        raise ValueError(f"Unsupported loss weighting strategy: {strategy}")

    counts: Counter[str] = Counter()
    for sequence in train_labels:
        counts.update(sequence)

    total = sum(counts.get(label, 0) for label in label2id)
    mean_count = total / max(1, len(label2id))
    for label, idx in label2id.items():
        count = max(1, counts.get(label, 0))
        weight = sqrt(mean_count / count)
        weights[idx] = max(float(min_weight), min(float(max_weight), float(weight)))

    return weights


def calculate_weighted_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted token-classification CE while ignoring ``-100`` labels."""
    num_labels = logits.shape[-1]
    return F.cross_entropy(
        logits.view(-1, num_labels),
        labels.view(-1),
        weight=class_weights.to(logits.device),
        ignore_index=-100,
    )
