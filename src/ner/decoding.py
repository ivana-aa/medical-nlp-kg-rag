"""Decoding helpers for token-classification NER outputs."""

from __future__ import annotations

import torch


def apply_entity_confidence_threshold(
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    id2label: dict[int, str],
    threshold: float = 0.0,
) -> torch.Tensor:
    """Convert low-confidence entity predictions to ``O``.

    ``threshold=0`` preserves the original argmax predictions. Only non-``O``
    predictions are filtered; existing ``O`` tokens remain unchanged.
    """
    if threshold <= 0:
        return predictions

    o_id = next((idx for idx, label in id2label.items() if label == "O"), 0)
    entity_ids = torch.tensor(
        [idx for idx, label in id2label.items() if label != "O"],
        device=predictions.device,
        dtype=predictions.dtype,
    )
    if entity_ids.numel() == 0:
        return predictions

    is_entity = torch.isin(predictions, entity_ids)
    low_confidence = confidences < float(threshold)
    filtered = predictions.clone()
    filtered[is_entity & low_confidence] = int(o_id)
    return filtered
