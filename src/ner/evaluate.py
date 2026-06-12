"""Evaluation entrypoints for saved NER checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.ner.dataset import MedicalNERDataset, read_bio_file
from src.ner.decoding import apply_entity_confidence_threshold
from src.ner.paths import resolve_ner_model_dir
from src.ner.train import get_device, load_config
from src.utils.metrics import compute_ner_metrics


@torch.no_grad()
def evaluate_model(
    config_path: str | Path = "configs/config.yaml",
    split: str = "test",
    model_dir: str | Path | None = None,
    ner_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate a saved model on train/dev/test BIO files."""
    config = load_config(config_path)
    ner_config = dict(config["ner"])
    if ner_overrides:
        ner_config.update({key: value for key, value in ner_overrides.items() if value is not None})
    project_config = config["project"]

    output_dir = Path(model_dir) if model_dir is not None else resolve_ner_model_dir(ner_config["output_dir"])
    label_map_path = output_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError("Saved model not found. Run `python scripts/train_ner.py` first.")

    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    label2id = {k: int(v) for k, v in label_map["label2id"].items()}
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

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
    dataloader = DataLoader(
        dataset,
        batch_size=int(ner_config["batch_size"]),
        shuffle=False,
    )

    all_predictions: list[list[int]] = []
    all_labels: list[list[int]] = []
    entity_confidence_threshold = float(ner_config.get("entity_confidence_threshold", 0.0))
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        if entity_confidence_threshold > 0:
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidences, predictions = torch.max(probabilities, dim=-1)
            predictions = apply_entity_confidence_threshold(
                predictions,
                confidences,
                id2label,
                threshold=entity_confidence_threshold,
            )
        else:
            predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    return compute_ner_metrics(all_predictions, all_labels, id2label)


if __name__ == "__main__":
    result = evaluate_model()
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1:        {result['f1']:.4f}")
    print(result["report"])
