"""Training loop for Chinese medical NER."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.ner.dataset import MedicalNERDataset, read_bio_file
from src.ner.decoding import apply_entity_confidence_threshold
from src.ner.losses import build_label_loss_weights, calculate_weighted_token_loss
from src.ner.model import build_token_classifier
from src.ner.paths import resolve_ner_model_dir, write_active_model_pointer
from src.utils.logger import setup_logger
from src.utils.metrics import compute_ner_metrics


logger = setup_logger(__name__)


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML config from the project root."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Make demo training as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config_device: str = "auto") -> torch.device:
    """Choose CUDA when available unless the config forces CPU."""
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


@torch.no_grad()
def evaluate_dataloader(
    model,
    dataloader: DataLoader,
    device: torch.device,
    id2label: dict[int, str],
    entity_confidence_threshold: float = 0.0,
) -> dict[str, object]:
    """Evaluate a model on a dataloader and return seqeval metrics."""
    model.eval()
    all_predictions: list[list[int]] = []
    all_labels: list[list[int]] = []

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


def train(
    config_path: str | Path = "configs/config.yaml",
    resume_active: bool = False,
    num_epochs: int | None = None,
    ner_overrides: dict[str, Any] | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, object]:
    """Train the NER model and save the best checkpoint."""
    config = load_config(config_path)
    project_config = config["project"]
    ner_config = dict(config["ner"])
    if ner_overrides:
        ner_config.update({key: value for key, value in ner_overrides.items() if value is not None})
    set_seed(int(project_config.get("seed", 42)))

    labels = ner_config["labels"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    data_dir = Path(ner_config["data_dir"])
    train_path = data_dir / ner_config["train_file"]
    dev_path = data_dir / ner_config["dev_file"]
    output_dir = Path(ner_config["output_dir"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = output_dir.parent / "ner_model_runs" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            "NER demo data not found. Run `python scripts/prepare_ner_data.py` first."
        )

    if resume_from is not None:
        model_source = str(Path(resume_from))
    elif resume_active:
        model_source = str(resolve_ner_model_dir(output_dir))
    else:
        model_source = ner_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    train_tokens, train_labels = read_bio_file(train_path)
    dev_tokens, dev_labels = read_bio_file(dev_path)

    train_dataset = MedicalNERDataset(
        train_tokens,
        train_labels,
        tokenizer,
        label2id,
        max_length=int(ner_config["max_length"]),
    )
    dev_dataset = MedicalNERDataset(
        dev_tokens,
        dev_labels,
        tokenizer,
        label2id,
        max_length=int(ner_config["max_length"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(ner_config["batch_size"]),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=int(ner_config["batch_size"]),
        shuffle=False,
    )

    device = get_device(project_config.get("device", "auto"))
    model = build_token_classifier(
        model_source,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    loss_weighting = str(ner_config.get("loss_weighting", "none"))
    class_weights = build_label_loss_weights(
        loss_weighting,
        label2id,
        train_labels,
        entity_weight=float(ner_config.get("entity_loss_weight", 1.0)),
        min_weight=float(ner_config.get("min_class_weight", 0.5)),
        max_weight=float(ner_config.get("max_class_weight", 3.0)),
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(ner_config["learning_rate"]),
        weight_decay=float(ner_config["weight_decay"]),
    )
    epochs = int(num_epochs if num_epochs is not None else ner_config["num_epochs"])
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    history: list[dict[str, float]] = []
    entity_confidence_threshold = float(ner_config.get("entity_confidence_threshold", 0.0))
    logger.info(
        (
            "Start training on %s with %d train examples | source=%s | epochs=%d | "
            "batch_size=%d | max_length=%d | lr=%.8f | weight_decay=%.4f | "
            "loss_weighting=%s | entity_threshold=%.3f"
        ),
        device,
        len(train_dataset),
        model_source,
        epochs,
        int(ner_config["batch_size"]),
        int(ner_config["max_length"]),
        float(ner_config["learning_rate"]),
        float(ner_config["weight_decay"]),
        loss_weighting,
        entity_confidence_threshold,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            if class_weights is None:
                outputs = model(**batch)
                loss = outputs.loss
            else:
                labels_tensor = batch["labels"]
                model_inputs = {key: value for key, value in batch.items() if key != "labels"}
                outputs = model(**model_inputs)
                loss = calculate_weighted_token_loss(outputs.logits, labels_tensor, class_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        metrics = evaluate_dataloader(
            model,
            dev_loader,
            device,
            id2label,
            entity_confidence_threshold=entity_confidence_threshold,
        )
        f1 = float(metrics["f1"])
        history.append({"epoch": epoch + 1, "loss": avg_loss, "dev_f1": f1})

        logger.info(
            "Epoch %d finished | loss=%.4f | dev_precision=%.4f | dev_recall=%.4f | dev_f1=%.4f",
            epoch + 1,
            avg_loss,
            float(metrics["precision"]),
            float(metrics["recall"]),
            f1,
        )

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            with (checkpoint_dir / "label_map.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info("Saved best model to %s", checkpoint_dir)

    with (checkpoint_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    write_active_model_pointer(output_dir, checkpoint_dir)
    logger.info("Active model pointer updated for %s", checkpoint_dir)

    return {"best_f1": best_f1, "history": history, "output_dir": str(checkpoint_dir)}


if __name__ == "__main__":
    train()
