"""Command-line wrapper for NER training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ner.train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Chinese medical NER model.")
    parser.add_argument(
        "--resume-active",
        action="store_true",
        help="Continue training from the active checkpoint in outputs/ner_model_active.txt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs for this run.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Continue training from a specific checkpoint directory.",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Override NER learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override NER batch size.")
    parser.add_argument("--max-length", type=int, default=None, help="Override NER max sequence length.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override NER weight decay.")
    parser.add_argument(
        "--loss-weighting",
        choices=["none", "entity", "inverse_sqrt"],
        default=None,
        help="Optional weighted CE strategy for token classification.",
    )
    parser.add_argument(
        "--entity-loss-weight",
        type=float,
        default=None,
        help="Weight for non-O labels when --loss-weighting entity is used.",
    )
    parser.add_argument("--min-class-weight", type=float, default=None, help="Minimum inverse-sqrt class weight.")
    parser.add_argument("--max-class-weight", type=float, default=None, help="Maximum inverse-sqrt class weight.")
    parser.add_argument(
        "--entity-threshold",
        type=float,
        default=None,
        help="Optional non-O token confidence threshold used during dev evaluation.",
    )
    args = parser.parse_args()

    ner_overrides = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "loss_weighting": args.loss_weighting,
        "entity_loss_weight": args.entity_loss_weight,
        "min_class_weight": args.min_class_weight,
        "max_class_weight": args.max_class_weight,
        "entity_confidence_threshold": args.entity_threshold,
    }
    result = train(
        ROOT / "configs" / "config.yaml",
        resume_active=args.resume_active,
        num_epochs=args.epochs,
        ner_overrides=ner_overrides,
        resume_from=args.resume_from,
    )
    print(f"Best dev F1: {result['best_f1']:.4f}")
    print(f"Model saved to: {result['output_dir']}")
