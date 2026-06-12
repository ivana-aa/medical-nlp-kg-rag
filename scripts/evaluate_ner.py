"""Command-line wrapper for NER evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ner.evaluate import evaluate_model
from src.ner.paths import resolve_ner_model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved NER model.")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Evaluate a specific checkpoint directory instead of the active model.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override NER evaluation batch size.")
    parser.add_argument("--max-length", type=int, default=None, help="Override NER max sequence length.")
    parser.add_argument(
        "--entity-threshold",
        type=float,
        default=None,
        help="Convert non-O token predictions below this confidence to O.",
    )
    args = parser.parse_args()

    config_path = ROOT / "configs" / "config.yaml"
    ner_overrides = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "entity_confidence_threshold": args.entity_threshold,
    }
    result = evaluate_model(
        config_path,
        split=args.split,
        model_dir=args.model_dir,
        ner_overrides=ner_overrides,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = args.model_dir if args.model_dir is not None else resolve_ner_model_dir(ROOT / config["ner"]["output_dir"])
    report_name = "eval_report.json"
    if args.entity_threshold is not None and args.entity_threshold > 0:
        report_name = f"eval_report_threshold_{args.entity_threshold:.2f}.json"
    report_path = model_dir / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1": float(result["f1"]),
                "entity_threshold": args.entity_threshold,
                "report": result["report"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Split:     {args.split}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1:        {result['f1']:.4f}")
    print(f"Saved report to: {report_path}")
    print(result["report"])


if __name__ == "__main__":
    main()
