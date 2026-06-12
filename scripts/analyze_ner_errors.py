"""Run error analysis for a saved Chinese medical NER model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ner.error_analysis import analyze_saved_model_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze NER false positives, false negatives, and span errors.")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--model-dir", type=Path, default=None, help="Analyze a specific checkpoint directory.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override NER batch size.")
    parser.add_argument("--max-length", type=int, default=None, help="Override NER max sequence length.")
    parser.add_argument("--entity-threshold", type=float, default=None, help="Optional entity confidence threshold.")
    parser.add_argument("--max-examples", type=int, default=30, help="Maximum example rows per error section.")
    parser.add_argument("--top-k", type=int, default=20, help="Maximum frequent entity rows per top-error section.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON report path.")
    args = parser.parse_args()

    report = analyze_saved_model_errors(
        ROOT / "configs" / "config.yaml",
        split=args.split,
        model_dir=args.model_dir,
        ner_overrides={
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "entity_confidence_threshold": args.entity_threshold,
        },
        max_examples=args.max_examples,
        top_k=args.top_k,
    )

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ROOT / "outputs" / "logs" / f"ner_error_analysis_{args.split}_{timestamp}.json"
    elif not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Split:     {report['split']}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall:    {report['recall']:.4f}")
    print(f"F1:        {report['f1']:.4f}")
    print(f"TP/FP/FN:  {report['totals']['tp']}/{report['totals']['fp']}/{report['totals']['fn']}")
    print(f"Boundary errors sampled: {len(report['boundary_errors'])}")
    print(f"Label confusions sampled: {len(report['label_confusions'])}")
    print(f"Saved report to: {output_path}")

    print("\nTop false positives:")
    for row in report["top_false_positive_entities"][:5]:
        print(f"- {row['label']} {row['text']} x{row['count']}")

    print("\nTop false negatives:")
    for row in report["top_false_negative_entities"][:5]:
        print(f"- {row['label']} {row['text']} x{row['count']}")


if __name__ == "__main__":
    main()
