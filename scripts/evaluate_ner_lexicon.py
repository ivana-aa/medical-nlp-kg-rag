"""Evaluate optional lexicon post-processing for the NER model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ner.lexicon_postprocess import evaluate_saved_model_with_lexicon


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lexicon-based NER post-processing.")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--add-missing", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate_saved_model_with_lexicon(
        ROOT / "configs" / "config.yaml",
        split=args.split,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        min_count=args.min_count,
        add_missing=args.add_missing,
    )

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "add_missing" if args.add_missing else "correct_only"
        output_path = (
            ROOT
            / "outputs"
            / "logs"
            / f"ner_lexicon_eval_{args.split}_{mode}_min{args.min_count}_{timestamp}.json"
        )
    elif not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    raw = result["raw"]
    post = result["postprocessed"]
    print(f"Split:        {result['split']}")
    print(f"Lexicon size: {result['lexicon_size']}")
    print(f"Min count:    {result['min_count']}")
    print(f"Add missing:  {result['add_missing']}")
    print(f"Corrected:    {result['corrected_entity_count']}")
    print(f"Added:        {result['added_entity_count']}")
    print(f"Raw F1:       {raw['f1']:.4f}  P={raw['precision']:.4f} R={raw['recall']:.4f}")
    print(f"Post F1:      {post['f1']:.4f}  P={post['precision']:.4f} R={post['recall']:.4f}")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
