"""Create a smaller BIO dataset from a larger NER corpus.

This is useful on CPU-only laptops: keep the full public dataset under
``data/raw`` or ``data/processed``, then train a resume/demo checkpoint on a
real-data subset that finishes in minutes instead of hours.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_bio_examples(path: Path) -> list[list[tuple[str, str]]]:
    """Read a BIO file into sentence-level examples."""
    examples: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            if current:
                examples.append(current)
                current = []
            continue
        parts = line.split()
        if len(parts) >= 2:
            current.append((parts[0], parts[-1]))
    if current:
        examples.append(current)
    return examples


def write_bio_examples(path: Path, examples: list[list[tuple[str, str]]]) -> None:
    """Write sentence-level examples back to BIO format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for example in examples:
            for char, label in example:
                f.write(f"{char} {label}\n")
            f.write("\n")


def sample_split(source_file: Path, target_file: Path, limit: int | None) -> int:
    """Copy at most ``limit`` examples from one BIO split."""
    examples = read_bio_examples(source_file)
    if limit is not None and limit > 0:
        examples = examples[:limit]
    write_bio_examples(target_file, examples)
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a smaller BIO NER dataset.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "processed" / "ner_demo")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "data" / "processed" / "ner_train")
    parser.add_argument("--train", type=int, default=500, help="Maximum train examples.")
    parser.add_argument("--dev", type=int, default=150, help="Maximum dev examples.")
    parser.add_argument("--test", type=int, default=150, help="Maximum test examples.")
    args = parser.parse_args()

    split_limits = {"train": args.train, "dev": args.dev, "test": args.test}
    for split, limit in split_limits.items():
        count = sample_split(args.source_dir / f"{split}.bio", args.target_dir / f"{split}.bio", limit)
        print(f"{split}: {count} examples -> {args.target_dir / f'{split}.bio'}")


if __name__ == "__main__":
    main()
