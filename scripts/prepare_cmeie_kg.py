"""Convert CMeIE relation extraction data into the project KG CSV format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kg.cmeie_importer import convert_cmeie_to_kg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CMeIE JSON/JSONL spo_list records to a medical KG CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "raw" / "CMeIE",
        help="CMeIE file or directory containing JSON/JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "medical_kg" / "medical_kg_cmeie.csv",
        help="Output CSV path. The main demo KG is not overwritten by default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CMeIE records to convert.",
    )
    args = parser.parse_args()

    stats = convert_cmeie_to_kg(args.input, args.output, limit=args.limit)
    print(f"records: {stats.records}")
    print(f"spo_items: {stats.spo_items}")
    print(f"valid_triples: {stats.valid_triples}")
    print(f"written_triples: {stats.written_triples}")
    print(f"output: {stats.output_path}")


if __name__ == "__main__":
    main()
