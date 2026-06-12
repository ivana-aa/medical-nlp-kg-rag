"""Download CMeIE from Hugging Face Hub and optionally convert it to KG CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kg.cmeie_hf import CMeIE_HF_FILES, download_cmeie_files
from src.kg.cmeie_importer import convert_cmeie_to_kg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Aunderline/CMeIE JSONL files from Hugging Face Hub."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "CMeIE",
        help="Directory for downloaded CMeIE JSONL files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(CMeIE_HF_FILES),
        help="Dataset files to download from the Hugging Face repository.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for each file request.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert downloaded JSONL files into a separate KG CSV after download.",
    )
    parser.add_argument(
        "--kg-output",
        type=Path,
        default=ROOT / "data" / "medical_kg" / "medical_kg_cmeie.csv",
        help="Output path for the optional CMeIE KG CSV. The main demo KG is not overwritten.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CMeIE records to convert.",
    )
    args = parser.parse_args()

    paths = download_cmeie_files(
        args.output_dir,
        files=args.files,
        skip_existing=not args.overwrite,
        timeout=args.timeout,
    )

    print("Downloaded or reused files:")
    for path in paths:
        print(f"- {path}")

    if args.convert:
        stats = convert_cmeie_to_kg(args.output_dir, args.kg_output, limit=args.limit)
        print("Converted CMeIE to KG CSV:")
        print(f"records: {stats.records}")
        print(f"spo_items: {stats.spo_items}")
        print(f"valid_triples: {stats.valid_triples}")
        print(f"written_triples: {stats.written_triples}")
        print(f"output: {stats.output_path}")


if __name__ == "__main__":
    main()
