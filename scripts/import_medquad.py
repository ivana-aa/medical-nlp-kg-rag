"""Import a small, traceable MedQuAD sample into the local RAG knowledge base.

The full MedQuAD repository contains 47,457 medical QA pairs. Pulling all of it
is unnecessary for this beginner project, so this script downloads a controlled
sample from public GitHub XML files, parses question-answer pairs, and writes a
JSON knowledge file with source URL and license metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
GITHUB_API = "https://api.github.com/repos/abachaa/MedQuAD/contents"
RAW_BASE = "https://raw.githubusercontent.com/abachaa/MedQuAD/master"
DEFAULT_FOLDERS = [
    "1_CancerGov_QA",
    "2_GARD_QA",
    "5_NIDDK_QA",
    "6_NINDS_QA",
    "8_NHLBI_QA_XML",
    "9_CDC_QA",
]


def normalize_text(text: str | None) -> str:
    """Collapse whitespace from XML text."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def get_json(url: str) -> Any:
    """Fetch JSON from GitHub API."""
    response = requests.get(url, headers={"User-Agent": "medical-nlp-project"}, timeout=30)
    response.raise_for_status()
    return response.json()


def get_text(url: str) -> str:
    """Fetch UTF-8 text from GitHub raw content."""
    response = requests.get(url, headers={"User-Agent": "medical-nlp-project"}, timeout=30)
    response.raise_for_status()
    return response.text


def list_xml_files(folder: str, limit: int) -> list[dict[str, str]]:
    """List XML files in one MedQuAD folder through the GitHub contents API."""
    items = get_json(f"{GITHUB_API}/{folder}?ref=master")
    files = [
        item
        for item in items
        if item.get("type") == "file" and str(item.get("name", "")).lower().endswith(".xml")
    ]
    return sorted(files, key=lambda item: item["name"])[:limit]


def parse_medquad_xml(xml_text: str, github_path: str, raw_url: str) -> list[dict[str, str]]:
    """Parse one MedQuAD XML document into QA records."""
    root = ET.fromstring(xml_text)
    focus = normalize_text(root.findtext("Focus"))
    document_source = root.attrib.get("source", "MedQuAD")
    document_url = root.attrib.get("url", raw_url)

    records: list[dict[str, str]] = []
    for qa_pair in root.findall(".//QAPair"):
        question_node = qa_pair.find("Question")
        answer_node = qa_pair.find("Answer")
        question = normalize_text(question_node.text if question_node is not None else "")
        answer = normalize_text(answer_node.text if answer_node is not None else "")
        if not question or not answer:
            continue
        records.append(
            {
                "focus": focus,
                "question": question,
                "answer": answer,
                "qtype": question_node.attrib.get("qtype", "") if question_node is not None else "",
                "qid": question_node.attrib.get("qid", "") if question_node is not None else "",
                "document_source": document_source,
                "document_url": document_url,
                "github_path": github_path,
                "raw_url": raw_url,
            }
        )
    return records


def build_knowledge_item(record: dict[str, str], answer_char_limit: int) -> dict[str, str]:
    """Convert one MedQuAD QA record into the project's JSON knowledge format."""
    answer = record["answer"][:answer_char_limit].strip()
    content = (
        f"Topic: {record['focus']}\n"
        f"Question: {record['question']}\n"
        f"Answer: {answer}"
    )
    return {
        "title": f"MedQuAD - {record['focus'] or record['qid']}",
        "content": content,
        "source_dataset": "MedQuAD",
        "source_id": "medquad",
        "source_url": record["document_url"] or record["raw_url"],
        "license": "CC BY 4.0",
        "language": "English",
        "question_type": record["qtype"],
        "github_path": record["github_path"],
        "raw_url": record["raw_url"],
        "note": "Original MedQuAD QA pair imported from public GitHub XML.",
    }


def import_medquad(
    folders: list[str],
    max_files_per_folder: int,
    max_pairs: int,
    max_pairs_per_folder: int,
    answer_char_limit: int,
    output_path: Path,
    raw_dir: Path,
) -> int:
    """Download a MedQuAD sample and write project knowledge JSON."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    for folder in folders:
        if len(records) >= max_pairs:
            break
        files = list_xml_files(folder, max_files_per_folder)
        folder_count = 0
        print(f"{folder}: {len(files)} XML files selected")
        for file_item in files:
            if len(records) >= max_pairs or folder_count >= max_pairs_per_folder:
                break
            github_path = file_item["path"]
            raw_url = file_item.get("download_url") or f"{RAW_BASE}/{github_path}"
            xml_text = get_text(raw_url)
            local_file = raw_dir / github_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            local_file.write_text(xml_text, encoding="utf-8")
            parsed = parse_medquad_xml(xml_text, github_path, raw_url)
            for record in parsed:
                records.append(record)
                folder_count += 1
                if len(records) >= max_pairs or folder_count >= max_pairs_per_folder:
                    break
        print(f"{folder}: imported {folder_count} QA pairs")

    knowledge_items = [
        build_knowledge_item(record, answer_char_limit=answer_char_limit)
        for record in records
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(knowledge_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Imported MedQuAD QA pairs: {len(knowledge_items)}")
    print(f"Knowledge file written to: {output_path}")
    print(f"Raw XML cache written to: {raw_dir}")
    return len(knowledge_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a small MedQuAD sample for RAG.")
    parser.add_argument("--folders", nargs="*", default=DEFAULT_FOLDERS, help="MedQuAD folders to sample.")
    parser.add_argument("--max-files-per-folder", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=30)
    parser.add_argument("--max-pairs-per-folder", type=int, default=8)
    parser.add_argument("--answer-char-limit", type=int, default=900)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "knowledge_base" / "medquad_qa_sample.json",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "MedQuAD")
    args = parser.parse_args()

    try:
        count = import_medquad(
            folders=args.folders,
            max_files_per_folder=args.max_files_per_folder,
            max_pairs=args.max_pairs,
            max_pairs_per_folder=args.max_pairs_per_folder,
            answer_char_limit=args.answer_char_limit,
            output_path=args.output,
            raw_dir=args.raw_dir,
        )
    except requests.RequestException as exc:
        print(f"Network error while importing MedQuAD: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except ET.ParseError as exc:
        print(f"XML parse error while importing MedQuAD: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if count == 0:
        raise SystemExit("No MedQuAD QA pairs imported.")


if __name__ == "__main__":
    main()
