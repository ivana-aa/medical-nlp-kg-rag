"""Load local medical knowledge documents for the RAG demo."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """A source document before chunking."""

    content: str
    source: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_knowledge_base(root_dir: str | Path) -> list[Document]:
    """Load txt/md/csv/json files from a local knowledge base directory."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            documents.extend(_load_text_file(path, root))
        elif suffix == ".csv":
            documents.extend(_load_csv_file(path, root))
        elif suffix == ".json":
            documents.extend(_load_json_file(path, root))

    return [doc for doc in documents if doc.content.strip()]


def _relative_source(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _load_text_file(path: Path, root: Path) -> list[Document]:
    content = path.read_text(encoding="utf-8").strip()
    return [
        Document(
            content=content,
            source=_relative_source(path, root),
            metadata={"file_type": path.suffix.lower().lstrip(".")},
        )
    ]


def _load_csv_file(path: Path, root: Path) -> list[Document]:
    docs: list[Document] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            parts = [f"{key}: {value}" for key, value in row.items() if value]
            docs.append(
                Document(
                    content="\n".join(parts),
                    source=f"{_relative_source(path, root)}#row={row_idx}",
                    metadata={"file_type": "csv", "row": row_idx},
                )
            )
    return docs


def _load_json_file(path: Path, root: Path) -> list[Document]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else [raw]
    docs: list[Document] = []

    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            title = item.get("title") or item.get("name") or f"item_{idx}"
            content = item.get("content") or item.get("text") or json.dumps(item, ensure_ascii=False)
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"content", "text"}
                and isinstance(value, (str, int, float, bool))
            }
        else:
            title = f"item_{idx}"
            content = str(item)
            metadata = {}
        docs.append(
            Document(
                content=str(content),
                source=f"{_relative_source(path, root)}#{title}",
                metadata={"file_type": "json", "item": idx, "title": str(title), **metadata},
            )
        )

    return docs
