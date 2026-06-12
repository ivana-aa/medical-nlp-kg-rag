"""Text splitting utilities for local medical documents."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from src.rag.loader import Document


@dataclass
class TextChunk:
    """A retrievable text chunk with source metadata."""

    content: str
    source: str
    chunk_id: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace while preserving Chinese punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def split_documents(
    documents: list[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> list[TextChunk]:
    """Split documents into overlapping character chunks.

    Character-based splitting is simple and works well enough for a small
    Chinese demo. For production, consider semantic splitters or section-aware
    parsing.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[TextChunk] = []
    for doc_idx, doc in enumerate(documents):
        text = normalize_text(doc.content)
        start = 0
        local_idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"doc{doc_idx:04d}_chunk{local_idx:04d}"
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        source=doc.source,
                        chunk_id=chunk_id,
                        metadata={**doc.metadata, "start": start, "end": end},
                    )
                )
                local_idx += 1
            if end == len(text):
                break
            start = max(0, end - chunk_overlap)

    return chunks
