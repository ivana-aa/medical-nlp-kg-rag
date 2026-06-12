"""BM25 keyword retrieval and rank fusion utilities for local RAG."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.rag.splitter import TextChunk


_ALNUM_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize mixed Chinese/English medical text for lightweight BM25."""
    lowered = text.lower()
    tokens = _ALNUM_RE.findall(lowered)

    for span in _CJK_RE.findall(lowered):
        chars = list(span)
        tokens.extend(chars)
        for n in (2, 3, 4):
            if len(chars) < n:
                continue
            tokens.extend("".join(chars[idx : idx + n]) for idx in range(len(chars) - n + 1))
    return tokens


class BM25TextIndex:
    """A small in-memory BM25 index over RAG text chunks."""

    def __init__(
        self,
        chunks: Sequence[TextChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.document_lengths: list[int] = []

        for chunk in self.chunks:
            tokens = tokenize_for_bm25(chunk.content)
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(sum(frequencies.values()))
            self.document_frequencies.update(frequencies.keys())

        total_length = sum(self.document_lengths)
        self.average_document_length = total_length / len(self.document_lengths) if self.document_lengths else 0.0

    def search(self, query: str, top_k: int = 4) -> list[dict[str, object]]:
        """Return top-k chunks ranked by BM25 score."""
        if top_k <= 0 or not self.chunks:
            return []

        query_terms = set(tokenize_for_bm25(query))
        if not query_terms:
            return []

        scored: list[tuple[float, str, TextChunk]] = []
        for idx, chunk in enumerate(self.chunks):
            score = self._score_document(query_terms, idx)
            if score <= 0:
                continue
            scored.append((score, chunk.chunk_id, chunk))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{"score": float(score), "chunk": chunk, "retrieval": "bm25"} for score, _, chunk in scored[:top_k]]

    def _score_document(self, query_terms: set[str], doc_idx: int) -> float:
        frequencies = self.term_frequencies[doc_idx]
        doc_length = self.document_lengths[doc_idx]
        if doc_length == 0 or self.average_document_length == 0:
            return 0.0

        score = 0.0
        total_docs = len(self.chunks)
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self.document_frequencies[term]
            idf = math.log(1.0 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * doc_length / self.average_document_length
            )
            score += idf * (term_frequency * (self.k1 + 1.0) / denominator)
        return score


def reciprocal_rank_fusion(
    result_groups: Iterable[Sequence[Mapping[str, Any]]],
    top_k: int = 4,
    rank_constant: int = 60,
    source_names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Fuse ranked retrieval results using Reciprocal Rank Fusion."""
    if top_k <= 0:
        return []
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive.")

    fused_scores: defaultdict[str, float] = defaultdict(float)
    best_ranks: dict[str, int] = {}
    chunks_by_id: dict[str, object] = {}
    component_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)

    for group_idx, results in enumerate(result_groups):
        source_name = source_names[group_idx] if source_names and group_idx < len(source_names) else f"source_{group_idx}"
        for rank, result in enumerate(results, start=1):
            chunk = result.get("chunk")
            if chunk is None:
                continue
            chunk_id = str(getattr(chunk, "chunk_id", id(chunk)))
            fused_scores[chunk_id] += 1.0 / (rank_constant + rank)
            best_ranks[chunk_id] = min(best_ranks.get(chunk_id, rank), rank)
            chunks_by_id.setdefault(chunk_id, chunk)
            component_scores[chunk_id][source_name] = float(result.get("score", 0.0))

    ranked_ids = sorted(fused_scores, key=lambda chunk_id: (-fused_scores[chunk_id], best_ranks[chunk_id], chunk_id))
    fused: list[dict[str, object]] = []
    for chunk_id in ranked_ids[:top_k]:
        fused.append(
            {
                "score": float(fused_scores[chunk_id]),
                "chunk": chunks_by_id[chunk_id],
                "retrieval": "hybrid",
                "component_scores": component_scores[chunk_id],
            }
        )
    return fused
