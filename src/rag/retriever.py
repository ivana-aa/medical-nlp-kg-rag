"""Retriever that combines dense vector search and optional keyword search."""

from __future__ import annotations

from src.rag.bm25 import BM25TextIndex, reciprocal_rank_fusion
from src.rag.embedder import SentenceTransformerEmbedder
from src.rag.vector_store import FaissVectorStore


class MedicalRetriever:
    """Retrieve relevant medical text chunks for a question."""

    def __init__(
        self,
        vector_store_dir: str,
        embedding_model: str,
        top_k: int = 4,
        normalize_embeddings: bool = True,
        retrieval_mode: str = "vector",
        bm25_top_k: int | None = None,
        rrf_rank_constant: int = 60,
    ) -> None:
        retrieval_mode = retrieval_mode.lower().strip()
        if retrieval_mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("retrieval_mode must be one of: vector, bm25, hybrid.")

        self.embedder = SentenceTransformerEmbedder(
            embedding_model,
            normalize_embeddings=normalize_embeddings,
        )
        self.store = FaissVectorStore.load(vector_store_dir)
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.bm25_top_k = bm25_top_k or max(top_k * 2, top_k)
        self.rrf_rank_constant = rrf_rank_constant
        self.bm25_index = BM25TextIndex(self.store.chunks) if retrieval_mode in {"bm25", "hybrid"} else None

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, object]]:
        """Search the local vector database, keyword index, or fused hybrid index."""
        effective_top_k = top_k or self.top_k
        if self.retrieval_mode == "bm25":
            if self.bm25_index is None:
                return []
            return self.bm25_index.search(question, top_k=effective_top_k)

        vector_top_k = max(effective_top_k, self.bm25_top_k) if self.retrieval_mode == "hybrid" else effective_top_k
        query_embedding = self.embedder.encode([question])
        vector_results = self.store.search(query_embedding, top_k=vector_top_k)
        if self.retrieval_mode == "vector":
            return vector_results

        bm25_results = self.bm25_index.search(question, top_k=self.bm25_top_k) if self.bm25_index else []
        return reciprocal_rank_fusion(
            [vector_results, bm25_results],
            top_k=effective_top_k,
            rank_constant=self.rrf_rank_constant,
            source_names=["vector", "bm25"],
        )
