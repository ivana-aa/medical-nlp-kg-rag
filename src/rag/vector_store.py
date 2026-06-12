"""FAISS vector store for local RAG retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.rag.splitter import TextChunk


class FaissVectorStore:
    """A small wrapper around FAISS IndexFlatIP with a NumPy fallback.

    FAISS is the preferred backend. The fallback exists because some Windows
    Python environments have trouble installing ``faiss-cpu``; it keeps the
    teaching demo usable while preserving the same public interface.
    """

    def __init__(self) -> None:
        self.index = None
        self.chunks: list[TextChunk] = []
        self.backend = "faiss"

    def build(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        """Create an in-memory index from chunks and vectors."""
        if len(chunks) == 0:
            raise ValueError("No chunks to index.")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings must have shape [num_chunks, dim].")

        dim = int(embeddings.shape[1])
        vectors = np.asarray(embeddings, dtype="float32")
        try:
            import faiss

            self.index = faiss.IndexFlatIP(dim)
            self.index.add(vectors)
            self.backend = "faiss"
        except ImportError:
            self.index = vectors
            self.backend = "numpy"
        self.chunks = chunks

    def save(self, output_dir: str | Path) -> None:
        """Persist FAISS index and chunk metadata."""
        if self.index is None:
            raise RuntimeError("Build or load an index before saving.")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss":
            import faiss

            faiss.write_index(self.index, str(output / "index.faiss"))
        else:
            np.save(output / "embeddings.npy", self.index)
        (output / "backend.txt").write_text(self.backend, encoding="utf-8")
        with (output / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump([chunk.to_dict() for chunk in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, input_dir: str | Path) -> "FaissVectorStore":
        """Load a saved FAISS index."""
        input_path = Path(input_dir)
        index_path = input_path / "index.faiss"
        numpy_path = input_path / "embeddings.npy"
        chunks_path = input_path / "chunks.json"
        if not chunks_path.exists() or (not index_path.exists() and not numpy_path.exists()):
            raise FileNotFoundError("Vector database not found. Run `python scripts/build_vector_db.py` first.")

        store = cls()
        if index_path.exists():
            import faiss

            store.index = faiss.read_index(str(index_path))
            store.backend = "faiss"
        else:
            store.index = np.load(numpy_path)
            store.backend = "numpy"
        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        store.chunks = [TextChunk(**item) for item in raw_chunks]
        return store

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[dict[str, object]]:
        """Return top-k chunks with similarity scores."""
        if self.index is None:
            raise RuntimeError("Index is not loaded.")

        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if self.backend == "faiss":
            scores, indices = self.index.search(query, top_k)
            score_row = scores[0]
            index_row = indices[0]
        else:
            matrix = np.asarray(self.index, dtype="float32")
            similarities = matrix @ query.reshape(-1)
            index_row = np.argsort(-similarities)[:top_k]
            score_row = similarities[index_row]

        results: list[dict[str, object]] = []
        for score, idx in zip(score_row, index_row):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append({"score": float(score), "chunk": chunk})
        return results
