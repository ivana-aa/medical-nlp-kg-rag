"""Embedding wrapper built on sentence-transformers."""

from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    """Generate dense vectors for Chinese medical text chunks."""

    def __init__(self, model_name: str, normalize_embeddings: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into float32 vectors."""
        if not texts:
            return np.empty((0, 0), dtype="float32")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings, dtype="float32")
