"""Build a local FAISS vector database from medical knowledge files."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.embedder import SentenceTransformerEmbedder
from src.rag.loader import load_knowledge_base
from src.rag.splitter import split_documents
from src.rag.vector_store import FaissVectorStore


def main() -> None:
    config_path = ROOT / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rag_config = config["rag"]

    docs = load_knowledge_base(ROOT / rag_config["knowledge_base_dir"])
    chunks = split_documents(
        docs,
        chunk_size=int(rag_config["chunk_size"]),
        chunk_overlap=int(rag_config["chunk_overlap"]),
    )
    if not chunks:
        raise RuntimeError("No chunks generated. Add txt/md/csv/json files to data/knowledge_base.")

    print(f"Loaded documents: {len(docs)}")
    print(f"Generated chunks: {len(chunks)}")

    embedder = SentenceTransformerEmbedder(
        rag_config["embedding_model"],
        normalize_embeddings=bool(rag_config.get("normalize_embeddings", True)),
    )
    embeddings = embedder.encode([chunk.content for chunk in chunks])

    store = FaissVectorStore()
    store.build(chunks, embeddings)
    output_dir = ROOT / rag_config["vector_store_dir"]
    store.save(output_dir)
    print(f"Vector database saved to: {output_dir}")
    print(f"Vector backend: {store.backend}")


if __name__ == "__main__":
    main()
