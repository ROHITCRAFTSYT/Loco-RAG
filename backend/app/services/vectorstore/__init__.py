"""Vector store abstraction with a factory keyed on settings.vector_backend."""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.services.vectorstore.base import VectorRecord, VectorStore


@lru_cache
def get_vectorstore() -> VectorStore:
    if settings.vector_backend == "lancedb":
        from app.services.vectorstore.lance import LanceVectorStore

        return LanceVectorStore(settings.lancedb_dir)
    from app.services.vectorstore.chroma import ChromaVectorStore

    return ChromaVectorStore(settings.chroma_dir)


__all__ = ["VectorStore", "VectorRecord", "get_vectorstore"]
