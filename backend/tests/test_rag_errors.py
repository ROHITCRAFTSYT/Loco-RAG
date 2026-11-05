"""Embedding / vector-store failures surface as RetrievalError with the stage."""
from __future__ import annotations

import pytest

from app.config import settings
from app.services import rag


class BoomEmbedder:
    def embed(self, texts):
        raise ConnectionError("embedding server down")


class OkEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class BoomStore:
    def query(self, collection, embedding, top_k):
        raise RuntimeError("index corrupted")

    def all_texts(self, collection):
        return []


@pytest.fixture(autouse=True)
def _plain(monkeypatch):
    monkeypatch.setattr(settings, "enable_rerank", False, raising=False)
    monkeypatch.setattr(settings, "enable_mmr", False, raising=False)


def test_embedding_failure(monkeypatch):
    monkeypatch.setattr(rag, "get_embedder", lambda: BoomEmbedder())
    with pytest.raises(rag.RetrievalError) as ei:
        rag.retrieve("q", "kb")
    assert ei.value.stage == "embedding"
    assert "embedding server down" in str(ei.value)


def test_vector_store_failure(monkeypatch):
    monkeypatch.setattr(rag, "get_embedder", lambda: OkEmbedder())
    monkeypatch.setattr(rag, "get_vectorstore", lambda: BoomStore())
    with pytest.raises(rag.RetrievalError) as ei:
        rag.retrieve("q", "kb", hybrid=False)
    assert ei.value.stage == "vector store"


def test_store_construction_failure(monkeypatch):
    def broken():
        raise PermissionError("read-only volume")

    monkeypatch.setattr(rag, "get_embedder", lambda: OkEmbedder())
    monkeypatch.setattr(rag, "get_vectorstore", broken)
    with pytest.raises(rag.RetrievalError, match="read-only volume"):
        rag.retrieve("q", "kb")
