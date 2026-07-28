"""End-to-end retrieve() orchestration with fake embedder + store.

The unit tests cover mmr_order/build_context in isolation; this drives the whole
retrieve() path (dense query -> select -> Source mapping) with the store and
embedder stubbed, proving MMR actually diversifies the returned chunks and the
plain path preserves rank order. No fastembed / Chroma / network.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.services import rag
from app.services.vectorstore.base import QueryHit

# Deterministic 2-D "embeddings": query on the x-axis; docA and its near-duplicate
# point almost the same way, docB is clearly different.
VECS = {
    "q": [1.0, 0.0],
    "docA": [0.9, 0.1],
    "docA_dup": [0.88, 0.12],
    "docB": [0.6, 0.5],
}


class FakeEmbedder:
    def embed(self, texts):
        return [VECS[t] for t in texts]


class FakeStore:
    def __init__(self, hits):
        self._hits = hits

    def query(self, collection, embedding, top_k):
        return list(self._hits)[:top_k]

    def all_texts(self, collection):  # pragma: no cover - hybrid disabled in tests
        return list(self._hits)


@pytest.fixture
def wired(monkeypatch):
    hits = [
        QueryHit(id="A", text="docA", score=0.9, metadata={"document": "a.pdf"}),
        QueryHit(id="A2", text="docA_dup", score=0.88, metadata={"document": "a.pdf"}),
        QueryHit(id="B", text="docB", score=0.5, metadata={"document": "b.pdf"}),
    ]
    monkeypatch.setattr(rag, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(rag, "get_vectorstore", lambda: FakeStore(hits))
    monkeypatch.setattr(settings, "retrieve_top_k", 8, raising=False)
    monkeypatch.setattr(settings, "rerank_top_n", 2, raising=False)
    monkeypatch.setattr(settings, "enable_rerank", False, raising=False)


def test_plain_retrieve_preserves_rank_order(wired, monkeypatch):
    monkeypatch.setattr(settings, "enable_mmr", False, raising=False)
    sources = rag.retrieve("q", "kb", hybrid=False)
    assert [s.id for s in sources] == ["A", "A2"]  # top-2 by rank
    assert sources[0].document == "a.pdf"


def test_mmr_retrieve_skips_the_near_duplicate(wired, monkeypatch):
    monkeypatch.setattr(settings, "enable_mmr", True, raising=False)
    monkeypatch.setattr(settings, "mmr_lambda", 0.2, raising=False)  # favour diversity
    sources = rag.retrieve("q", "kb", hybrid=False)
    ids = [s.id for s in sources]
    assert ids == ["A", "B"]  # keeps the top hit, then the diverse one, not the dup


def test_retrieve_maps_metadata_into_sources(wired, monkeypatch):
    monkeypatch.setattr(settings, "enable_mmr", False, raising=False)
    sources = rag.retrieve("q", "kb", hybrid=False)
    s = sources[0]
    assert s.kind == "document"
    assert s.collection == "kb"
    assert isinstance(s.score, float)
