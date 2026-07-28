"""The document-retrieval preview endpoint.

Wiring test: the route must pass the query/collection/hybrid flag through to
rag.retrieve and serialize the resulting Sources. rag.retrieve is stubbed so no
embedder or vector store is needed.
"""
from __future__ import annotations

from app.routers import search
from app.schemas import Source


def test_document_search_passes_args_and_serializes(monkeypatch):
    captured = {}

    def fake_retrieve(q, collection=None, *, hybrid=True):
        captured["q"] = q
        captured["collection"] = collection
        captured["hybrid"] = hybrid
        return [Source(id="d1", document="a.pdf", text="chunk", page=3)]

    monkeypatch.setattr(search.rag, "retrieve", fake_retrieve)
    out = search.document_search("what is X", collection="kb", hybrid=False)

    assert captured == {"q": "what is X", "collection": "kb", "hybrid": False}
    assert isinstance(out, list)
    assert out[0]["id"] == "d1"
    assert out[0]["document"] == "a.pdf"
    assert out[0]["page"] == 3


def test_document_search_defaults(monkeypatch):
    monkeypatch.setattr(search.rag, "retrieve", lambda q, c=None, *, hybrid=True: [])
    assert search.document_search("q") == []
