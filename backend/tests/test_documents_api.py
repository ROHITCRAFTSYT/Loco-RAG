"""Document lifecycle through the API against a real (embedded) Chroma store.

upload -> background ingest -> listed as ready -> retrievable via the search
preview -> delete removes it from both SQLite and the index. Only the embedder is
faked (a deterministic bag-of-words hash), so parsing, chunking, the vector
store and hybrid retrieval are all real.
"""
from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import ingest, rag
from app.services.vectorstore.chroma import ChromaVectorStore

DIM = 32


class HashEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            out.append(v if any(v) else [1.0] + [0.0] * (DIM - 1))
        return out


@pytest.fixture
def client(monkeypatch, tmp_path):
    store = ChromaVectorStore(str(tmp_path / "chroma"))
    for mod in (ingest, rag):
        monkeypatch.setattr(mod, "get_embedder", lambda: HashEmbedder())
        monkeypatch.setattr(mod, "get_vectorstore", lambda: store)
    from app.routers import documents

    monkeypatch.setattr(documents, "get_vectorstore", lambda: store)
    monkeypatch.setattr(settings, "enable_rerank", False, raising=False)
    monkeypatch.setattr(settings, "enable_mmr", False, raising=False)
    with TestClient(app) as c:
        yield c


def test_upload_ingest_search_delete(client):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("tower.md", b"The Eiffel Tower is 330 metres tall.", "text/markdown")},
        data={"collection": "kb-test"},
    )
    assert r.status_code == 200
    doc_id = r.json()["document_id"]

    # BackgroundTasks run before TestClient returns, so ingestion is done.
    docs = client.get("/api/documents", params={"collection": "kb-test"}).json()
    assert docs[0]["status"] == "ready" and docs[0]["chunk_count"] == 1

    hits = client.get(
        "/api/search/documents", params={"q": "how tall is the eiffel tower", "collection": "kb-test"}
    ).json()
    assert hits[0]["document"] == "tower.md"
    assert hits[0]["content_hash"]

    stats = client.get("/api/documents/stats").json()
    assert stats["kb-test"]["ready"] == 1
    assert "kb-test" in client.get("/api/documents/collections").json()

    assert client.delete(f"/api/documents/{doc_id}").json() == {"ok": True}
    assert client.get("/api/documents", params={"collection": "kb-test"}).json() == []
    assert client.get(
        "/api/search/documents", params={"q": "eiffel", "collection": "kb-test"}
    ).json() == []


def test_empty_file_is_marked_error(client):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("blank.txt", b"   ", "text/plain")},
        data={"collection": "kb-test"},
    )
    doc = [d for d in client.get("/api/documents").json() if d["id"] == r.json()["document_id"]][0]
    assert doc["status"] == "error" and doc["error"] == "No extractable text"


def test_upload_over_size_limit_is_413(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1, raising=False)
    r = client.post(
        "/api/documents/upload",
        files={"file": ("big.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
        data={"collection": "kb-test"},
    )
    assert r.status_code == 413


def test_attach_is_searchable_in_its_conversation_collection(client):
    cid = client.post("/api/conversations", json={}).json()["id"]
    r = client.post(
        "/api/documents/attach",
        files={"file": ("note.txt", b"The launch code is pineapple.", "text/plain")},
        data={"conversation_id": cid},
    )
    body = r.json()
    assert body["status"] == "ready" and body["collection"] == f"chat-{cid}"
    # ephemeral collections are hidden from the KB picker
    assert body["collection"] not in client.get("/api/documents/collections").json()
