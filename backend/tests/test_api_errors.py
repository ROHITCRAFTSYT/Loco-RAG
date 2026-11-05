"""HTTP-level error handling, driven through the real FastAPI app.

Covers: request-id propagation, the JSON 500 / 502 / 503 mappings, collection
name validation on upload, and the chat route degrading (instead of 500-ing)
when retrieval or web search fails. Model and retrieval calls are stubbed; the
SQLite DB lives in the temp dir set up by conftest.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import chat as chat_router
from app.routers import search as search_router
from app.services.llm import LLMError
from app.services.llm.base import StreamChunk
from app.services.rag import RetrievalError


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_has_generated_request_id(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert len(r.headers["X-Request-ID"]) >= 8


def test_safe_incoming_request_id_is_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "edge-proxy.42"})
    assert r.headers["X-Request-ID"] == "edge-proxy.42"


def test_unsafe_incoming_request_id_is_replaced(client):
    r = client.get("/health", headers={"X-Request-ID": "x\" injected=1 " + "a" * 80})
    assert r.headers["X-Request-ID"] != "x\" injected=1 " + "a" * 80
    assert " " not in r.headers["X-Request-ID"]


def test_retrieval_error_maps_to_503(client, monkeypatch):
    def boom(q, collection=None, *, hybrid=True):
        raise RetrievalError("Document search failed during embedding (boom)", stage="embedding")

    monkeypatch.setattr(search_router.rag, "retrieve", boom)
    r = client.get("/api/search/documents", params={"q": "x"})
    assert r.status_code == 503
    body = r.json()
    assert body["stage"] == "embedding"
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_llm_error_maps_to_502(client, monkeypatch):
    async def boom(q):
        raise LLMError("Cannot reach ollama at http://x", provider="ollama")

    monkeypatch.setattr(search_router.websearch, "search", boom)
    r = client.get("/api/search", params={"q": "x"})
    assert r.status_code == 502
    assert r.json()["provider"] == "ollama"


def test_unhandled_error_is_json_500_with_request_id(client, monkeypatch):
    async def boom(q):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(search_router.websearch, "search", boom)
    r = client.get("/api/search", params={"q": "x"})
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error"
    assert "secret" not in r.text
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


@pytest.mark.parametrize("name", ["a", "bad name", "-lead", "trail-", "x..y", "chat-123"])
def test_upload_rejects_bad_collection_names(client, name):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"collection": name},
    )
    assert r.status_code == 422, name


# ---- chat degradation -------------------------------------------------------


class FakeLLM:
    def __init__(self, fail: Exception | None = None):
        self.fail = fail
        self.seen_messages = None

    async def stream_chat(self, messages, **kw):
        self.seen_messages = messages
        if self.fail:
            raise self.fail
        yield StreamChunk(delta="ok")
        yield StreamChunk(done=True)


def _new_conversation(client, **fields) -> str:
    r = client.post("/api/conversations", json=fields)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _events(text: str) -> list[str]:
    return [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("event:")]


def test_chat_degrades_when_retrieval_fails(client, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(chat_router, "get_provider", lambda name: llm)

    def boom(q, collections):
        raise RetrievalError(
            "Document search failed during vector store (OSError: disk full)",
            stage="vector store",
            cause="OSError: disk full",
        )

    monkeypatch.setattr(chat_router.rag, "retrieve", boom)
    cid = _new_conversation(client, rag_enabled=True)
    r = client.post(f"/api/conversations/{cid}/chat", json={"content": "hi"})

    assert r.status_code == 200
    events = _events(r.text)
    assert "warning" in events and "token" in events and events[-1] == "done"
    # The model is told retrieval failed so it can say so to the user.
    system_notes = [m["content"] for m in llm.seen_messages if m["role"] == "system"]
    assert any("Document search was unavailable" in n for n in system_notes)

    # ...and the warning is persisted, so the UI banner survives a reload.
    msgs = client.get(f"/api/conversations/{cid}/messages").json()
    assistant = [m for m in msgs if m["role"] == "assistant"][-1]
    assert assistant["warnings"] == [
        "Document search was unavailable for this message (vector store failed: OSError: disk full)."
    ]
    assert "warnings" not in assistant["usage"]
    user = [m for m in msgs if m["role"] == "user"][-1]
    assert user["warnings"] is None


def test_chat_degrades_when_web_search_fails(client, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(chat_router, "get_provider", lambda name: llm)

    async def boom(q):
        raise ConnectionError("ratelimited")

    monkeypatch.setattr(chat_router.websearch, "search", boom)
    cid = _new_conversation(client, rag_enabled=False, web_search=True)
    r = client.post(f"/api/conversations/{cid}/chat", json={"content": "hi"})
    assert r.status_code == 200
    assert "warning" in _events(r.text)


def test_chat_llm_error_streams_actionable_message(client, monkeypatch):
    llm = FakeLLM(fail=LLMError("Cannot reach ollama at http://x. Is it running?", provider="ollama"))
    monkeypatch.setattr(chat_router, "get_provider", lambda name: llm)
    cid = _new_conversation(client, rag_enabled=False)
    r = client.post(f"/api/conversations/{cid}/chat", json={"content": "hi"})
    assert "error" in _events(r.text)
    assert "Cannot reach ollama" in r.text


def test_chat_unexpected_error_is_generic(client, monkeypatch):
    llm = FakeLLM(fail=RuntimeError("internal detail"))
    monkeypatch.setattr(chat_router, "get_provider", lambda name: llm)
    cid = _new_conversation(client, rag_enabled=False)
    r = client.post(f"/api/conversations/{cid}/chat", json={"content": "hi"})
    assert "Generation failed (RuntimeError)" in r.text
    assert "internal detail" not in r.text


def test_chat_unknown_provider_is_400_and_persists_nothing(client):
    cid = _new_conversation(client, rag_enabled=False)
    r = client.post(f"/api/conversations/{cid}/chat", json={"content": "hi", "provider": "vllm"})
    assert r.status_code == 400
    assert client.get(f"/api/conversations/{cid}/messages").json() == []
