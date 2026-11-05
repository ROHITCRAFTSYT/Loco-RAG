"""LLM provider failures become actionable LLMErrors.

httpx is driven through a MockTransport so every failure mode (connection
refused, timeout, 404 for a missing model, 5xx, malformed body) is reproduced
without a running Ollama / LM Studio.
"""
from __future__ import annotations

import httpx
import pytest

from app.services.llm import base
from app.services.llm.base import LLMError, LLMProvider


def _patch_transport(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(base.httpx, "AsyncClient", factory)


def _provider():
    return LLMProvider("ollama", "http://ollama.test:11434", native_ollama=True)


async def test_connection_refused_names_provider_and_url(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("All connection attempts failed", request=request)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LLMError) as ei:
        await _provider().complete([{"role": "user", "content": "hi"}], model="llama3.1")
    msg = str(ei.value)
    assert "Cannot reach ollama at http://ollama.test:11434" in msg
    assert ei.value.provider == "ollama"


async def test_missing_model_404_suggests_pull(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(404, json={"error": "not found"}))
    with pytest.raises(LLMError) as ei:
        await _provider().complete([], model="qwen9")
    assert "ollama pull qwen9" in str(ei.value)
    assert ei.value.status_code == 404


async def test_timeout(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LLMError, match="timed out"):
        await _provider().chat_with_tools([], [], model="m")


async def test_server_error_status(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="HTTP 500") as ei:
        await _provider().complete([], model="m")
    assert ei.value.status_code == 500


async def test_malformed_body(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(200, json={"nope": True}))
    with pytest.raises(LLMError, match="unexpected response"):
        await _provider().complete([], model="m")


async def test_stream_errors_are_translated(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LLMError, match="Cannot reach"):
        async for _ in _provider().stream_chat([], model="m"):
            pass


async def test_stream_happy_path_still_works(monkeypatch):
    body = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    _patch_transport(monkeypatch, lambda r: httpx.Response(200, text=body))
    chunks = [c async for c in _provider().stream_chat([], model="m")]
    assert "".join(c.delta for c in chunks) == "Hello"
    assert chunks[-1].done
    assert any(c.completion_tokens == 2 for c in chunks)


async def test_list_models_errors_are_translated(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LLMError):
        await _provider().list_models()


def test_unknown_provider_name():
    with pytest.raises(ValueError, match="Unknown provider"):
        base.get_provider("vllm")
