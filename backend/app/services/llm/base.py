"""OpenAI-compatible LLM provider used for both Ollama and LM Studio."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.config import settings

# Heuristic: model name substrings that indicate vision capability.
_VISION_HINTS = ("llava", "vision", "bakllava", "moondream", "minicpm-v", "qwen2-vl", "llama3.2-vision")


@dataclass
class StreamChunk:
    delta: str = ""
    done: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider:
    """Thin async wrapper over an OpenAI-compatible /v1 endpoint."""

    def __init__(self, name: str, base_url: str, *, native_ollama: bool = False):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.native_ollama = native_ollama

    @property
    def _v1(self) -> str:
        # Ollama's OpenAI-compatible routes live under /v1 on the root URL.
        if self.native_ollama:
            return f"{self.base_url}/v1"
        return self.base_url

    async def list_models(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if self.native_ollama:
                r = await client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                models = r.json().get("models", [])
                return [
                    {
                        "id": m["name"],
                        "provider": self.name,
                        "supports_vision": _supports_vision(m.get("name", "")),
                    }
                    for m in models
                ]
            r = await client.get(f"{self._v1}/models")
            r.raise_for_status()
            data = r.json().get("data", [])
            return [
                {
                    "id": m["id"],
                    "provider": self.name,
                    "supports_vision": _supports_vision(m.get("id", "")),
                }
                for m in data
            ]

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            async with client.stream(
                "POST", f"{self._v1}/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        yield StreamChunk(done=True)
                        return
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {}).get("content")
                        if delta:
                            yield StreamChunk(delta=delta)
                    usage = obj.get("usage")
                    if usage:
                        yield StreamChunk(
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                        )

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        model: str,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Non-streaming completion that may return tool_calls. Returns the raw
        assistant message dict (with optional ``tool_calls``)."""
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            r = await client.post(f"{self._v1}/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]

    async def complete(
        self, messages: list[dict[str, Any]], *, model: str, temperature: float = 0.3
    ) -> str:
        """Non-streaming completion (used for summarization / titling)."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            r = await client.post(f"{self._v1}/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


def _supports_vision(model_id: str) -> bool:
    mid = model_id.lower()
    return any(h in mid for h in _VISION_HINTS)


def get_provider(name: str | None = None) -> LLMProvider:
    name = name or settings.default_provider
    if name == "ollama":
        return LLMProvider("ollama", settings.ollama_base_url, native_ollama=True)
    if name == "lmstudio":
        return LLMProvider("lmstudio", settings.lmstudio_base_url)
    raise ValueError(f"Unknown provider: {name}")


def all_providers() -> list[LLMProvider]:
    return [
        LLMProvider("ollama", settings.ollama_base_url, native_ollama=True),
        LLMProvider("lmstudio", settings.lmstudio_base_url),
    ]
