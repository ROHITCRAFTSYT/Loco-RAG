"""LLM provider abstraction.

Both Ollama and LM Studio expose an OpenAI-compatible ``/v1/chat/completions``
endpoint, so a single async client covers chat for both. Ollama's native API is
used only for richer model discovery (capabilities, families).
"""
from __future__ import annotations

from app.services.llm.base import LLMProvider, get_provider

__all__ = ["LLMProvider", "get_provider"]
