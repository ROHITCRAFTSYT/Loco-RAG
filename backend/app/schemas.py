"""Pydantic request/response schemas (API layer, separate from ORM tables)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    rag_enabled: Optional[bool] = None
    collection: Optional[str] = None
    web_search: Optional[bool] = None
    pinned: Optional[bool] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    rag_enabled: Optional[bool] = None
    collection: Optional[str] = None
    web_search: Optional[bool] = None
    pinned: Optional[bool] = None


class ChatRequest(BaseModel):
    """Send a new user message and stream the assistant reply."""

    content: str
    # Per-request overrides (fall back to conversation settings)
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None
    rag_enabled: Optional[bool] = None
    collection: Optional[str] = None
    web_search: Optional[bool] = None
    # Ephemeral per-message attachments ("talk to this doc") — retrieved even when
    # the conversation's persistent RAG is off.
    attachment_collections: Optional[list[str]] = None
    # Agentic tool-calling: let the model decide when to search docs/web.
    agent_mode: Optional[bool] = None


class ModelInfo(BaseModel):
    id: str
    provider: str
    supports_vision: bool = False


class Source(BaseModel):
    id: str
    document: str
    collection: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    score: float = 0.0
    text: str
    kind: str = "document"  # "document" | "web" | "memory"
    url: Optional[str] = None


class IngestResponse(BaseModel):
    document_id: str
    status: str
    chunk_count: int


class SavedPromptIn(BaseModel):
    name: str
    content: str


def jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable structures."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, list):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    return obj
