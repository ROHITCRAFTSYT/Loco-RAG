"""ORM models (SQLModel tables) and shared enums."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    title: str = "New chat"
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    # RAG settings persisted per conversation
    rag_enabled: bool = False
    collection: Optional[str] = None
    web_search: bool = False
    pinned: bool = False
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Message(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    conversation_id: str = Field(index=True, foreign_key="conversation.id")
    role: str  # "system" | "user" | "assistant"
    content: str
    # JSON-encoded list of citation/source dicts attached to assistant messages
    sources: Optional[str] = None
    # JSON-encoded usage stats {prompt_tokens, completion_tokens, tps}
    usage: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


class Document(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    collection: str = Field(index=True)
    filename: str
    content_type: Optional[str] = None
    size_bytes: int = 0
    status: str = "pending"  # pending | processing | ready | error
    error: Optional[str] = None
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=_now)


class SavedPrompt(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    name: str
    content: str
    created_at: datetime = Field(default_factory=_now)
