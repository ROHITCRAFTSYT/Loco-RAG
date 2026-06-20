"""Core chat endpoint: orchestrates memory + RAG + web search, then streams the
assistant reply over SSE. Persists user and assistant messages.
"""
from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.db import get_session
from app.models import Conversation, Message
from app.schemas import ChatRequest, Source
from app.services import agent, memory, rag, websearch
from app.services.llm import get_provider

router = APIRouter(prefix="/api", tags=["chat"])


def _history(session: Session, conversation_id: str) -> list[dict[str, Any]]:
    rows = session.exec(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    ).all()
    return [{"role": m.role, "content": m.content} for m in rows]


async def _gather_sources(req: ChatRequest, conv: Conversation, query: str) -> list[Source]:
    sources: list[Source] = []
    rag_on = req.rag_enabled if req.rag_enabled is not None else conv.rag_enabled
    web_on = req.web_search if req.web_search is not None else conv.web_search

    # Collections to search: persistent KB (when RAG on) + any ephemeral attachments
    # (always searched — the user explicitly attached them for this message).
    collections: list[str] = []
    if rag_on:
        collections.append(req.collection or conv.collection or settings.default_collection)
    if req.attachment_collections:
        collections += req.attachment_collections

    if collections:
        sources += rag.retrieve(query, collections)
    if web_on:
        sources += await websearch.search(query)
    return sources


@router.post("/conversations/{conversation_id}/chat")
async def chat(
    conversation_id: str, req: ChatRequest, session: Session = Depends(get_session)
):
    conv = session.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")

    # Persist the user message immediately.
    user_msg = Message(conversation_id=conversation_id, role="user", content=req.content)
    session.add(user_msg)
    session.commit()

    model = req.model or conv.model or settings.default_model
    provider = req.provider or conv.provider or settings.default_provider
    temperature = req.temperature if req.temperature is not None else conv.temperature

    history = _history(session, conversation_id)

    # Memory: summarize older turns if needed.
    summary, recent = await memory.maybe_summarize(history, model=model, provider=provider)

    # Base messages (system prompt + memory summary + recent turns).
    messages: list[dict[str, Any]] = []
    if conv.system_prompt:
        messages.append({"role": "system", "content": conv.system_prompt})
    if summary:
        messages.append({"role": "system", "content": f"Conversation summary so far:\n{summary}"})

    if req.agent_mode:
        # Let the model decide which tools to call; it accumulates its own sources.
        default_collection = req.collection or conv.collection or settings.default_collection
        agent_messages, sources = await agent.run(
            messages + recent,
            model=model,
            provider=provider,
            default_collection=default_collection,
        )
        messages = agent_messages
    else:
        # Deterministic retrieval (RAG + web) unified through the context builder.
        sources = await _gather_sources(req, conv, req.content)
        if sources:
            messages.append({"role": "system", "content": rag.build_context(sources)})
        messages += recent

    llm = get_provider(provider)

    async def event_gen():
        # Emit sources first so the UI can render the citation panel.
        if sources:
            yield {"event": "sources", "data": json.dumps([s.model_dump() for s in sources])}

        acc = ""
        start = time.perf_counter()
        prompt_tokens = completion_tokens = None
        try:
            async for chunk in llm.stream_chat(
                messages, model=model, temperature=temperature, max_tokens=conv.max_tokens
            ):
                if chunk.delta:
                    acc += chunk.delta
                    yield {"event": "token", "data": chunk.delta}
                if chunk.prompt_tokens is not None:
                    prompt_tokens = chunk.prompt_tokens
                if chunk.completion_tokens is not None:
                    completion_tokens = chunk.completion_tokens
                if chunk.done:
                    break
        except Exception as exc:  # surface provider errors to the client
            yield {"event": "error", "data": str(exc)}
            return

        elapsed = max(time.perf_counter() - start, 1e-6)
        tps = round((completion_tokens or len(acc.split())) / elapsed, 1)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tps": tps,
        }

        # Persist the assistant message with sources + usage.
        with get_session_ctx() as s2:
            assistant = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=acc,
                sources=json.dumps([s.model_dump() for s in sources]) if sources else None,
                usage=json.dumps(usage),
            )
            s2.add(assistant)
            conv2 = s2.get(Conversation, conversation_id)
            if conv2 and conv2.title == "New chat":
                conv2.title = req.content[:60]
            s2.commit()

        yield {"event": "usage", "data": json.dumps(usage)}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())


def get_session_ctx():
    """Standalone session context for use inside the async generator."""
    from contextlib import contextmanager

    from app.db import engine

    @contextmanager
    def _ctx():
        with Session(engine) as s:
            yield s

    return _ctx()
