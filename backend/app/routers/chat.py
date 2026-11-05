"""Core chat endpoint: orchestrates memory + RAG + web search, then streams the
assistant reply over SSE. Persists user and assistant messages.
"""
from __future__ import annotations

import asyncio
import json
import logging
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
from app.services.llm import LLMError, get_provider

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

# Injected when a retrieval source fails so the model tells the user, instead of
# silently answering as if their documents had been consulted.
_DEGRADED_NOTE = (
    "Note: {warning} Tell the user briefly that you could not consult it, then "
    "answer as best you can without it."
)


def _history(session: Session, conversation_id: str) -> list[dict[str, Any]]:
    rows = session.exec(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    ).all()
    return [{"role": m.role, "content": m.content} for m in rows]


async def _gather_sources(
    req: ChatRequest, conv: Conversation, query: str
) -> tuple[list[Source], list[str]]:
    """Run document retrieval and/or web search. Returns ``(sources, warnings)``:
    a failing source is logged and reported as a warning rather than failing
    the whole chat turn."""
    sources: list[Source] = []
    warnings: list[str] = []
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
        try:
            # Embedding + BM25 are CPU-bound and synchronous: keep them off the
            # event loop so other requests' token streams don't stall.
            sources += await asyncio.to_thread(rag.retrieve, query, collections)
        except rag.RetrievalError as exc:
            detail = f"{exc.stage} failed: {exc.cause}" if exc.cause else f"{exc.stage} failed"
            warnings.append(f"Document search was unavailable for this message ({detail}).")
    if web_on:
        try:
            sources += await websearch.search(query)
        except Exception as exc:  # provider rate limits / network errors
            logger.warning("web search failed", extra={"error": f"{type(exc).__name__}: {exc}"})
            warnings.append(
                f"Web search was unavailable for this message ({type(exc).__name__})."
            )
    return sources, warnings


@router.post("/conversations/{conversation_id}/chat")
async def chat(
    conversation_id: str, req: ChatRequest, session: Session = Depends(get_session)
):
    conv = session.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")

    model = req.model or conv.model or settings.default_model
    provider = req.provider or conv.provider or settings.default_provider
    try:
        llm = get_provider(provider)
    except ValueError as exc:  # validate before persisting anything
        raise HTTPException(400, str(exc)) from exc

    # Persist the user message immediately.
    user_msg = Message(conversation_id=conversation_id, role="user", content=req.content)
    session.add(user_msg)
    session.commit()
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

    warnings: list[str] = []
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
        sources, warnings = await _gather_sources(req, conv, req.content)
        if sources:
            messages.append({"role": "system", "content": rag.build_context(sources)})
        for warning in warnings:
            messages.append(
                {"role": "system", "content": _DEGRADED_NOTE.format(warning=warning)}
            )
        messages += recent

    async def event_gen():
        # Emit sources first so the UI can render the citation panel.
        if sources:
            yield {"event": "sources", "data": json.dumps([s.model_dump() for s in sources])}
        for warning in warnings:
            yield {"event": "warning", "data": warning}

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
        except LLMError as exc:  # already logged with provider/model context
            yield {"event": "error", "data": str(exc)}
            return
        except Exception as exc:  # anything else: log the traceback, keep the message generic
            logger.exception("chat stream failed", extra={"model": model, "provider": provider})
            yield {"event": "error", "data": f"Generation failed ({type(exc).__name__})"}
            return

        elapsed = max(time.perf_counter() - start, 1e-6)
        tps = round((completion_tokens or len(acc.split())) / elapsed, 1)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tps": tps,
        }
        logger.info(
            "chat turn complete",
            extra={
                "conversation_id": conversation_id,
                "model": model,
                "provider": provider,
                "sources": len(sources),
                "degraded": bool(warnings),
                "completion_tokens": completion_tokens,
                "tps": tps,
            },
        )

        # Persist the assistant message with sources + usage.
        with get_session_ctx() as s2:
            assistant = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=acc,
                sources=json.dumps([s.model_dump() for s in sources]) if sources else None,
                # Warnings ride in the usage JSON (no schema migration); the
                # messages API lifts them out into a top-level "warnings" field.
                usage=json.dumps({**usage, "warnings": warnings} if warnings else usage),
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
