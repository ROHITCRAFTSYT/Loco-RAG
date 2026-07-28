"""Standalone search endpoints: web search, and a document-retrieval preview
(also used internally by the chat orchestrator)."""
from __future__ import annotations

from fastapi import APIRouter

from app.services import rag, websearch

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("")
async def web_search(q: str):
    sources = await websearch.search(q)
    return [s.model_dump() for s in sources]


@router.get("/documents")
def document_search(q: str, collection: str | None = None, hybrid: bool = True):
    """Preview the chunks RAG would retrieve for ``q`` — the same ranked, cited
    Sources (after hybrid fusion, optional rerank and MMR) that get injected as
    context. Handy for tuning retrieval without running a full chat turn."""
    sources = rag.retrieve(q, collection, hybrid=hybrid)
    return [s.model_dump() for s in sources]
