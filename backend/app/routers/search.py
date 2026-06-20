"""Standalone web-search endpoint (also used internally by the chat orchestrator)."""
from __future__ import annotations

from fastapi import APIRouter

from app.services import websearch

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("")
async def web_search(q: str):
    sources = await websearch.search(q)
    return [s.model_dump() for s in sources]
