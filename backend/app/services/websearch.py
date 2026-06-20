"""Web search tool. Fetches results, extracts page text, returns Sources that
flow through the same context builder as document RAG.
"""
from __future__ import annotations

import asyncio

import httpx

from app.config import settings
from app.schemas import Source


async def _ddg_results(query: str, max_results: int) -> list[dict]:
    from duckduckgo_search import DDGS

    def _run() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    return await asyncio.to_thread(_run)


async def _searxng_results(query: str, max_results: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{settings.searxng_base_url}/search",
            params={"q": query, "format": "json"},
        )
        r.raise_for_status()
        results = r.json().get("results", [])[:max_results]
        return [{"title": x.get("title"), "href": x.get("url"), "body": x.get("content")} for x in results]


async def _extract(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            html = r.text
        import trafilatura

        text = trafilatura.extract(html) or ""
        return text[:4000]
    except Exception:
        return ""


async def search(query: str) -> list[Source]:
    max_results = settings.websearch_max_results
    if settings.websearch_provider == "searxng":
        results = await _searxng_results(query, max_results)
    else:
        results = await _ddg_results(query, max_results)

    sources: list[Source] = []
    extractions = await asyncio.gather(
        *[_extract(r.get("href") or r.get("url") or "") for r in results]
    )
    for i, (r, body) in enumerate(zip(results, extractions)):
        url = r.get("href") or r.get("url") or ""
        text = body or r.get("body") or ""
        if not text:
            continue
        sources.append(
            Source(
                id=f"web:{i}",
                document=r.get("title") or url,
                score=1.0 - i * 0.05,
                text=text,
                kind="web",
                url=url,
            )
        )
    return sources
