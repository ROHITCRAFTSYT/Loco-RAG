"""Web search tool. Fetches results, extracts page text, returns Sources that
flow through the same context builder as document RAG.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from app.config import settings
from app.schemas import Source


async def _is_safe_public_url(url: str) -> bool:
    """SSRF guard: allow only http(s) URLs whose host resolves entirely to
    global (public) addresses.

    Result URLs come from an external search provider and are fetched
    server-side, so without this a poisoned result (or a compromised SearXNG
    instance) could steer the fetch at cloud-metadata (169.254.169.254),
    loopback, or RFC-1918 internal services. Every resolved address must be
    global — a host that resolves to even one private IP is rejected.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = await asyncio.to_thread(
            socket.getaddrinfo, parsed.hostname, port, 0, socket.SOCK_STREAM
        )
        if not infos:
            return False
        return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)
    except Exception:
        return False


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
    if not await _is_safe_public_url(url):
        return ""
    try:
        # follow_redirects is disabled: an allowed public URL could otherwise
        # 3xx-redirect to an internal address, bypassing the pre-fetch guard.
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=False) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            html = r.text
        import trafilatura

        text = trafilatura.extract(html) or ""
        return text[:4000]
    except Exception:
        return ""


async def search(query: str) -> list[Source]:
    """Run a web search and return extracted-text Sources.

    Uses SearXNG or DuckDuckGo per ``settings.websearch_provider``, fetches
    and extracts each result page (falling back to the result snippet when
    extraction is empty), and skips results with no usable text.
    """
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
                # Key by URL, not position: the agent may call web_search several
                # times per turn, and positional ids (web:0, web:1, …) collide
                # across calls so its id-based dedup would drop valid results.
                id=f"web:{url}" if url else f"web:{i}",
                document=r.get("title") or url,
                score=max(0.0, 1.0 - i * 0.05),
                text=text,
                kind="web",
                url=url,
            )
        )
    return sources
