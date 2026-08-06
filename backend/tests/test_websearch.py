"""Web-search result assembly.

search() fans out extraction over provider results and turns them into Sources
that flow through the same context builder as document RAG. The provider call and
page fetch are mocked so no network is touched; these pin the Source shape,
the descending rank score, the fallback to the result snippet when extraction is
empty, and the skip of results with no usable text.
"""
from __future__ import annotations

import pytest

from app.services import websearch


@pytest.fixture(autouse=True)
def _force_ddg(monkeypatch):
    # Keep the provider deterministic regardless of settings.
    monkeypatch.setattr(websearch.settings, "websearch_provider", "ddg", raising=False)
    monkeypatch.setattr(websearch.settings, "websearch_max_results", 5, raising=False)


def _mock_results(monkeypatch, results):
    async def fake(query, max_results):
        return results

    monkeypatch.setattr(websearch, "_ddg_results", fake)


def _mock_extract(monkeypatch, mapping):
    async def fake(url):
        return mapping.get(url, "")

    monkeypatch.setattr(websearch, "_extract", fake)


async def test_builds_sources_with_descending_scores(monkeypatch):
    _mock_results(monkeypatch, [
        {"title": "A", "href": "https://a.com", "body": "snippet a"},
        {"title": "B", "href": "https://b.com", "body": "snippet b"},
    ])
    _mock_extract(monkeypatch, {"https://a.com": "full a", "https://b.com": "full b"})
    sources = await websearch.search("q")
    assert [s.id for s in sources] == ["web:https://a.com", "web:https://b.com"]
    assert all(s.kind == "web" for s in sources)
    assert sources[0].score > sources[1].score          # rank-decayed
    assert sources[0].url == "https://a.com"
    assert sources[0].text == "full a"                  # extracted text preferred


async def test_falls_back_to_snippet_when_extraction_empty(monkeypatch):
    _mock_results(monkeypatch, [{"title": "A", "href": "https://a.com", "body": "just the snippet"}])
    _mock_extract(monkeypatch, {})  # extraction returns "" for everything
    sources = await websearch.search("q")
    assert len(sources) == 1
    assert sources[0].text == "just the snippet"


async def test_skips_results_with_no_usable_text(monkeypatch):
    _mock_results(monkeypatch, [
        {"title": "A", "href": "https://a.com", "body": ""},      # nothing to show
        {"title": "B", "href": "https://b.com", "body": "keep me"},
    ])
    _mock_extract(monkeypatch, {})
    sources = await websearch.search("q")
    assert [s.document for s in sources] == ["B"]


async def test_empty_provider_results_yield_no_sources(monkeypatch):
    _mock_results(monkeypatch, [])
    _mock_extract(monkeypatch, {})
    assert await websearch.search("q") == []


async def test_ids_are_url_stable_across_calls(monkeypatch):
    # The agent can call web_search multiple times per turn and dedups Sources by
    # id. Ids must be derived from the URL so results from a second call don't
    # collide with the first (positional ids would both start at web:0).
    _mock_extract(monkeypatch, {})
    _mock_results(monkeypatch, [{"title": "A", "href": "https://a.com", "body": "a"}])
    first = await websearch.search("q1")
    _mock_results(monkeypatch, [{"title": "B", "href": "https://b.com", "body": "b"}])
    second = await websearch.search("q2")
    assert first[0].id != second[0].id
    assert first[0].id == "web:https://a.com"
