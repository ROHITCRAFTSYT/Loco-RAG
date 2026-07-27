"""The agentic tool-calling loop.

agent.run drives up to MAX_ROUNDS of model-decided tool calls, accumulates
Sources, and hands back an augmented message list. These tests use a scripted
fake provider and stubbed retrieval so the loop's control flow is exercised with
no LLM or network: early exit, dedup, robustness to bad tool args and unknown
tools, the tools-unsupported fallback, and the round cap.
"""
from __future__ import annotations

import pytest

from app.schemas import Source
from app.services import agent


class FakeProvider:
    """Returns a scripted list of assistant turns, one per chat_with_tools call."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = 0

    async def chat_with_tools(self, messages, tools, *, model):
        self.calls += 1
        if not self._turns:
            return {"content": "done", "tool_calls": []}
        return self._turns.pop(0)


def _tool_call(name, arguments='{"query": "q"}', call_id="c1"):
    return {"tool_calls": [{"id": call_id, "function": {"name": name, "arguments": arguments}}]}


@pytest.fixture(autouse=True)
def _stub_tools(monkeypatch):
    """Stub the actual retrieval so no vector store / network is touched."""
    recorded = {"doc_args": [], "web_args": []}

    def fake_retrieve(query, collection=None, **kw):
        recorded["doc_args"].append((query, collection))
        return [Source(id="doc-1", document="d.pdf", text="t")]

    async def fake_web(query):
        recorded["web_args"].append(query)
        return [Source(id="web-1", document="W", text="t", kind="web")]

    monkeypatch.setattr(agent.rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(agent.websearch, "search", fake_web)
    return recorded


def _use(monkeypatch, provider):
    monkeypatch.setattr(agent, "get_provider", lambda name: provider)


async def test_no_tool_calls_exits_after_one_round(monkeypatch):
    fp = FakeProvider([{"content": "hi", "tool_calls": []}])
    _use(monkeypatch, fp)
    messages, sources = await agent.run(
        [{"role": "user", "content": "hello"}], model="m", provider="p",
        default_collection="kb")
    assert fp.calls == 1
    assert sources == []
    assert messages[-1]["content"] == "hello"  # base messages untouched


async def test_document_tool_call_collects_sources(monkeypatch, _stub_tools):
    fp = FakeProvider([_tool_call("search_documents"), {"tool_calls": []}])
    _use(monkeypatch, fp)
    messages, sources = await agent.run(
        [{"role": "user", "content": "q"}], model="m", provider="p",
        default_collection="kb")
    assert [s.id for s in sources] == ["doc-1"]
    assert _stub_tools["doc_args"] == [("q", "kb")]        # default collection applied
    assert any(m["role"] == "tool" for m in messages)


async def test_sources_are_deduplicated_by_id(monkeypatch):
    fp = FakeProvider([
        _tool_call("search_documents", call_id="a"),
        _tool_call("search_documents", call_id="b"),
        {"tool_calls": []},
    ])
    _use(monkeypatch, fp)
    _, sources = await agent.run(
        [{"role": "user", "content": "q"}], model="m", provider="p",
        default_collection="kb")
    assert [s.id for s in sources] == ["doc-1"]  # same id across two rounds -> one


async def test_bad_json_arguments_default_to_empty(monkeypatch, _stub_tools):
    fp = FakeProvider([_tool_call("search_documents", arguments="{not json"),
                       {"tool_calls": []}])
    _use(monkeypatch, fp)
    await agent.run([{"role": "user", "content": "q"}], model="m", provider="p",
                    default_collection="kb")
    # empty query survived (args -> {}), default collection still applied
    assert _stub_tools["doc_args"] == [("", "kb")]


async def test_unknown_tool_yields_no_results(monkeypatch):
    fp = FakeProvider([_tool_call("nonexistent_tool"), {"tool_calls": []}])
    _use(monkeypatch, fp)
    messages, sources = await agent.run(
        [{"role": "user", "content": "q"}], model="m", provider="p",
        default_collection="kb")
    assert sources == []
    assert any(m.get("content") == "No results found." for m in messages)


async def test_provider_without_tool_support_falls_back(monkeypatch):
    class Boom:
        async def chat_with_tools(self, *a, **k):
            raise RuntimeError("tools unsupported")

    _use(monkeypatch, Boom())
    base = [{"role": "user", "content": "q"}]
    messages, sources = await agent.run(base, model="m", provider="p",
                                        default_collection="kb")
    assert sources == []
    assert messages == base  # unchanged, ready for a plain answer


async def test_loop_stops_at_max_rounds(monkeypatch):
    # Provider always asks for another tool call; the loop must still terminate.
    fp = FakeProvider([_tool_call("web_search") for _ in range(agent.MAX_ROUNDS + 3)])
    _use(monkeypatch, fp)
    await agent.run([{"role": "user", "content": "q"}], model="m", provider="p",
                    default_collection="kb")
    assert fp.calls == agent.MAX_ROUNDS
