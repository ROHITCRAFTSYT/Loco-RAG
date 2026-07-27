"""Unit tests for conversation-memory token accounting and the summarize gate.

maybe_summarize decides when older turns get compacted; getting the gate wrong
either blows the context window or summarizes needlessly. The token counter and
the short-circuit paths are pure and testable without an LLM provider.
"""
from __future__ import annotations

import asyncio

from app.services import memory


def test_count_sums_token_lengths():
    msgs = [{"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi"}]
    assert memory._count(msgs) > 0
    assert memory._count([]) == 0


def test_count_ignores_missing_content():
    assert memory._count([{"role": "system"}]) == 0


def test_short_history_passes_through_unchanged():
    history = [{"role": "user", "content": "hi"}]
    summary, recent = asyncio.run(
        memory.maybe_summarize(history, model="m", provider="p"))
    assert summary is None
    assert recent == history


def test_history_within_token_budget_is_untouched():
    # More than KEEP_RECENT messages, but tiny — under the token budget.
    history = [{"role": "user", "content": "a"} for _ in range(memory.KEEP_RECENT + 4)]
    summary, recent = asyncio.run(
        memory.maybe_summarize(history, model="m", provider="p"))
    assert summary is None
    assert recent == history
