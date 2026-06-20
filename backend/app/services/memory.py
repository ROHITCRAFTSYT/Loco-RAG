"""Conversation memory: summarize older turns to fit the context window.

Strategy: keep the last N turns verbatim; if the running history exceeds a token
budget, summarize everything older into a compact "memory" system note.
"""
from __future__ import annotations

import tiktoken

from app.config import settings
from app.services.llm import get_provider

_enc = tiktoken.get_encoding("cl100k_base")

KEEP_RECENT = 6  # always keep this many most-recent messages verbatim


def _count(messages: list[dict]) -> int:
    return sum(len(_enc.encode(m.get("content", ""))) for m in messages)


async def maybe_summarize(
    history: list[dict], *, model: str, provider: str
) -> tuple[str | None, list[dict]]:
    """Return (summary_or_None, recent_messages_to_send).

    If the history is within budget, summary is None and all messages pass through.
    """
    if _count(history) <= settings.memory_summarize_after_tokens or len(history) <= KEEP_RECENT:
        return None, history

    older = history[:-KEEP_RECENT]
    recent = history[-KEEP_RECENT:]

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in older)
    prompt = [
        {
            "role": "system",
            "content": "Summarize the conversation so far into concise bullet points "
            "capturing facts, decisions, and user preferences. Be terse.",
        },
        {"role": "user", "content": transcript},
    ]
    try:
        summary = await get_provider(provider).complete(prompt, model=model, temperature=0.2)
    except Exception:
        # On any failure, fall back to recent-only to avoid blowing the context window.
        return None, recent
    return summary, recent
