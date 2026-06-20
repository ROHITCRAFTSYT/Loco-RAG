"""Agentic tool-calling loop.

When agent mode is on, the model is given two tools — document search and web
search — and decides for itself when (and how many times) to call them. We run the
tool rounds non-streaming, accumulate the resulting Sources, then hand the augmented
message list back to the caller to stream the final answer.
"""
from __future__ import annotations

import json
from typing import Any

from app.schemas import Source
from app.services import rag, websearch
from app.services.llm import get_provider

MAX_ROUNDS = 4

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the user's uploaded documents / knowledge base for "
            "relevant passages. Use this for questions about uploaded files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "collection": {
                        "type": "string",
                        "description": "Optional collection name to scope the search",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for current or external information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
]


async def _run_tool(name: str, args: dict[str, Any], default_collection: str | None) -> list[Source]:
    if name == "search_documents":
        collection = args.get("collection") or default_collection
        return rag.retrieve(args.get("query", ""), collection)
    if name == "web_search":
        return await websearch.search(args.get("query", ""))
    return []


async def run(
    base_messages: list[dict[str, Any]],
    *,
    model: str,
    provider: str,
    default_collection: str | None,
) -> tuple[list[dict[str, Any]], list[Source]]:
    """Run tool-calling rounds. Returns (messages_for_final_answer, all_sources)."""
    llm = get_provider(provider)
    messages = list(base_messages)
    all_sources: list[Source] = []

    for _ in range(MAX_ROUNDS):
        try:
            assistant = await llm.chat_with_tools(messages, TOOLS, model=model)
        except Exception:
            # Model/endpoint doesn't support tools — fall back to plain answer.
            break

        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            break

        # Record the assistant's tool-call turn, then answer each call.
        messages.append({"role": "assistant", "content": assistant.get("content") or "", "tool_calls": tool_calls})
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            sources = await _run_tool(name, args, default_collection)
            all_sources += sources
            tool_result = rag.build_context(sources) if sources else "No results found."
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", name),
                    "name": name,
                    "content": tool_result,
                }
            )

    # De-duplicate sources by id, preserving order.
    seen: set[str] = set()
    unique: list[Source] = []
    for s in all_sources:
        if s.id not in seen:
            seen.add(s.id)
            unique.append(s)
    return messages, unique
