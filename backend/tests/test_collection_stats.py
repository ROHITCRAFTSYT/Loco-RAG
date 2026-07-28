"""Per-collection knowledge-base stats aggregation.

aggregate_collection_stats rolls Document rows into per-collection totals for the
KB view. Pure over duck-typed rows, so it's tested with lightweight stand-ins —
no database. Ephemeral chat-* attachment collections must be excluded.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.ingest import aggregate_collection_stats


@dataclass
class Doc:
    collection: str
    chunk_count: int = 0
    size_bytes: int = 0
    status: str = "ready"


def test_empty_input():
    assert aggregate_collection_stats([]) == {}


def test_rolls_up_per_collection():
    docs = [
        Doc("kb", chunk_count=10, size_bytes=100, status="ready"),
        Doc("kb", chunk_count=5, size_bytes=50, status="ready"),
        Doc("kb", chunk_count=0, size_bytes=20, status="error"),
        Doc("notes", chunk_count=3, size_bytes=30, status="ready"),
    ]
    stats = aggregate_collection_stats(docs)
    assert stats["kb"] == {"documents": 3, "ready": 2, "chunks": 15, "size_bytes": 170}
    assert stats["notes"] == {"documents": 1, "ready": 1, "chunks": 3, "size_bytes": 30}


def test_excludes_ephemeral_chat_collections():
    docs = [Doc("kb", chunk_count=1), Doc("chat-abc123", chunk_count=99)]
    stats = aggregate_collection_stats(docs)
    assert "kb" in stats
    assert "chat-abc123" not in stats


def test_handles_none_counts_defensively():
    d = Doc("kb")
    d.chunk_count = None  # type: ignore[assignment]
    d.size_bytes = None  # type: ignore[assignment]
    stats = aggregate_collection_stats([d])
    assert stats["kb"]["chunks"] == 0
    assert stats["kb"]["size_bytes"] == 0
