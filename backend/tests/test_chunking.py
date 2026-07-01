"""Unit tests for token-aware chunking and RRF fusion (no external services)."""
from __future__ import annotations

from app.services.ingest import chunk_tokens
from app.services.rag import _rrf_fuse
from app.services.vectorstore.base import QueryHit


def test_chunking_respects_overlap_and_indexes():
    long_text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_tokens([(1, long_text)])
    assert len(chunks) > 1
    # chunk indices are contiguous starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # every chunk carries the page tag
    assert all(c.page == 1 for c in chunks)


def test_chunking_empty_text():
    assert chunk_tokens([(None, "   ")]) == []


def test_chunking_short_text_single_chunk():
    # well under chunk_size_tokens -> exactly one chunk, index 0
    chunks = chunk_tokens([(1, "just a short sentence")])
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].page == 1


def test_chunking_preserves_page_boundaries_across_multiple_pages():
    pages = [(1, " ".join(f"a{i}" for i in range(50))), (2, " ".join(f"b{i}" for i in range(50)))]
    chunks = chunk_tokens(pages)
    # page 1 chunks all come before page 2 chunks, and pages are never mixed
    pages_seen = [c.page for c in chunks]
    assert pages_seen == sorted(pages_seen)
    assert set(pages_seen) == {1, 2}
    # chunk_index is a single contiguous sequence across pages, not reset per page
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_rrf_fuse_prefers_items_high_in_both_lists():
    a = [QueryHit("x", "x", 0.9, {}), QueryHit("y", "y", 0.8, {})]
    b = [QueryHit("y", "y", 0.7, {}), QueryHit("z", "z", 0.6, {})]
    fused = _rrf_fuse([a, b])
    ids = [h.id for h in fused]
    # y appears in both lists near the top -> should rank first
    assert ids[0] == "y"
    assert set(ids) == {"x", "y", "z"}
