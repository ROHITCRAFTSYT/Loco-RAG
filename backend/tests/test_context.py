"""Unit tests for the shared context builder and retrieval helpers.

build_context is what every answer path (document RAG, web search, memory) feeds
to the model, so its citation formatting is worth pinning down. All pure — no
vector store, embedder, or network.
"""
from __future__ import annotations

from app.schemas import Source
from app.services.rag import _rrf_fuse, _tokenize, build_context
from app.services.vectorstore.base import QueryHit


# --------------------------------------------------------------- tokenize
def test_tokenize_lowercases_and_splits_on_words():
    assert _tokenize("Hello, World! FOO_bar") == ["hello", "world", "foo_bar"]


def test_tokenize_empty():
    assert _tokenize("   ") == []


# --------------------------------------------------------------- build_context
def test_build_context_empty_sources_is_blank():
    assert build_context([]) == ""


def test_build_context_numbers_citations_from_one():
    sources = [
        Source(id="a", document="doc-a.pdf", text="alpha"),
        Source(id="b", document="doc-b.pdf", text="beta"),
    ]
    ctx = build_context(sources)
    assert "[1] doc-a.pdf" in ctx
    assert "[2] doc-b.pdf" in ctx
    assert "alpha" in ctx and "beta" in ctx
    assert "Cite them inline as [n]" in ctx


def test_build_context_uses_url_over_document_and_annotates_page():
    src = Source(id="w", document="Some Title", text="body", kind="web",
                 url="https://example.com/a", page=None)
    assert "https://example.com/a" in build_context([src])

    paged = Source(id="d", document="book.pdf", text="body", page=42)
    assert "(p.42)" in build_context([paged])


# --------------------------------------------------------------- rrf edges
def test_rrf_single_list_preserves_order():
    only = [QueryHit("a", "a", 0.9, {}), QueryHit("b", "b", 0.5, {})]
    fused = _rrf_fuse([only])
    assert [h.id for h in fused] == ["a", "b"]


def test_rrf_deduplicates_and_boosts_repeated_ids():
    a = [QueryHit("dup", "t", 0.9, {}), QueryHit("x", "x", 0.1, {})]
    b = [QueryHit("dup", "t", 0.9, {}), QueryHit("y", "y", 0.1, {})]
    fused = _rrf_fuse([a, b])
    ids = [h.id for h in fused]
    assert ids.count("dup") == 1          # merged, not duplicated
    assert ids[0] == "dup"                # appearing in both lists wins
    assert set(ids) == {"dup", "x", "y"}


def test_rrf_empty_input():
    assert _rrf_fuse([]) == []
