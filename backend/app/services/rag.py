"""RAG pipeline: hybrid retrieval (dense + BM25 -> RRF) -> optional rerank ->
unified context builder shared by document RAG, web search, and memory.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.config import settings
from app.schemas import Source
from app.services.embeddings import get_embedder
from app.services.vectorstore import get_vectorstore
from app.services.vectorstore.base import QueryHit

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_search(collection: str, query: str, top_k: int) -> list[QueryHit]:
    from rank_bm25 import BM25Okapi

    corpus = get_vectorstore().all_texts(collection)
    if not corpus:
        return []
    bm25 = BM25Okapi([_tokenize(h.text) for h in corpus])
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [QueryHit(id=h.id, text=h.text, score=float(s), metadata=h.metadata) for h, s in ranked]


def _rrf_fuse(rank_lists: list[list[QueryHit]], k: int = 60) -> list[QueryHit]:
    """Reciprocal Rank Fusion of multiple ranked lists keyed by hit id."""
    scores: dict[str, float] = {}
    by_id: dict[str, QueryHit] = {}
    for hits in rank_lists:
        for rank, hit in enumerate(hits):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
            by_id[hit.id] = hit
    fused = sorted(by_id.values(), key=lambda h: scores[h.id], reverse=True)
    for h in fused:
        h.score = scores[h.id]
    return fused


@lru_cache
def _get_reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=settings.rerank_model)


def _rerank(query: str, hits: list[QueryHit], top_n: int) -> list[QueryHit]:
    if not hits:
        return hits
    scores = list(_get_reranker().rerank(query, [h.text for h in hits]))
    for h, s in zip(hits, scores):
        h.score = float(s)
    return sorted(hits, key=lambda h: h.score, reverse=True)[:top_n]


def retrieve(
    query: str, collection: str | list[str] | None = None, *, hybrid: bool = True
) -> list[Source]:
    """Retrieve relevant document chunks for a query.

    ``collection`` may be a single name or a list — passing several merges results
    across collections (used for per-message ephemeral attachments alongside a
    persistent knowledge base).
    """
    if collection is None:
        collections = [settings.default_collection]
    elif isinstance(collection, str):
        collections = [collection]
    else:
        collections = [c for c in collection if c]
    if not collections:
        collections = [settings.default_collection]

    top_k = settings.retrieve_top_k
    dense_q = get_embedder().embed([query])[0]
    store = get_vectorstore()

    rank_lists: list[list[QueryHit]] = []
    for coll in collections:
        rank_lists.append(store.query(coll, dense_q, top_k))
        if hybrid:
            rank_lists.append(_bm25_search(coll, query, top_k))

    hits = _rrf_fuse(rank_lists) if len(rank_lists) > 1 else (rank_lists[0] if rank_lists else [])

    if settings.enable_rerank and hits:
        hits = _rerank(query, hits, settings.rerank_top_n)
    else:
        hits = hits[: settings.rerank_top_n]

    return [
        Source(
            id=h.id,
            document=h.metadata.get("document", "unknown"),
            collection=h.metadata.get("collection", collections[0]),
            page=h.metadata.get("page"),
            chunk_index=h.metadata.get("chunk_index"),
            score=round(h.score, 4),
            text=h.text,
            kind="document",
        )
        for h in hits
    ]


def build_context(sources: list[Source]) -> str:
    """Format retrieved sources into a single context block with [n] citations.

    Shared by document RAG, web search, and memory so injected context is
    uniform and the model can cite consistently.
    """
    if not sources:
        return ""
    blocks = []
    for i, s in enumerate(sources, start=1):
        loc = s.url or s.document
        if s.page:
            loc += f" (p.{s.page})"
        blocks.append(f"[{i}] {loc}\n{s.text}")
    joined = "\n\n".join(blocks)
    return (
        "Use the following sources to answer the question. Cite them inline as [n] "
        "where relevant. If the answer is not in the sources, say so.\n\n"
        f"{joined}"
    )
