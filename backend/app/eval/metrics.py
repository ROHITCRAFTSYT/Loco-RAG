"""Information-retrieval metrics for scoring the RAG retriever.

Every function takes ``ranked`` — candidate ids best-first — and ``relevant`` —
the ids that should have been retrieved — and returns a score in ``[0, 1]``.
They are pure (no embedder, store, or config), so they unit-test in isolation
and can score any retriever whose output is reduced to a ranked id list.

Ids are opaque strings: use document ids for document-level evaluation or chunk
ids for chunk-level. Duplicate ids in ``ranked`` are treated positionally (the
first occurrence is what earns the rank).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """``1 / rank`` of the first relevant hit (rank is 1-based), or 0.0 if none."""
    rel = set(relevant)
    for i, cid in enumerate(ranked, start=1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-``k``, else 0.0."""
    rel = set(relevant)
    return 1.0 if any(cid in rel for cid in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of all relevant ids that appear in the top-``k``.

    0.0 when nothing is relevant (an undefined ratio scored as a miss).
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & rel) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-``k`` that are relevant.

    Divided by the number of items actually returned (``min(k, len(ranked))``)
    so a retriever that returns fewer than ``k`` results is not penalised for the
    empty slots.
    """
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for cid in top if cid in rel) / len(top)


def average_precision(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """Average precision: mean of precision@i over the ranks i that hold a
    relevant id, normalised by the total number of relevant ids."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = 0
    score = 0.0
    seen: set[str] = set()
    for i, cid in enumerate(ranked, start=1):
        if cid in rel and cid not in seen:
            seen.add(cid)
            hits += 1
            score += hits / i
    return score / len(rel)


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance normalised DCG at ``k``.

    DCG sums ``1 / log2(rank + 1)`` over relevant hits in the top-``k``; the ideal
    DCG places all (up to ``k``) relevant items first. Returns 0.0 when nothing is
    relevant.
    """
    rel = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, cid in enumerate(ranked[:k], start=1)
        if cid in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
