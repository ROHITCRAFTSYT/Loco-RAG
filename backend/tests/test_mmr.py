"""Maximal Marginal Relevance ordering.

mmr_order is the diversity re-ranker that stops the retrieved context from being
several near-duplicate chunks. Pure vector math (no embedder/store), so its
relevance/diversity trade-off is exactly checkable.
"""
from __future__ import annotations

from app.services.rag import _cosine, mmr_order


def test_cosine_basics():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert _cosine([0, 0], [1, 1]) == 0.0  # zero vector -> 0, no div-by-zero


def test_lambda_one_is_pure_relevance_order():
    query = [1.0, 0.0]
    cands = [[0.2, 1.0], [0.9, 0.1], [0.6, 0.4]]  # relevance to query: c1<c2, c0 lowest
    order = mmr_order(query, cands, k=3, lambda_=1.0)
    # most query-similar first
    assert order[0] == 1


def test_lambda_zero_maximizes_diversity():
    query = [1.0, 0.0]
    # two near-duplicates (0,1) and one orthogonal-ish; pure diversity must avoid
    # picking both duplicates back to back.
    cands = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    order = mmr_order(query, cands, k=2, lambda_=0.0)
    # after the first pick, the second should be the dissimilar one (index 2)
    assert order[1] == 2


def test_returns_at_most_k_and_valid_indices():
    cands = [[1, 0], [0, 1], [1, 1]]
    order = mmr_order([1, 0], cands, k=2)
    assert len(order) == 2
    assert set(order) <= {0, 1, 2}
    assert len(set(order)) == 2  # no repeats


def test_k_larger_than_candidates_and_empty():
    assert mmr_order([1, 0], [[1, 0]], k=5) == [0]
    assert mmr_order([1, 0], [], k=3) == []


def test_diversity_beats_a_duplicate_when_favoured():
    query = [1.0, 0.0]
    # c0 most relevant; c1 is a near-duplicate of c0; c2 relevant-but-different.
    cands = [[0.9, 0.1], [0.88, 0.12], [0.6, 0.5]]
    order = mmr_order(query, cands, k=2, lambda_=0.2)  # favour diversity
    assert order[0] == 0
    assert order[1] == 2  # skips the near-duplicate in favour of the diverse chunk
