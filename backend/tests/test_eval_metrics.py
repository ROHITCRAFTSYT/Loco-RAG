"""IR metrics for retrieval evaluation.

These pin the exact values of hit-rate, MRR, recall, precision, MAP and nDCG on
small hand-checkable rankings, plus the degenerate cases (no relevant ids, fewer
results than k, relevant item ranked last) that a retriever hits in practice.
"""
from __future__ import annotations

import math

from app.eval.metrics import (
    average_precision,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["b", "a"], {"b"}) == 1.0
    # First relevant wins even if a later one also matches.
    assert reciprocal_rank(["x", "b", "c"], {"b", "c"}) == 0.5


def test_reciprocal_rank_no_hit_is_zero():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_hit_rate_respects_k():
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0  # c is out of the top-2
    assert hit_rate_at_k(["a", "b"], {"z"}, k=5) == 0.0


def test_recall_counts_distinct_relevant_in_top_k():
    # 2 of the 3 relevant ids are in the top-3.
    assert recall_at_k(["a", "b", "x"], {"a", "b", "c"}, k=3) == 2 / 3
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["a"], set(), k=3) == 0.0


def test_precision_divides_by_returned_not_k():
    # Only 2 results returned but k=5: denominator is 2, not 5.
    assert precision_at_k(["a", "b"], {"a"}, k=5) == 0.5
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 0.5


def test_precision_edge_cases():
    assert precision_at_k([], {"a"}, k=3) == 0.0
    assert precision_at_k(["a"], {"a"}, k=0) == 0.0


def test_average_precision_matches_hand_calc():
    # Relevant at ranks 1 and 3 out of 2 relevant total:
    # (1/1 + 2/3) / 2 = (1 + 0.6667) / 2 = 0.8333...
    ap = average_precision(["a", "x", "b"], {"a", "b"})
    assert math.isclose(ap, (1.0 + 2 / 3) / 2)


def test_average_precision_no_relevant():
    assert average_precision(["a", "b"], set()) == 0.0
    assert average_precision(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert math.isclose(ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3), 1.0)


def test_ndcg_penalises_lower_placement():
    # Single relevant id demoted from rank 1 to rank 3:
    # DCG = 1/log2(4); IDCG = 1/log2(2) = 1 -> ndcg = 1/log2(4) = 0.5
    assert math.isclose(ndcg_at_k(["x", "y", "a"], {"a"}, k=3), 1.0 / math.log2(4))


def test_ndcg_empty_relevant_is_zero():
    assert ndcg_at_k(["a", "b"], set(), k=3) == 0.0
