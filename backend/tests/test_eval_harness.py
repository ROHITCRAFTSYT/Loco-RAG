"""Eval harness aggregation.

Uses a stub retriever (a dict of query -> ranked ids) so the harness is exercised
with no embedder or store. Pins the per-case scores, the mean aggregation across
cases, the empty-suite behaviour, and JSON round-tripping of a dataset.
"""
from __future__ import annotations

import json
import math

from app.eval.harness import EvalCase, evaluate, load_cases


def _stub(mapping):
    return lambda query: mapping.get(query, [])


def test_evaluate_aggregates_means_across_cases():
    cases = [
        EvalCase(query="q1", relevant=["a"]),
        EvalCase(query="q2", relevant=["b"]),
    ]
    # q1: relevant 'a' at rank 1 (perfect). q2: relevant 'b' at rank 2.
    retrieve = _stub({"q1": ["a", "z"], "q2": ["x", "b"]})
    report = evaluate(cases, retrieve, k=5)

    assert report.k == 5
    assert len(report.cases) == 2
    # MRR = mean(1/1, 1/2) = 0.75
    assert math.isclose(report.aggregate["mrr"], 0.75)
    # hit_rate = mean(1, 1) = 1.0 (both relevant ids are within k=5)
    assert report.aggregate["hit_rate"] == 1.0
    # every metric is reported
    assert set(report.aggregate) == {"hit_rate", "mrr", "recall", "precision", "map", "ndcg"}


def test_k_bounds_hit_rate():
    cases = [EvalCase(query="q", relevant=["b"])]
    retrieve = _stub({"q": ["a", "b"]})  # relevant at rank 2
    assert evaluate(cases, retrieve, k=1).aggregate["hit_rate"] == 0.0
    assert evaluate(cases, retrieve, k=2).aggregate["hit_rate"] == 1.0


def test_evaluate_empty_suite_is_all_zero():
    report = evaluate([], _stub({}), k=3)
    assert report.cases == []
    assert all(v == 0.0 for v in report.aggregate.values())


def test_report_to_dict_and_table_render():
    cases = [EvalCase(query="q1", relevant=["a"])]
    report = evaluate(cases, _stub({"q1": ["a"]}), k=3)
    d = report.to_dict()
    assert d["k"] == 3
    assert d["cases"][0]["query"] == "q1"
    assert "mrr" in report.format_table()


def test_load_cases_round_trip(tmp_path):
    data = [{"query": "what is x", "relevant": ["doc-a.md", "doc-b.md"]}]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].query == "what is x"
    assert cases[0].relevant == ["doc-a.md", "doc-b.md"]
