"""Aggregate IR metrics over a labelled question set against any retriever.

``evaluate`` takes eval cases (a query plus the ids that should be retrieved)
and a ``retrieve`` callable ``query -> ranked ids``, and returns an
:class:`EvalReport` with per-case and mean-aggregated metrics. The retriever is
injected, so this layer has no dependency on the embedder or vector store and is
fully unit-testable with a stub; the CLI runner wires in the real pipeline.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.eval import metrics

#: A retriever reduces a query to a ranked list of ids (best first).
Retriever = Callable[[str], Sequence[str]]

_METRICS = ("hit_rate", "mrr", "recall", "precision", "map", "ndcg")


@dataclass
class EvalCase:
    """One labelled question: the query and the ids that should be retrieved."""

    query: str
    relevant: list[str]


@dataclass
class CaseResult:
    query: str
    relevant: list[str]
    ranked: list[str]
    scores: dict[str, float]


@dataclass
class EvalReport:
    k: int
    cases: list[CaseResult]
    aggregate: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "aggregate": self.aggregate,
            "cases": [
                {
                    "query": c.query,
                    "relevant": c.relevant,
                    "ranked": c.ranked,
                    "scores": c.scores,
                }
                for c in self.cases
            ],
        }

    def format_table(self) -> str:
        """Render the aggregate metrics as a fixed-width table."""
        lines = [f"Retrieval evaluation over {len(self.cases)} queries (k={self.k})", ""]
        width = max((len(m) for m in self.aggregate), default=6)
        for name, value in self.aggregate.items():
            lines.append(f"  {name:<{width}}  {value:.3f}")
        return "\n".join(lines)


def score_case(query: str, relevant: Sequence[str], ranked: Sequence[str], k: int) -> dict[str, float]:
    """Compute the full metric bundle for a single query's ranking."""
    return {
        "hit_rate": metrics.hit_rate_at_k(ranked, relevant, k),
        "mrr": metrics.reciprocal_rank(ranked, relevant),
        "recall": metrics.recall_at_k(ranked, relevant, k),
        "precision": metrics.precision_at_k(ranked, relevant, k),
        "map": metrics.average_precision(ranked, relevant),
        "ndcg": metrics.ndcg_at_k(ranked, relevant, k),
    }


def evaluate(cases: Sequence[EvalCase], retrieve: Retriever, k: int = 5) -> EvalReport:
    """Run ``retrieve`` over every case and aggregate the metrics as means."""
    results: list[CaseResult] = []
    for case in cases:
        ranked = list(retrieve(case.query))
        results.append(
            CaseResult(
                query=case.query,
                relevant=list(case.relevant),
                ranked=ranked,
                scores=score_case(case.query, case.relevant, ranked, k),
            )
        )

    aggregate = {
        name: (sum(r.scores[name] for r in results) / len(results) if results else 0.0)
        for name in _METRICS
    }
    return EvalReport(k=k, cases=results, aggregate=aggregate)


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load eval cases from a JSON file of ``[{"query": ..., "relevant": [...]}, ...]``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(query=item["query"], relevant=list(item["relevant"])) for item in raw]
