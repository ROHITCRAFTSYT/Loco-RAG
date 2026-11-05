"""The eval CLI runner: multi-k scoring, context recall, thresholds, reports.

rag.retrieve, ingestion and the store are stubbed so this runs without an
embedder; the live pipeline is exercised by the CI ``eval`` job.
"""
from __future__ import annotations

import json

import pytest

from app.config import settings
from app.eval.harness import EvalCase, evaluate_multi
from app.schemas import Source
from scripts import eval_retrieval as runner

DOCS = {
    "a.md": "Chunks overlap by 64 tokens between consecutive windows.",
    "b.md": "Reciprocal Rank Fusion merges dense and BM25 result lists.",
}


class NullStore:
    def drop_collection(self, name):
        pass


@pytest.fixture
def stubbed(monkeypatch):
    seen_depth = {}

    def fake_retrieve(query, collection):
        seen_depth["n"] = settings.rerank_top_n
        order = ["b.md", "a.md"] if "fusion" in query else ["a.md", "b.md"]
        return [Source(id=d, document=d, text=DOCS[d]) for d in order]

    monkeypatch.setattr(runner, "get_vectorstore", lambda: NullStore())
    monkeypatch.setattr(runner, "_ingest_corpus", lambda corpus, coll: len(DOCS))
    monkeypatch.setattr(runner.rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(settings, "rerank_top_n", 1, raising=False)
    return seen_depth


CASES = [
    EvalCase("how much do chunks overlap", ["a.md"], "Chunks overlap by 64 tokens."),
    EvalCase("what is fusion", ["b.md"], "Reciprocal Rank Fusion merges result lists."),
    EvalCase("overlap and fusion", ["a.md", "b.md"], None),
]


def test_run_scores_multiple_k_and_context_recall(stubbed, tmp_path):
    result = runner.run(CASES, tmp_path, [1, 2])
    assert stubbed["n"] == 2  # retrieved deep enough for the largest k
    assert settings.rerank_top_n == 1  # and restored afterwards
    assert result.retrieval[2].aggregate["recall"] == 1.0
    # case 3 needs two docs; only one fits in k=1 -> recall 0.5 for that case
    assert result.retrieval[1].aggregate["recall"] == pytest.approx((1 + 1 + 0.5) / 3)
    # context recall uses the production top-n (1 chunk): both references supported
    assert result.context_top_n == 1
    assert result.context_recall == 1.0
    assert result.faithfulness is None


def test_thresholds(stubbed, tmp_path):
    result = runner.run(CASES, tmp_path, [1])
    assert runner.check_thresholds(result, min_recall=0.5, min_context_recall=0.9, min_faithfulness=None) == []
    failures = runner.check_thresholds(result, min_recall=0.99, min_context_recall=None, min_faithfulness=0.5)
    assert len(failures) == 2 and failures[0].startswith("recall@1")


def test_main_writes_reports_and_exit_code(stubbed, tmp_path):
    dataset = tmp_path / "ds.json"
    dataset.write_text(json.dumps(
        [{"query": c.query, "relevant": c.relevant, "reference": c.reference} for c in CASES]
    ))
    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    rc = runner.main([
        "--dataset", str(dataset), "--corpus", str(tmp_path), "--k", "1,2",
        "--json", str(out_json), "--markdown", str(out_md), "--min-recall", "0.9",
    ])
    assert rc == 0
    assert json.loads(out_json.read_text())["context_recall"] == 1.0
    assert "| recall |" in out_md.read_text()
    assert runner.main(["--dataset", str(dataset), "--corpus", str(tmp_path), "--k", "1",
                        "--min-recall", "0.99"]) == 1


def test_answers_mode_scores_faithfulness(stubbed, tmp_path, monkeypatch):
    async def fake_answer(question, context, model, provider):
        return context[0].text + " It also runs on quantum hardware in orbit."

    monkeypatch.setattr(runner, "_answer", fake_answer)
    result = runner.run(CASES[:2], tmp_path, [1], answers=True)
    assert result.faithfulness == pytest.approx(0.5)
    assert result.per_case[0]["answer_unsupported"]


def test_evaluate_multi_calls_retriever_once_per_case():
    calls = []

    def retrieve(q):
        calls.append(q)
        return ["a"]

    reports = evaluate_multi([EvalCase("q", ["a"])], retrieve, [1, 3, 5])
    assert calls == ["q"]
    assert set(reports) == {1, 3, 5}


def test_bad_k_argument():
    with pytest.raises(SystemExit):
        runner.main(["--k", "0"])
