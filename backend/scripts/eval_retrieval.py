"""Evaluate the real RAG pipeline against the labelled golden set.

Ingests ``eval/corpus`` into a throwaway collection, runs the actual
``rag.retrieve`` for every question in ``eval/dataset.json`` and reports:

* **Retrieval** (document level, no LLM): recall@k, hit-rate, MRR, nDCG, … at
  each ``--k`` cut-off. Chunks of the same document collapse to one entry ranked
  by its best chunk.
* **Context recall** (no LLM): share of each reference answer's claims that the
  top ``RERANK_TOP_N`` chunks (what production actually sends the model) support.
* **Answer faithfulness** (``--answers``, needs a running Ollama/LM Studio): the
  model answers each question from that same context; the share of its claims the
  context supports is scored.

Grounding scores are lexical proxies — see ``app/eval/faithfulness.py`` for what
that does and doesn't catch.

Usage:
    python -m scripts.eval_retrieval
    python -m scripts.eval_retrieval --k 1,3,5 --json report.json --markdown report.md
    python -m scripts.eval_retrieval --answers --model llama3.1
    python -m scripts.eval_retrieval --min-recall 0.9 --min-context-recall 0.8   # CI gate

Exits 1 when a ``--min-*`` threshold is missed. The eval collection is dropped
afterwards so a run leaves no residue in the vector store.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.eval.faithfulness import grounding
from app.eval.harness import EvalCase, EvalReport, evaluate_multi, load_cases
from app.schemas import Source
from app.services import rag
from app.services.ingest import ingest_document
from app.services.vectorstore import get_vectorstore

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
DEFAULT_CORPUS = EVAL_DIR / "corpus"
DEFAULT_DATASET = EVAL_DIR / "dataset.json"
EVAL_COLLECTION = "eval-retrieval-scratch"

ANSWER_SYSTEM = "Answer in two or three sentences using only the sources."


@dataclass
class RunResult:
    retrieval: dict[int, EvalReport]
    context_top_n: int
    context_recall: float | None
    faithfulness: float | None = None
    latency_ms_p50: float = 0.0
    per_case: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "retrieval": {str(k): r.to_dict() for k, r in self.retrieval.items()},
            "context_top_n": self.context_top_n,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "latency_ms_p50": self.latency_ms_p50,
            "cases": self.per_case,
        }

    def format_table(self) -> str:
        ks = sorted(self.retrieval)
        n = len(next(iter(self.retrieval.values())).cases) if self.retrieval else 0
        lines = [f"Evaluation over {n} queries", ""]
        for metric in ("recall", "hit_rate", "mrr", "ndcg"):
            cells = "  ".join(f"@{k}={self.retrieval[k].aggregate[metric]:.3f}" for k in ks)
            lines.append(f"  {metric:<16}{cells}")
        if self.context_recall is not None:
            lines.append(f"  {'context_recall':<16}{self.context_recall:.3f}  (top-{self.context_top_n} chunks)")
        if self.faithfulness is not None:
            lines.append(f"  {'faithfulness':<16}{self.faithfulness:.3f}")
        lines.append(f"  {'retrieve p50':<16}{self.latency_ms_p50:.1f} ms")
        return "\n".join(lines)

    def format_markdown(self) -> str:
        ks = sorted(self.retrieval)
        head = "| Metric | " + " | ".join(f"@{k}" for k in ks) + " |"
        sep = "|---|" + "---|" * len(ks)
        rows = [
            f"| {m} | " + " | ".join(f"{self.retrieval[k].aggregate[m]:.3f}" for k in ks) + " |"
            for m in ("recall", "hit_rate", "mrr", "ndcg")
        ]
        extra = []
        if self.context_recall is not None:
            extra.append(f"| context recall (top-{self.context_top_n}) | {self.context_recall:.3f} |")
        if self.faithfulness is not None:
            extra.append(f"| answer faithfulness | {self.faithfulness:.3f} |")
        extra.append(f"| retrieve latency p50 | {self.latency_ms_p50:.1f} ms |")
        return "\n".join(
            [head, sep, *rows, "", "| Metric | Value |", "|---|---|", *extra]
        )


def _ingest_corpus(corpus_dir: Path, collection: str) -> int:
    files = sorted(p for p in corpus_dir.glob("*") if p.is_file())
    for path in files:
        ingest_document(
            document_id=path.name,
            collection=collection,
            filename=path.name,
            content_type="text/markdown",
            data=path.read_bytes(),
        )
    return len(files)


def _docs(sources: list[Source]) -> list[str]:
    """Collapse chunks to documents, keeping each document's best rank."""
    ranked: list[str] = []
    for s in sources:
        if s.document not in ranked:
            ranked.append(s.document)
    return ranked


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


async def _answer(question: str, context: list[Source], model: str, provider: str) -> str:
    from app.services.llm import get_provider

    messages = [
        {"role": "system", "content": f"{ANSWER_SYSTEM}\n\n{rag.build_context(context)}"},
        {"role": "user", "content": question},
    ]
    return await get_provider(provider).complete(messages, model=model, temperature=0.0)


def run(
    cases: list[EvalCase],
    corpus_dir: Path,
    ks: list[int],
    *,
    answers: bool = False,
    model: str | None = None,
    provider: str | None = None,
) -> RunResult:
    store = get_vectorstore()
    try:
        store.drop_collection(EVAL_COLLECTION)  # start clean if a prior run died
    except Exception:
        pass

    # Retrieve deep enough to score the largest k, but score context recall /
    # faithfulness on the production context size.
    production_top_n = settings.rerank_top_n
    saved = (settings.rerank_top_n, settings.retrieve_top_k)
    depth = max(max(ks), production_top_n)
    settings.rerank_top_n = depth
    settings.retrieve_top_k = max(settings.retrieve_top_k, depth)

    n = _ingest_corpus(corpus_dir, EVAL_COLLECTION)
    print(f"Ingested {n} documents into '{EVAL_COLLECTION}' (retrieval depth {depth}).")
    try:
        retrieved: dict[str, list[Source]] = {}
        latencies: list[float] = []
        for c in cases:
            t0 = time.perf_counter()
            retrieved[c.query] = rag.retrieve(c.query, EVAL_COLLECTION)
            latencies.append((time.perf_counter() - t0) * 1000)

        reports = evaluate_multi(cases, lambda q: _docs(retrieved[q]), ks)

        per_case: list[dict] = []
        recalls: list[float] = []
        faiths: list[float] = []
        for c in cases:
            context = retrieved[c.query][:production_top_n]
            texts = [s.text for s in context]
            row: dict = {"query": c.query, "ranked_docs": _docs(retrieved[c.query])}
            if c.reference:
                g = grounding(c.reference, texts)
                recalls.append(g.score)
                row["context_recall"] = g.score
                row["reference_unsupported"] = g.unsupported
            if answers:
                answer = asyncio.run(
                    _answer(
                        c.query,
                        context,
                        model or settings.default_model,
                        provider or settings.default_provider,
                    )
                )
                g = grounding(answer, texts)
                faiths.append(g.score)
                row.update(answer=answer, faithfulness=g.score, answer_unsupported=g.unsupported)
            per_case.append(row)

        return RunResult(
            retrieval=reports,
            context_top_n=production_top_n,
            context_recall=sum(recalls) / len(recalls) if recalls else None,
            faithfulness=sum(faiths) / len(faiths) if faiths else None,
            latency_ms_p50=round(_median(latencies), 1),
            per_case=per_case,
        )
    finally:
        settings.rerank_top_n, settings.retrieve_top_k = saved
        try:
            store.drop_collection(EVAL_COLLECTION)
        except Exception:
            pass


def check_thresholds(
    result: RunResult,
    *,
    min_recall: float | None,
    min_context_recall: float | None,
    min_faithfulness: float | None,
) -> list[str]:
    """Return a human-readable failure per missed threshold (empty = pass).

    ``min_recall`` is checked at the largest k."""
    failures = []
    k = max(result.retrieval)
    recall = result.retrieval[k].aggregate["recall"]
    if min_recall is not None and recall < min_recall:
        failures.append(f"recall@{k} {recall:.3f} < {min_recall}")
    if min_context_recall is not None and (result.context_recall or 0.0) < min_context_recall:
        failures.append(f"context_recall {result.context_recall or 0.0:.3f} < {min_context_recall}")
    if min_faithfulness is not None and (result.faithfulness or 0.0) < min_faithfulness:
        failures.append(f"faithfulness {result.faithfulness or 0.0:.3f} < {min_faithfulness}")
    return failures


def _parse_ks(raw: str) -> list[int]:
    ks = sorted({int(x) for x in raw.split(",") if x.strip()})
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError("--k needs positive integers, e.g. 1,3,5")
    return ks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer grounding.")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--k", type=_parse_ks, default=[1, 3, 5], help="comma-separated cut-offs")
    ap.add_argument("--answers", action="store_true", help="generate answers and score faithfulness (needs a running LLM)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    ap.add_argument("--markdown", type=Path, default=None, help="write a markdown summary here")
    ap.add_argument("--min-recall", type=float, default=None)
    ap.add_argument("--min-context-recall", type=float, default=None)
    ap.add_argument("--min-faithfulness", type=float, default=None)
    args = ap.parse_args(argv)

    cases = load_cases(args.dataset)
    result = run(
        cases, args.corpus, args.k, answers=args.answers, model=args.model, provider=args.provider
    )
    print()
    print(result.format_table())
    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.json}")
    if args.markdown:
        args.markdown.write_text(result.format_markdown() + "\n", encoding="utf-8")

    failures = check_thresholds(
        result,
        min_recall=args.min_recall,
        min_context_recall=args.min_context_recall,
        min_faithfulness=args.min_faithfulness,
    )
    for f in failures:
        print(f"THRESHOLD FAILED: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
