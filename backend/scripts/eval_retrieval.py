"""Run the retrieval evaluation against the real RAG pipeline.

Ingests the labelled eval corpus into a throwaway collection, runs the actual
``rag.retrieve`` for every labelled query, reduces the returned Sources to the
documents they came from, and scores document-level retrieval with the metric
harness. Prints an aggregate table and, with ``--json``, writes the full report.

Usage:
    python -m scripts.eval_retrieval                 # default corpus + dataset
    python -m scripts.eval_retrieval --k 3 --json out.json

The eval collection is dropped afterwards so a run leaves no residue in the
vector store.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.eval.harness import EvalReport, evaluate, load_cases
from app.services import rag
from app.services.ingest import ingest_document
from app.services.vectorstore import get_vectorstore

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
DEFAULT_CORPUS = EVAL_DIR / "corpus"
DEFAULT_DATASET = EVAL_DIR / "dataset.json"
EVAL_COLLECTION = "eval-retrieval-scratch"


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


def _doc_retriever(collection: str):
    """Wrap rag.retrieve as a query -> ranked document-id list.

    Multiple chunks of the same document collapse to one entry, keeping the rank
    of the document's best-scoring chunk.
    """
    def retrieve(query: str) -> list[str]:
        ranked_docs: list[str] = []
        for source in rag.retrieve(query, collection):
            if source.document not in ranked_docs:
                ranked_docs.append(source.document)
        return ranked_docs

    return retrieve


def run(corpus_dir: Path, dataset_path: Path, k: int) -> EvalReport:
    store = get_vectorstore()
    try:
        store.drop_collection(EVAL_COLLECTION)  # start clean if a prior run died
    except Exception:
        pass

    n = _ingest_corpus(corpus_dir, EVAL_COLLECTION)
    print(f"Ingested {n} documents into '{EVAL_COLLECTION}'.")
    try:
        cases = load_cases(dataset_path)
        return evaluate(cases, _doc_retriever(EVAL_COLLECTION), k=k)
    finally:
        try:
            store.drop_collection(EVAL_COLLECTION)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--json", type=Path, default=None, help="write the full report to this path")
    args = ap.parse_args()

    report = run(args.corpus, args.dataset, args.k)
    print()
    print(report.format_table())
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.json}")


if __name__ == "__main__":
    main()
