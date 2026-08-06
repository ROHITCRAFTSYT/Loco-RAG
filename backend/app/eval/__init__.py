"""Retrieval-quality evaluation for the RAG pipeline.

``metrics`` holds pure IR metrics (hit-rate, MRR, recall, precision, MAP, nDCG);
``harness`` aggregates them over a labelled question set against any retriever.
Kept dependency-light so the metrics unit-test without an embedder or store.
"""
