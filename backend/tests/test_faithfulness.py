"""Lexical claim-grounding metrics (faithfulness / context recall)."""
from __future__ import annotations

import math

from app.eval.faithfulness import (
    claim_support,
    content_words,
    context_recall,
    faithfulness,
    grounding,
    split_claims,
)

CTX = [
    "The default window is 512 tokens with 64 tokens of overlap between consecutive chunks.",
    "Retrieval merges dense and BM25 lists with Reciprocal Rank Fusion.",
]


def test_content_words_drop_stopwords_and_stem():
    assert content_words("The chunks are overlapping") == {"chunk", "overlapp"}


def test_split_claims_strips_citations_and_fragments():
    claims = split_claims("Chunks overlap by 64 tokens [1]. Yes.\n- RRF merges lists [1, 2].")
    assert claims == ["Chunks overlap by 64 tokens .", "RRF merges lists ."]


def test_supported_answer_scores_one():
    answer = "Consecutive chunks overlap by 64 tokens [1]. Dense and BM25 lists merge with Reciprocal Rank Fusion [2]."
    assert faithfulness(answer, CTX) == 1.0


def test_hallucinated_claim_lowers_score():
    answer = (
        "Consecutive chunks overlap by 64 tokens. "
        "The embeddings are stored in Pinecone on AWS servers."
    )
    g = grounding(answer, CTX)
    assert math.isclose(g.score, 0.5)
    assert g.unsupported == ["The embeddings are stored in Pinecone on AWS servers."]


def test_support_must_come_from_one_passage():
    # Half the words from each passage: not credited (combination isn't support).
    claim = "Window tokens overlap Reciprocal Rank Fusion"
    assert claim_support(claim, CTX) < 0.6


def test_edge_cases():
    assert faithfulness("", CTX) == 1.0  # nothing asserted
    assert context_recall("Chunks overlap by 64 tokens.", []) == 0.0  # nothing retrieved
