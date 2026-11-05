"""Claim-level grounding scores: is a piece of text supported by the context?

Two metrics share one primitive (``claim_support``):

- ``faithfulness(answer, contexts)`` — the share of a *generated* answer's
  claims that the retrieved context supports. Low = the model said things the
  sources don't back (hallucination or outside knowledge).
- ``context_recall(reference, contexts)`` — the share of a *reference* answer's
  claims that the retrieved context supports. Low = retrieval didn't surface what
  a correct answer needs. Needs no LLM, so it runs in CI.

A claim is a sentence. It is "supported" when at least ``threshold`` of its
content words (lower-cased, stop-words and very short tokens dropped, crude
suffix stemming) appear in a single context passage. This is a deliberately
simple **lexical** proxy, in the spirit of RAGAS faithfulness / context recall
but without an LLM judge. Known limits — read the numbers with these in mind:

- paraphrase is under-credited (synonyms don't match), so scores skew low for
  abstractive answers;
- negation and numbers are not understood ("is" vs "is not" both match);
- support from *combining* two passages is not credited.

Treat it as a regression signal on a fixed golden set, not an absolute truth.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from had has have how i if in
    into is it its may might more most no not of on or our so such than that the their them
    then there these they this those to too was we were what when where which while who why
    will with would you your also just only very each any all some""".split()
)

DEFAULT_THRESHOLD = 0.6


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    """Normalised content-word set used for matching."""
    return {
        _stem(w)
        for w in _WORD_RE.findall(text.lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def split_claims(text: str) -> list[str]:
    """Split text into sentence-level claims, dropping inline [n] citations and
    fragments with fewer than two content words (headings, "Yes.", etc.)."""
    cleaned = _CITATION_RE.sub("", text)
    claims = []
    for raw in _SENT_RE.split(cleaned):
        s = raw.strip(" -*#\t")
        if len(content_words(s)) >= 2:
            claims.append(s)
    return claims


def claim_support(claim: str, contexts: Sequence[str]) -> float:
    """Best fraction of the claim's content words found in any one context."""
    words = content_words(claim)
    if not words:
        return 0.0
    best = 0.0
    for ctx in contexts:
        overlap = len(words & content_words(ctx)) / len(words)
        best = max(best, overlap)
    return best


@dataclass
class GroundingResult:
    score: float
    claims: list[str]
    supported: list[bool]

    @property
    def unsupported(self) -> list[str]:
        return [c for c, ok in zip(self.claims, self.supported) if not ok]


def grounding(
    text: str, contexts: Sequence[str], threshold: float = DEFAULT_THRESHOLD
) -> GroundingResult:
    """Score how much of ``text`` the ``contexts`` support (0..1).

    Text with no scoreable claims scores 1.0 (nothing asserted, nothing
    unsupported); text with claims but no contexts scores 0.0.
    """
    claims = split_claims(text)
    if not claims:
        return GroundingResult(score=1.0, claims=[], supported=[])
    supported = [claim_support(c, contexts) >= threshold for c in claims]
    return GroundingResult(
        score=sum(supported) / len(claims), claims=claims, supported=supported
    )


def faithfulness(answer: str, contexts: Sequence[str], threshold: float = DEFAULT_THRESHOLD) -> float:
    """Share of the generated answer's claims supported by the retrieved context."""
    return grounding(answer, contexts, threshold).score


def context_recall(
    reference: str, contexts: Sequence[str], threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Share of the reference answer's claims supported by the retrieved context."""
    return grounding(reference, contexts, threshold).score
