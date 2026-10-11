"""Reranker: reorders candidate passages by query relevance.

Retrieval (dense/sparse) optimizes recall with cheap scoring. A reranker then does a more
precise relevance judgement over the shortlist. In real mode this would be a cross-encoder
or an LLM relevance call; here we ship a deterministic lexical reranker so the pipeline is
testable offline. The interface is identical, so swapping in a real model is isolated.
"""

from __future__ import annotations

from app.domain import ScoredChunk
from app.retrieval.bm25 import tokenize

# Common English stopwords removed so generic words ("of", "the") don't create false
# relevance. Keeping this small and explicit avoids an extra NLP dependency.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "do",
    "does",
    "did",
    "i",
    "you",
    "it",
    "this",
    "that",
    "with",
    "as",
    "at",
    "by",
    "from",
    "what",
    "which",
    "who",
    "how",
    "many",
    "much",
    "my",
    "me",
}


def _terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1}


class LexicalReranker:
    """Deterministic cross-encoder stand-in using term-overlap + coverage."""

    name = "lexical"

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        q_terms = _terms(query)
        if not q_terms:
            return candidates[:top_k]
        rescored: list[ScoredChunk] = []
        for c in candidates:
            doc_set = _terms(c.chunk.text)
            if not doc_set:
                rel = 0.0
            else:
                overlap = len(q_terms & doc_set)
                coverage = overlap / len(q_terms)  # how much of the query is covered
                density = overlap / len(doc_set)  # signal-to-noise within the passage
                rel = 0.7 * coverage + 0.3 * density
            rescored.append(ScoredChunk(chunk=c.chunk, score=rel, source="rerank", rank=0))
        rescored.sort(key=lambda x: x.score, reverse=True)
        for rank, c in enumerate(rescored):
            c.rank = rank
        return rescored[:top_k]


def create_reranker() -> LexicalReranker:
    return LexicalReranker()
