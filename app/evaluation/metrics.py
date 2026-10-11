"""Retrieval and generation evaluation metrics.

Two families (an important interview distinction):

- **Deterministic retrieval metrics** (Precision@k, Recall@k, MRR, Hit-Rate@k, nDCG@k) only
  need ground-truth relevant items — no LLM, fully reproducible.
- **Generation/RAG metrics** (groundedness, answer/context relevance, citation correctness,
  abstention correctness, completeness) are harder. We provide deterministic proxies here
  and document that higher-fidelity scores need an evaluator model (e.g. RAGAS) — whose
  outputs are estimates, not ground truth (see ``docs/EVALUATION.md``).
"""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg else 0.0


def aggregate_retrieval(per_query: list[dict]) -> dict:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}


# ---- Generation metrics (deterministic proxies) ----------------------------


def _tokens(text: str) -> set[str]:
    import re

    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def groundedness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer tokens supported by the provided context (proxy)."""
    a = _tokens(answer)
    if not a:
        return 0.0
    ctx = set().union(*[_tokens(c) for c in contexts]) if contexts else set()
    return round(len(a & ctx) / len(a), 4)


def answer_relevance(answer: str, question: str) -> float:
    """Lexical overlap proxy between answer and question terms."""
    q = _tokens(question)
    a = _tokens(answer)
    if not q:
        return 0.0
    return round(len(q & a) / len(q), 4)


def context_relevance(contexts: list[str], question: str) -> float:
    q = _tokens(question)
    if not q or not contexts:
        return 0.0
    ctx = set().union(*[_tokens(c) for c in contexts])
    return round(len(q & ctx) / len(q), 4)


def citation_correctness(cited_sources: list[str], relevant: set[str]) -> float:
    if not cited_sources:
        return 0.0
    return round(sum(1 for c in cited_sources if c in relevant) / len(cited_sources), 4)


def completeness(answer: str, reference: str) -> float:
    """Token recall of the reference answer within the generated answer (proxy)."""
    ref = _tokens(reference)
    if not ref:
        return 0.0
    return round(len(ref & _tokens(answer)) / len(ref), 4)
