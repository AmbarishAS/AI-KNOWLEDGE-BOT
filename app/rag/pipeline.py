"""RAG pipeline: the heart of grounded question answering.

Orchestrates: (optional) query rewrite/multi-query → retrieve → threshold/abstain →
rerank → parent expansion → context construction (dedup + token budget) → generate →
citation validation → Answer. Each enhancement is **configurable** so it can be ablated in
the evaluation suite (see ``docs/EVALUATION.md``).
"""

from __future__ import annotations

import time

from app.config import Settings
from app.domain import Answer, Message, ScoredChunk, TokenUsage
from app.llm.base import LLMProvider
from app.llm.mock import ABSTAIN
from app.llm.tokens import count_tokens, estimate_cost
from app.logging_config import get_logger
from app.rag.citations import build_citations, is_grounded, strip_invalid_markers
from app.rag.prompts import build_rag_messages
from app.retrieval.reranker import LexicalReranker
from app.retrieval.retriever import Retriever, apply_threshold

logger = get_logger(__name__)


def _normalize_scores(passages: list[ScoredChunk]) -> list[ScoredChunk]:
    """Scale scores to 0..1 by the top score so thresholds are comparable."""
    if not passages:
        return passages
    top = max((p.score for p in passages), default=1.0) or 1.0
    return [ScoredChunk(chunk=p.chunk, score=p.score / top, source=p.source, rank=p.rank) for p in passages]


def deduplicate(passages: list[ScoredChunk]) -> list[ScoredChunk]:
    """Drop exact/near-duplicate passages (same chunk id or identical text)."""
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    out: list[ScoredChunk] = []
    for p in passages:
        key = p.chunk.text.strip().lower()
        if p.chunk.chunk_id in seen_ids or key in seen_text:
            continue
        seen_ids.add(p.chunk.chunk_id)
        seen_text.add(key)
        out.append(p)
    return out


def fit_token_budget(passages: list[ScoredChunk], budget_tokens: int, model: str) -> list[ScoredChunk]:
    """Keep top passages until the token budget is exhausted."""
    out: list[ScoredChunk] = []
    used = 0
    for p in passages:
        cost = count_tokens(p.chunk.text, model)
        if used + cost > budget_tokens and out:
            break
        used += cost
        out.append(p)
    return out


class RAGConfig:
    """Per-request overrides so experiments/ablations can toggle features."""

    def __init__(
        self,
        rerank: bool | None = None,
        multi_query: bool = False,
        expand_parents: bool = False,
        top_k: int | None = None,
        context_budget_tokens: int = 2000,
    ) -> None:
        self.rerank = rerank
        self.multi_query = multi_query
        self.expand_parents = expand_parents
        self.top_k = top_k
        self.context_budget_tokens = context_budget_tokens


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        settings: Settings,
        reranker: LexicalReranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings
        self._reranker = reranker or LexicalReranker()

    async def _rewrite(self, question: str) -> list[str]:
        if not self._settings.enable_query_rewrite:
            return [question]
        # In mock mode we keep it deterministic; real mode could call the LLM.
        return [question]

    async def answer(
        self,
        question: str,
        tenant: str,
        history: list[Message] | None = None,
        config: RAGConfig | None = None,
        filters: dict | None = None,
    ) -> Answer:
        t0 = time.perf_counter()
        cfg = config or RAGConfig()
        top_k = cfg.top_k or self._settings.top_k
        rerank = self._settings.rerank_enabled if cfg.rerank is None else cfg.rerank

        # 1) Retrieve (optionally multi-query).
        queries = await self._rewrite(question)
        if cfg.multi_query and len(queries) > 1:
            candidates = await self._retriever.retrieve_multi(queries, tenant, top_k * 3, filters)
        else:
            candidates = await self._retriever.retrieve(question, tenant, top_k * 3, filters)

        if not candidates:
            return self._abstain(t0, reason="no_candidates")

        # 2) Score for relevance. The reranker produces calibrated 0..1 relevance, which is
        #    what we threshold on for abstention. (Raw RRF/dense scores are not comparable
        #    across modes, so thresholding happens *after* reranking.)
        if rerank:
            scored = self._reranker.rerank(question, candidates, top_k)
        else:
            scored = _normalize_scores(candidates[:top_k])

        # 3) Threshold → abstain if nothing clears the relevance bar.
        scored = apply_threshold(scored, self._settings.score_threshold)
        if not scored:
            return self._abstain(t0, reason="below_threshold")
        candidates = scored

        # 4) Parent-child expansion for richer context.
        if cfg.expand_parents:
            candidates = await self._retriever.expand_parents(candidates)

        # 5) Context construction: dedupe + token budget.
        passages = deduplicate(candidates)
        passages = fit_token_budget(passages, cfg.context_budget_tokens, self._model_name())
        if not passages:
            return self._abstain(t0, reason="empty_context")

        # 6) Generate grounded answer.
        messages = build_rag_messages(question, passages, history)
        result = await self._llm.chat(
            messages,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )

        # 7) Validate citations; detect abstention.
        answer_text = strip_invalid_markers(result.text, len(passages))
        abstained = ABSTAIN[:20].lower() in answer_text.lower() or not is_grounded(answer_text, passages)
        citations = build_citations(answer_text, passages)
        latency_ms = (time.perf_counter() - t0) * 1000
        cost = estimate_cost(result.usage.prompt_tokens, result.usage.completion_tokens, result.model)
        logger.info(
            "rag answer",
            extra={
                "extra": {
                    "tenant": tenant,
                    "abstained": abstained,
                    "num_passages": len(passages),
                    "num_citations": len(citations),
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "latency_ms": round(latency_ms, 1),
                    "cost_usd": cost,
                }
            },
        )
        return Answer(
            text=answer_text,
            citations=citations,
            abstained=abstained,
            provider=result.provider,
            mode=self._settings.app_mode.value,
            usage=result.usage,
            estimated_cost_usd=cost,
            latency_ms=latency_ms,
            retrieved=passages,
        )

    def _model_name(self) -> str:
        return getattr(self._llm, "model", "mock")

    def _abstain(self, t0: float, reason: str) -> Answer:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("rag abstain", extra={"extra": {"reason": reason}})
        return Answer(
            text=ABSTAIN,
            citations=[],
            abstained=True,
            provider=self._llm.name,
            mode=self._settings.app_mode.value,
            usage=TokenUsage(),
            latency_ms=latency_ms,
        )
