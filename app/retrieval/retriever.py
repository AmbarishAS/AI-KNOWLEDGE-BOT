"""Retriever: dense, sparse (BM25), and hybrid (RRF) retrieval.

Design choices (interview-relevant):
- **Dense** finds semantic matches; **sparse/BM25** finds exact-term matches. **Hybrid**
  fuses both with **Reciprocal Rank Fusion (RRF)**, which combines rankings without needing
  to calibrate score scales across the two systems.
- **Metadata/tenant filtering** is passed down to the store and applied before scoring.
- A **score threshold** supports *abstention*: if nothing clears the bar, the pipeline can
  refuse to answer instead of hallucinating.
- **Parent-child expansion**: when a child chunk is retrieved we can swap in its parent for
  richer generation context.
"""

from __future__ import annotations

from app.config import RetrievalMode, Settings
from app.domain import Chunk, ScoredChunk
from app.retrieval.bm25 import BM25, tokenize
from app.vectorstore.base import VectorStore

RRF_K = 60  # standard RRF constant


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder, settings: Settings) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._settings = settings

    async def _dense(self, query: str, tenant: str, top_k: int, filters: dict | None) -> list[ScoredChunk]:
        qvec = await self._embedder.embed_query(query)
        return await self._store.dense_search(qvec, top_k, tenant, filters)

    async def _sparse(self, query: str, tenant: str, top_k: int, filters: dict | None) -> list[ScoredChunk]:
        chunks = await self._store.all_chunks(tenant)
        if filters:
            chunks = [c for c in chunks if _match_filters(c, filters)]
        if not chunks:
            return []
        corpus = [tokenize(c.text) for c in chunks]
        bm25 = BM25(corpus)
        ranked = bm25.rank(tokenize(query), top_k)
        # Normalize BM25 scores to 0..1 for comparability/thresholds.
        max_score = ranked[0][1] if ranked else 1.0
        results = []
        for rank, (idx, score) in enumerate(ranked):
            results.append(
                ScoredChunk(
                    chunk=chunks[idx],
                    score=score / (max_score or 1.0),
                    source="sparse",
                    rank=rank,
                )
            )
        return results

    @staticmethod
    def _rrf(result_lists: list[list[ScoredChunk]], top_k: int) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for results in result_lists:
            for rank, sc in enumerate(results):
                cid = sc.chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                chunks[cid] = sc.chunk
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk=chunks[cid], score=score, source="hybrid", rank=rank)
            for rank, (cid, score) in enumerate(fused)
        ]

    async def retrieve(
        self,
        query: str,
        tenant: str,
        top_k: int | None = None,
        filters: dict | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[ScoredChunk]:
        top_k = top_k or self._settings.top_k
        mode = mode or self._settings.retrieval_mode
        # Fetch a larger candidate pool before fusion/threshold.
        pool = max(top_k * 4, 20)
        if mode == RetrievalMode.DENSE:
            return await self._dense(query, tenant, top_k, filters)
        if mode == RetrievalMode.SPARSE:
            return await self._sparse(query, tenant, top_k, filters)
        dense = await self._dense(query, tenant, pool, filters)
        sparse = await self._sparse(query, tenant, pool, filters)
        return self._rrf([dense, sparse], top_k)

    async def retrieve_multi(
        self, queries: list[str], tenant: str, top_k: int | None = None, filters: dict | None = None
    ) -> list[ScoredChunk]:
        """Multi-query retrieval: run several query variants and fuse with RRF."""
        top_k = top_k or self._settings.top_k
        lists = [await self.retrieve(q, tenant, top_k * 2, filters) for q in queries]
        return self._rrf(lists, top_k)

    async def expand_parents(self, results: list[ScoredChunk]) -> list[ScoredChunk]:
        """Replace child chunks with their parent for richer generation context."""
        seen_parents: set[str] = set()
        expanded: list[ScoredChunk] = []
        for sc in results:
            parent_id = sc.chunk.parent_id
            if parent_id and parent_id not in seen_parents:
                parent = await self._store.get(parent_id)
                if parent is not None:
                    seen_parents.add(parent_id)
                    expanded.append(ScoredChunk(chunk=parent, score=sc.score, source="parent"))
                    continue
            expanded.append(sc)
        return expanded


def _match_filters(chunk: Chunk, filters: dict) -> bool:
    for k, v in filters.items():
        actual = getattr(chunk, k, None)
        if actual is None:
            actual = chunk.extra.get(k)
        if str(actual) != str(v):
            return False
    return True


def apply_threshold(results: list[ScoredChunk], threshold: float) -> list[ScoredChunk]:
    """Drop results below the score threshold (supports abstention)."""
    return [r for r in results if r.score >= threshold]
