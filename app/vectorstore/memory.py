"""In-memory vector store using NumPy (default, zero-infra).

Great for tests/demos and small corpora. Cosine similarity is computed with a vectorized
dot product (vectors are L2-normalized at embed time). Metadata/tenant filtering is applied
*before* scoring so access scope is always enforced.

This is a process-local singleton so the API, ingestion, and retrieval share one index.
"""

from __future__ import annotations

import numpy as np

from app.domain import Chunk, ScoredChunk
from app.vectorstore.base import StoredRecord


class InMemoryVectorStore:
    name = "memory"

    def __init__(self) -> None:
        self._records: dict[str, StoredRecord] = {}

    async def upsert(self, records: list[StoredRecord]) -> None:
        for r in records:
            self._records[r.chunk.chunk_id] = r

    def _matches(self, chunk: Chunk, tenant: str, filters: dict[str, str] | None) -> bool:
        if chunk.tenant != tenant:
            return False
        if filters:
            for key, value in filters.items():
                actual = getattr(chunk, key, None)
                if actual is None:
                    actual = chunk.extra.get(key)
                if str(actual) != str(value):
                    return False
        return True

    async def dense_search(
        self,
        query_vector: list[float],
        top_k: int,
        tenant: str,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        candidates = [r for r in self._records.values() if self._matches(r.chunk, tenant, filters)]
        if not candidates:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        matrix = np.asarray([r.vector for r in candidates], dtype=np.float32)
        # Cosine similarity; guard against zero-norm vectors.
        qn = q / (np.linalg.norm(q) or 1.0)
        mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        scores = mn @ qn
        order = np.argsort(-scores)[:top_k]
        results: list[ScoredChunk] = []
        for rank, idx in enumerate(order):
            results.append(
                ScoredChunk(
                    chunk=candidates[int(idx)].chunk,
                    score=float(scores[int(idx)]),
                    source="dense",
                    rank=rank,
                )
            )
        return results

    async def all_chunks(self, tenant: str) -> list[Chunk]:
        return [r.chunk for r in self._records.values() if r.chunk.tenant == tenant]

    async def get(self, chunk_id: str) -> Chunk | None:
        r = self._records.get(chunk_id)
        return r.chunk if r else None

    async def delete_document(self, tenant: str, document_id: str) -> int:
        to_delete = [
            cid for cid, r in self._records.items() if r.chunk.tenant == tenant and r.chunk.document_id == document_id
        ]
        for cid in to_delete:
            del self._records[cid]
        return len(to_delete)

    async def count(self) -> int:
        return len(self._records)

    async def clear(self) -> None:
        self._records.clear()
