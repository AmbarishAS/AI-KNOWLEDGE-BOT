"""Qdrant-backed vector store (optional; used when VECTOR_STORE=qdrant).

Guarded import: if ``qdrant-client`` is not installed we raise a clear error at
construction time, so the default in-memory path never depends on it. Qdrant provides
production-grade filtering and persistence. For sparse/BM25 we keep a local text mirror
(the retriever builds BM25 from ``all_chunks``) — documented in ``docs/RAG_DEEP_DIVE.md``.
"""

from __future__ import annotations

import uuid

from app.domain import Chunk, ScoredChunk
from app.vectorstore.base import StoredRecord


def _point_id(chunk_id: str) -> str:
    # Qdrant point ids must be UUIDs or ints; derive a stable UUID from the chunk id.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    name = "qdrant"

    def __init__(self, url: str, collection: str, dim: int) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("qdrant-client is not installed. Install it or set VECTOR_STORE=memory.") from exc
        self._qm = qm
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._dim = dim
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

    async def upsert(self, records: list[StoredRecord]) -> None:
        qm = self._qm
        points = [
            qm.PointStruct(
                id=_point_id(r.chunk.chunk_id),
                vector=r.vector,
                payload={
                    "chunk": r.chunk.model_dump(),
                    "tenant": r.chunk.tenant,
                    "document_id": r.chunk.document_id,
                    "chunk_id": r.chunk.chunk_id,
                },
            )
            for r in records
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def _filter(self, tenant: str, filters: dict[str, str] | None):
        qm = self._qm
        must = [qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant))]
        for k, v in (filters or {}).items():
            must.append(qm.FieldCondition(key=f"chunk.{k}", match=qm.MatchValue(value=v)))
        return qm.Filter(must=must)

    async def dense_search(self, query_vector, top_k, tenant, filters=None) -> list[ScoredChunk]:
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=self._filter(tenant, filters),
            limit=top_k,
        )
        out: list[ScoredChunk] = []
        for rank, h in enumerate(hits):
            chunk = Chunk(**h.payload["chunk"])
            out.append(ScoredChunk(chunk=chunk, score=float(h.score), source="dense", rank=rank))
        return out

    async def all_chunks(self, tenant: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=self._filter(tenant, None),
                with_payload=True,
                limit=256,
                offset=offset,
            )
            chunks.extend(Chunk(**p.payload["chunk"]) for p in points)
            if offset is None:
                break
        return chunks

    async def get(self, chunk_id: str) -> Chunk | None:
        res = self._client.retrieve(collection_name=self._collection, ids=[_point_id(chunk_id)], with_payload=True)
        if not res:
            return None
        return Chunk(**res[0].payload["chunk"])

    async def delete_document(self, tenant: str, document_id: str) -> int:
        qm = self._qm
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant)),
                qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id)),
            ]
        )
        self._client.delete(collection_name=self._collection, points_selector=flt)
        return 1

    async def count(self) -> int:
        return self._client.count(collection_name=self._collection).count
