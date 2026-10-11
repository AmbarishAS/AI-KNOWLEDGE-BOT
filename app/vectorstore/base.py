"""Vector store interface.

The store holds the chunk, its dense vector, and its metadata payload. It supports:
- dense similarity search with **metadata/tenant filtering** (enforced at query time),
- listing a tenant's chunks (used to build the BM25 sparse index),
- lookup by id (used for parent-child retrieval),
- deletion by document (for the DELETE endpoint).
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from app.domain import Chunk, ScoredChunk


class StoredRecord(BaseModel):
    chunk: Chunk
    vector: list[float]


class VectorStore(Protocol):
    name: str

    async def upsert(self, records: list[StoredRecord]) -> None: ...

    async def dense_search(
        self,
        query_vector: list[float],
        top_k: int,
        tenant: str,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]: ...

    async def all_chunks(self, tenant: str) -> list[Chunk]: ...

    async def get(self, chunk_id: str) -> Chunk | None: ...

    async def delete_document(self, tenant: str, document_id: str) -> int: ...

    async def count(self) -> int: ...
