"""Vector store factory with a process-local singleton.

The singleton ensures ingestion and retrieval share the same in-memory index within a
process. Tests can reset it via ``reset_vector_store``.
"""

from __future__ import annotations

from app.config import Settings
from app.vectorstore.base import VectorStore
from app.vectorstore.memory import InMemoryVectorStore

_instance: VectorStore | None = None


def create_vector_store(settings: Settings, dim: int) -> VectorStore:
    global _instance
    if _instance is not None:
        return _instance
    if settings.vector_store == "qdrant":
        from app.vectorstore.qdrant_store import QdrantVectorStore

        _instance = QdrantVectorStore(url=settings.qdrant_url, collection=settings.qdrant_collection, dim=dim)
    else:
        _instance = InMemoryVectorStore()
    return _instance


async def reset_vector_store() -> None:
    global _instance
    if isinstance(_instance, InMemoryVectorStore):
        await _instance.clear()
    _instance = None
