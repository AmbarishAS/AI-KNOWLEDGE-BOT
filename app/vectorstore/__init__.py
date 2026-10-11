"""Vector store: interface, in-memory default, optional Qdrant."""

from app.vectorstore.base import StoredRecord, VectorStore
from app.vectorstore.factory import create_vector_store, reset_vector_store
from app.vectorstore.memory import InMemoryVectorStore

__all__ = [
    "StoredRecord",
    "VectorStore",
    "InMemoryVectorStore",
    "create_vector_store",
    "reset_vector_store",
]
