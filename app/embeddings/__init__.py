"""Embeddings: provider interface, mock + OpenAI-compatible implementations."""

from app.embeddings.base import EmbeddingProvider, cosine_similarity, l2_normalize
from app.embeddings.factory import create_embedding_provider

__all__ = [
    "EmbeddingProvider",
    "cosine_similarity",
    "l2_normalize",
    "create_embedding_provider",
]
