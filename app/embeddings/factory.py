"""Embedding provider factory (mock vs real), driven by Settings."""

from __future__ import annotations

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.mock import MockEmbeddings


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.is_mock:
        # Smaller dim keeps the mock fast; real mode uses the provider's native dim.
        return MockEmbeddings(dim=256)
    settings.require_real_credentials()
    from app.embeddings.openai_compatible import OpenAICompatibleEmbeddings

    return OpenAICompatibleEmbeddings(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        timeout=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
    )
