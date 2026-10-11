"""Embedding provider interface + helpers.

An embedding maps text to a dense vector so semantically similar text lands nearby in
vector space. We program against this interface so business logic never imports a provider
SDK directly (see ADR 0001).

Key interview points surfaced here:
- **Dimensionality** must match between indexing and querying — mixing models/dims makes
  similarity meaningless, which is why vectors must be regenerated when the embedding model
  changes incompatibly (see ``docs/RAG_DEEP_DIVE.md``).
- We **L2-normalize** vectors so cosine similarity == dot product.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0:
        return 0.0
    return float(np.dot(av, bv) / denom)
