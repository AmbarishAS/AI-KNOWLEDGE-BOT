"""Deterministic mock embeddings — no network, reproducible, and *meaningfully* ordered.

We build a vector from hashed word features so that documents sharing words get similar
vectors. This is NOT a real semantic model, but it is deterministic and good enough to make
retrieval tests assertable. Output is tagged via ``name='mock'`` so it is never mistaken for
a real provider.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from app.embeddings.base import l2_normalize

_WORD = re.compile(r"[a-z0-9]+")


class MockEmbeddings:
    name = "mock"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = _WORD.findall(text.lower())
        if not words:
            return vec.tolist()
        for w in words:
            # Hash each word to a stable index + sign; a "bag of hashed words" embedding.
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        return l2_normalize(vec).tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
