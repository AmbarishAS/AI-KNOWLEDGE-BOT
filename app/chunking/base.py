"""Chunking interface and shared text utilities.

Chunking splits a parsed document into retrievable units. The *right* strategy depends on
the document: see ``docs/RAG_DEEP_DIVE.md`` for a comparison. All strategies preserve
source references (page/section/ordinal/parent) so citations stay accurate.
"""

from __future__ import annotations

import re
from typing import Protocol

from app.domain import Chunk, ParsedDocument

# Rough token estimate: ~4 chars/token for English. Good enough for budgeting without
# importing a tokenizer in the hot path (real token counts come from app.llm.tokens).
CHARS_PER_TOKEN = 4

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no NLTK dependency)."""
    parts = [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class Chunker(Protocol):
    name: str

    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...


def _finalize(chunks: list[Chunk]) -> list[Chunk]:
    """Assign sequential ordinals."""
    for i, c in enumerate(chunks):
        c.ordinal = i
    return chunks
