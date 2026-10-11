"""Parser interface and shared helpers.

A parser turns raw bytes into a :class:`ParsedDocument` with normalized text and optional
structural *blocks* (headings/pages). Blocks are what enable structure-aware chunking and
accurate page/section citations.
"""

from __future__ import annotations

import re
from typing import Protocol

from app.domain import DocumentBlock, DocumentMetadata, ParsedDocument


class ParserError(Exception):
    """Raised when a document cannot be parsed."""


class Parser(Protocol):
    """Protocol all parsers implement."""

    extensions: tuple[str, ...]

    def parse(self, data: bytes, metadata: DocumentMetadata) -> ParsedDocument: ...


_WS = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


def blocks_to_text(blocks: list[DocumentBlock]) -> str:
    return normalize_text("\n\n".join(b.text for b in blocks if b.text.strip()))
