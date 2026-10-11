"""Phase 2 — parsers and chunking strategies."""

from __future__ import annotations

import pytest

from app.chunking.strategies import create_chunker
from app.config import ChunkStrategy, get_settings
from app.domain import DocumentMetadata
from app.parsers import ParserError, parse_document

MD = b"""# Title One

Intro paragraph about retrieval augmented generation and embeddings.

## Section Two

Second section discusses chunking strategies and metadata preservation in detail.
"""


def _meta(filename: str = "doc.md") -> DocumentMetadata:
    return DocumentMetadata(document_id="d1", filename=filename, tenant="acme")


def test_markdown_parser_extracts_headings_and_title():
    doc = parse_document("doc.md", MD, _meta())
    assert doc.metadata.title == "Title One"
    headings = [b.text for b in doc.blocks if b.kind == "heading"]
    assert "Title One" in headings and "Section Two" in headings
    # Paragraph inherits its section.
    paras = [b for b in doc.blocks if b.kind == "paragraph"]
    assert any(b.section == "Section Two" for b in paras)


def test_text_parser_rejects_empty():
    with pytest.raises(ParserError):
        parse_document("x.txt", b"   \n  ", _meta("x.txt"))


def test_pdf_parser_rejects_corrupt():
    with pytest.raises(ParserError):
        parse_document("x.pdf", b"not a real pdf", _meta("x.pdf"))


def test_unsupported_extension_raises():
    with pytest.raises(ParserError):
        parse_document("x.exe", b"data", _meta("x.exe"))


@pytest.mark.parametrize(
    "strategy",
    [
        ChunkStrategy.FIXED,
        ChunkStrategy.RECURSIVE,
        ChunkStrategy.STRUCTURE,
        ChunkStrategy.SEMANTIC,
        ChunkStrategy.PARENT_CHILD,
    ],
)
def test_all_chunk_strategies_preserve_metadata(strategy):
    s = get_settings()
    s.chunk_size = 200
    doc = parse_document("doc.md", MD, _meta())
    chunker = create_chunker(strategy, s)
    chunks = chunker.chunk(doc)
    assert chunks, f"{strategy} produced no chunks"
    for i, c in enumerate(chunks):
        assert c.ordinal == i  # sequential ordinals
        assert c.document_id == "d1"
        assert c.tenant == "acme"
        assert c.text.strip()


def test_parent_child_links_children_to_parents():
    s = get_settings()
    s.chunk_size = 200
    doc = parse_document("doc.md", MD, _meta())
    chunks = create_chunker(ChunkStrategy.PARENT_CHILD, s).chunk(doc)
    parents = {c.chunk_id for c in chunks if c.extra.get("role") == "parent"}
    children = [c for c in chunks if c.extra.get("role") == "child"]
    assert parents and children
    assert all(c.parent_id in parents for c in children)


def test_structure_chunking_sets_sections():
    s = get_settings()
    s.chunk_size = 500
    doc = parse_document("doc.md", MD, _meta())
    chunks = create_chunker(ChunkStrategy.STRUCTURE, s).chunk(doc)
    assert any(c.section for c in chunks)
