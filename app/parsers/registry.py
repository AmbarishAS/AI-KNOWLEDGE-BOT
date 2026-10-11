"""Parser registry: maps file extensions to the correct parser."""

from __future__ import annotations

import os

from app.domain import DocumentMetadata, ParsedDocument
from app.parsers.base import Parser, ParserError
from app.parsers.docx_parser import DocxParser
from app.parsers.pdf_parser import PdfParser
from app.parsers.text_parser import MarkdownParser, TextParser

_PARSERS: list[Parser] = [TextParser(), MarkdownParser(), PdfParser(), DocxParser()]
_BY_EXT: dict[str, Parser] = {ext: p for p in _PARSERS for ext in p.extensions}


def get_parser(filename: str) -> Parser:
    ext = os.path.splitext(filename)[1].lower()
    parser = _BY_EXT.get(ext)
    if parser is None:
        raise ParserError(f"No parser registered for '{ext}'")
    return parser


def parse_document(filename: str, data: bytes, metadata: DocumentMetadata) -> ParsedDocument:
    """Parse ``data`` using the parser matching ``filename``'s extension."""
    return get_parser(filename).parse(data, metadata)


def supported_extensions() -> set[str]:
    return set(_BY_EXT.keys())
