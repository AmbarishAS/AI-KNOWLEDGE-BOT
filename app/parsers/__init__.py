"""Document parsers (PDF, DOCX, Markdown, TXT)."""

from app.parsers.base import ParserError
from app.parsers.registry import get_parser, parse_document, supported_extensions

__all__ = ["ParserError", "get_parser", "parse_document", "supported_extensions"]
