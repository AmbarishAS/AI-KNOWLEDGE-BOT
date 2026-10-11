"""PDF parser using pypdf (pure Python, no system dependencies).

Each page becomes a block carrying its page number so citations can reference the exact
page. Scanned/image-only PDFs yield little/no text — we raise a clear error rather than
silently returning empty content.
"""

from __future__ import annotations

import io

from app.domain import DocumentBlock, DocumentMetadata, ParsedDocument
from app.parsers.base import ParserError, blocks_to_text, normalize_text


class PdfParser:
    extensions = (".pdf",)

    def parse(self, data: bytes, metadata: DocumentMetadata) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ParserError("pypdf is not installed") from exc

        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as exc:
            raise ParserError(f"Invalid or corrupted PDF: {exc}") from exc

        blocks: list[DocumentBlock] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = normalize_text(text)
            if text:
                blocks.append(DocumentBlock(text=text, kind="page", page=i))

        if not blocks:
            raise ParserError("No extractable text found (PDF may be scanned/image-only; OCR is required)")

        # Pull a title from PDF metadata if present.
        if metadata.title is None and reader.metadata and reader.metadata.title:
            metadata.title = str(reader.metadata.title)

        return ParsedDocument(metadata=metadata, text=blocks_to_text(blocks), blocks=blocks)
