"""Plain-text and Markdown parsers.

Markdown parsing is intentionally lightweight (no HTML rendering): we extract ATX headings
(``#``) to build section-aware blocks, which improves structure-aware chunking and lets us
cite the section a passage came from.
"""

from __future__ import annotations

import re

from app.domain import DocumentBlock, DocumentMetadata, ParsedDocument
from app.parsers.base import ParserError, blocks_to_text, normalize_text

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ParserError("Unable to decode text file with common encodings")


class TextParser:
    extensions = (".txt",)

    def parse(self, data: bytes, metadata: DocumentMetadata) -> ParsedDocument:
        raw = _decode(data)
        text = normalize_text(raw)
        if not text:
            raise ParserError("Empty text document")
        blocks = [DocumentBlock(text=p.strip(), kind="paragraph") for p in text.split("\n\n") if p.strip()]
        return ParsedDocument(metadata=metadata, text=text, blocks=blocks)


class MarkdownParser:
    extensions = (".md", ".markdown")

    def parse(self, data: bytes, metadata: DocumentMetadata) -> ParsedDocument:
        raw = _decode(data)
        if not raw.strip():
            raise ParserError("Empty markdown document")
        blocks: list[DocumentBlock] = []
        current_section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                para = normalize_text("\n".join(buffer))
                if para:
                    blocks.append(DocumentBlock(text=para, kind="paragraph", section=current_section))
                buffer.clear()

        for line in raw.splitlines():
            m = _HEADING.match(line)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                current_section = title
                blocks.append(DocumentBlock(text=title, kind="heading", section=title, level=level))
            elif line.strip() == "":
                flush()
            else:
                buffer.append(line)
        flush()

        if not blocks:
            raise ParserError("Markdown document contained no content")
        # Title defaults to first heading.
        if metadata.title is None:
            for b in blocks:
                if b.kind == "heading":
                    metadata.title = b.text
                    break
        return ParsedDocument(metadata=metadata, text=blocks_to_text(blocks), blocks=blocks)
