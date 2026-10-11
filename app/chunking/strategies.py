"""Chunking strategies.

Each strategy is selectable via configuration so they can be *compared* in the evaluation
suite (see ``docs/EVALUATION.md``) rather than hard-wired. Trade-offs are documented in
``docs/RAG_DEEP_DIVE.md``.

Strategies implemented:

- ``fixed``        : fixed character window with overlap. Simple, ignores structure.
- ``recursive``    : split on a hierarchy of separators (paragraph → sentence → word),
                     packing up to ``chunk_size`` with overlap. Good general default.
- ``structure``    : one chunk per document block/section, merging small blocks. Best for
                     well-structured docs (markdown/docx) and precise section citations.
- ``semantic``     : group adjacent sentences while they stay lexically similar; split at
                     topic shifts. ``similarity_fn`` is pluggable (lexical by default; an
                     embedding-based function can be injected in real mode).
- ``parent_child`` : large parent chunks for context + small child chunks for precise
                     retrieval. Children carry ``parent_id`` for parent-child retrieval.
"""

from __future__ import annotations

from collections.abc import Callable

from app.chunking.base import Chunker, _finalize, split_sentences
from app.config import ChunkStrategy, Settings
from app.domain import Chunk, ParsedDocument

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _new_chunk(doc: ParsedDocument, text: str, strategy: str, **kw) -> Chunk:
    md = doc.metadata
    return Chunk(
        document_id=md.document_id,
        tenant=md.tenant,
        filename=md.filename,
        text=text.strip(),
        strategy=strategy,
        **kw,
    )


# --------------------------------------------------------------------------- fixed


class FixedSizeChunker:
    name = "fixed"

    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        self.size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        text = doc.text
        step = max(1, self.size - self.overlap)
        chunks: list[Chunk] = []
        for start in range(0, len(text), step):
            piece = text[start : start + self.size].strip()
            if piece:
                chunks.append(_new_chunk(doc, piece, self.name))
            if start + self.size >= len(text):
                break
        return _finalize(chunks)


# --------------------------------------------------------------------------- recursive


class RecursiveChunker:
    name = "recursive"

    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        self.size = chunk_size
        self.overlap = overlap

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.size or not separators:
            return [text]
        sep, *rest = separators
        parts = text.split(sep) if sep else list(text)
        out: list[str] = []
        buf = ""
        for part in parts:
            candidate = f"{buf}{sep}{part}" if buf else part
            if len(candidate) <= self.size:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                if len(part) > self.size:
                    out.extend(self._split(part, rest))
                    buf = ""
                else:
                    buf = part
        if buf:
            out.append(buf)
        return out

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        pieces = self._split(doc.text, _RECURSIVE_SEPARATORS)
        # Apply overlap by prefixing the tail of the previous piece.
        chunks: list[Chunk] = []
        prev_tail = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            text = (prev_tail + " " + piece).strip() if prev_tail else piece
            chunks.append(_new_chunk(doc, text, self.name))
            prev_tail = piece[-self.overlap :] if self.overlap else ""
        return _finalize(chunks)


# --------------------------------------------------------------------------- structure


class StructureAwareChunker:
    name = "structure"

    def __init__(self, chunk_size: int = 800, overlap: int = 0) -> None:
        self.size = chunk_size

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        if not doc.blocks:
            return RecursiveChunker(self.size).chunk(doc)
        chunks: list[Chunk] = []
        buf: list[str] = []
        section: str | None = None
        page: int | None = None

        def flush() -> None:
            if buf:
                chunks.append(_new_chunk(doc, " ".join(buf), self.name, section=section, page=page))
                buf.clear()

        for block in doc.blocks:
            if block.kind == "heading":
                flush()
                section = block.section or block.text
                continue
            if block.page is not None:
                page = block.page
            # If adding this block would exceed the size, flush first.
            if buf and sum(len(b) for b in buf) + len(block.text) > self.size:
                flush()
            buf.append(block.text)
        flush()
        if not chunks:  # all headings / empty
            return RecursiveChunker(self.size).chunk(doc)
        return _finalize(chunks)


# --------------------------------------------------------------------------- semantic


def _lexical_similarity(a: str, b: str) -> float:
    """Jaccard similarity over lowercased word sets — a cheap 'semantic' proxy."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class SemanticChunker:
    name = "semantic"

    def __init__(
        self,
        chunk_size: int = 800,
        threshold: float = 0.15,
        similarity_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.size = chunk_size
        self.threshold = threshold
        self.similarity_fn = similarity_fn or _lexical_similarity

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        if not sentences:
            return []
        chunks: list[Chunk] = []
        buf = [sentences[0]]
        for prev, curr in zip(sentences, sentences[1:], strict=False):
            sim = self.similarity_fn(prev, curr)
            current_len = sum(len(s) for s in buf)
            # Split on a topic shift (low similarity) or when the budget is exceeded.
            if sim < self.threshold or current_len + len(curr) > self.size:
                chunks.append(_new_chunk(doc, " ".join(buf), self.name))
                buf = [curr]
            else:
                buf.append(curr)
        if buf:
            chunks.append(_new_chunk(doc, " ".join(buf), self.name))
        return _finalize(chunks)


# --------------------------------------------------------------------------- parent-child


class ParentChildChunker:
    name = "parent_child"

    def __init__(self, chunk_size: int = 800, child_size: int = 300, overlap: int = 60) -> None:
        self.parent_size = chunk_size
        self.child_size = child_size
        self.overlap = overlap

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        parents = RecursiveChunker(self.parent_size, self.overlap).chunk(doc)
        all_chunks: list[Chunk] = []
        for parent in parents:
            parent.strategy = self.name
            parent.extra = {**parent.extra, "role": "parent"}
            all_chunks.append(parent)
            child_splitter = RecursiveChunker(self.child_size, self.overlap)
            # Build a throwaway doc so the child splitter reuses metadata.
            children = child_splitter._split(parent.text, _RECURSIVE_SEPARATORS)
            for child_text in children:
                child_text = child_text.strip()
                if not child_text:
                    continue
                child = _new_chunk(
                    doc,
                    child_text,
                    self.name,
                    parent_id=parent.chunk_id,
                    section=parent.section,
                    page=parent.page,
                )
                child.extra = {"role": "child"}
                all_chunks.append(child)
        return _finalize(all_chunks)


# --------------------------------------------------------------------------- factory

_STRATEGIES = {
    ChunkStrategy.FIXED: FixedSizeChunker,
    ChunkStrategy.RECURSIVE: RecursiveChunker,
    ChunkStrategy.STRUCTURE: StructureAwareChunker,
    ChunkStrategy.SEMANTIC: SemanticChunker,
    ChunkStrategy.PARENT_CHILD: ParentChildChunker,
}


def create_chunker(strategy: ChunkStrategy | str, settings: Settings) -> Chunker:
    """Build a chunker for the given strategy using settings for sizing."""
    if isinstance(strategy, str):
        strategy = ChunkStrategy(strategy)
    cls = _STRATEGIES[strategy]
    if strategy == ChunkStrategy.SEMANTIC:
        return cls(chunk_size=settings.chunk_size)  # type: ignore[call-arg]
    if strategy == ChunkStrategy.STRUCTURE:
        return cls(chunk_size=settings.chunk_size)  # type: ignore[call-arg]
    return cls(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)  # type: ignore[call-arg]
