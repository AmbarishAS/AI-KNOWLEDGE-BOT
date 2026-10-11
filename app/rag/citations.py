"""Citation building and validation.

A citation links a marker like ``[1]`` in the answer back to the concrete source chunk that
grounds it. We **validate** that every marker the model emitted actually corresponds to a
passage we provided — dropping hallucinated markers — and surface the real source metadata
(filename/page/section) for the UI.
"""

from __future__ import annotations

import re

from app.domain import Citation, ScoredChunk

_MARKER = re.compile(r"\[(\d+)\]")


def extract_markers(answer: str) -> list[int]:
    return sorted({int(m) for m in _MARKER.findall(answer)})


def build_citations(answer: str, passages: list[ScoredChunk]) -> list[Citation]:
    """Return citations only for markers that map to a real provided passage."""
    citations: list[Citation] = []
    for num in extract_markers(answer):
        if 1 <= num <= len(passages):
            ch = passages[num - 1].chunk
            snippet = ch.text[:200].strip()
            citations.append(
                Citation(
                    marker=f"[{num}]",
                    chunk_id=ch.chunk_id,
                    document_id=ch.document_id,
                    filename=ch.filename,
                    page=ch.page,
                    section=ch.section,
                    snippet=snippet,
                )
            )
    return citations


def strip_invalid_markers(answer: str, num_passages: int) -> str:
    """Remove citation markers that point outside the provided passage range."""

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= num_passages else ""

    return _MARKER.sub(repl, answer).replace("  ", " ").strip()


def is_grounded(answer: str, passages: list[ScoredChunk]) -> bool:
    """A heuristic groundedness check: answer cites at least one valid passage."""
    markers = extract_markers(answer)
    return any(1 <= n <= len(passages) for n in markers)
