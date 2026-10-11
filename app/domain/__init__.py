"""Domain models and business rules (framework-free Pydantic v2 models).

These types are the shared vocabulary across layers. They intentionally do not import
FastAPI, SQLAlchemy, or any provider SDK — keeping the domain independent of infrastructure.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Principal(BaseModel):
    """The authenticated caller and their access scope."""

    subject: str = "demo-user"
    tenant: str = "public"
    scopes: list[str] = Field(default_factory=lambda: ["read", "write"])

    def can(self, scope: str) -> bool:
        return scope in self.scopes


class DocumentMetadata(BaseModel):
    """Metadata *stored alongside* chunks (not blindly embedded into the text).

    See ``docs/RAG_DEEP_DIVE.md`` for why we separate stored metadata from embedded text:
    stored metadata powers filtering & citations; only a small, high-signal subset (e.g.
    section heading) is prepended to the embedded text to improve retrieval.
    """

    document_id: str
    filename: str
    tenant: str = "public"
    content_type: str = "text/plain"
    title: str | None = None
    author: str | None = None
    created_at: datetime = Field(default_factory=_now)
    extra: dict[str, str] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Output of a parser: normalized text plus optional structural blocks."""

    metadata: DocumentMetadata
    text: str
    # Structural blocks preserve headings/pages for structure-aware chunking & citations.
    blocks: list[DocumentBlock] = Field(default_factory=list)


class DocumentBlock(BaseModel):
    """A structural unit of a document (heading, paragraph, page)."""

    text: str
    kind: str = "paragraph"  # heading | paragraph | page | list_item | table
    page: int | None = None
    section: str | None = None
    level: int | None = None  # heading depth


class Chunk(BaseModel):
    """A retrievable unit with preserved source references for citation."""

    chunk_id: str = Field(default_factory=_uuid)
    document_id: str
    tenant: str = "public"
    text: str
    # Text actually embedded (may include a small metadata prefix). Kept separate so the
    # displayed/cited text stays clean.
    embed_text: str | None = None
    ordinal: int = 0
    page: int | None = None
    section: str | None = None
    parent_id: str | None = None  # for parent-child / hierarchical chunking
    filename: str = ""
    strategy: str = "recursive"
    extra: dict[str, str] = Field(default_factory=dict)

    @property
    def text_for_embedding(self) -> str:
        return self.embed_text or self.text


class ScoredChunk(BaseModel):
    """A chunk returned by retrieval with a relevance score and its provenance."""

    chunk: Chunk
    score: float
    source: str = "dense"  # dense | sparse | hybrid | rerank
    rank: int = 0


class Citation(BaseModel):
    """A citation that links an answer back to a concrete source chunk."""

    marker: str  # e.g. "[1]"
    chunk_id: str
    document_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    snippet: str = ""


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Answer(BaseModel):
    """Grounded answer returned by the RAG pipeline."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    provider: str = "mock"
    mode: str = "mock"
    usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    retrieved: list[ScoredChunk] = Field(default_factory=list)


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None  # tool name for tool messages


class DocumentRecord(BaseModel):
    """Persisted record describing an ingested document."""

    document_id: str
    filename: str
    tenant: str
    content_type: str
    num_chunks: int
    chunk_strategy: str
    title: str | None = None
    created_at: datetime = Field(default_factory=_now)


ParsedDocument.model_rebuild()
