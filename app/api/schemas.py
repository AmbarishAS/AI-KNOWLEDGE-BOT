"""Pydantic request/response schemas for the REST API (the API contract)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain import Citation, DocumentRecord, Role, ScoredChunk


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    mode: str


class ReadyResponse(BaseModel):
    status: str
    database: bool
    vector_store: bool
    chunks_indexed: int


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    tenant: str
    content_type: str
    num_chunks: int
    chunk_strategy: str
    title: str | None = None
    created_at: datetime

    @classmethod
    def from_record(cls, r: DocumentRecord) -> DocumentResponse:
        return cls(**r.model_dump())


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    use_memory: bool = True
    rerank: bool | None = None
    multi_query: bool = False
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationResponse(BaseModel):
    marker: str
    document_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    snippet: str

    @classmethod
    def from_citation(cls, c: Citation) -> CitationResponse:
        return cls(
            marker=c.marker,
            document_id=c.document_id,
            filename=c.filename,
            page=c.page,
            section=c.section,
            snippet=c.snippet,
        )


class RetrievedPassage(BaseModel):
    chunk_id: str
    filename: str
    score: float
    text: str

    @classmethod
    def from_scored(cls, s: ScoredChunk) -> RetrievedPassage:
        return cls(
            chunk_id=s.chunk.chunk_id,
            filename=s.chunk.filename,
            score=round(s.score, 4),
            text=s.chunk.text[:400],
        )


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    abstained: bool
    citations: list[CitationResponse]
    retrieved: list[RetrievedPassage]
    provider: str
    mode: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    latency_ms: float


class MessageResponse(BaseModel):
    role: Role
    content: str


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: list[MessageResponse]


class EvaluationRequest(BaseModel):
    dataset: str | None = None
    rerank: bool = True
    retrieval_mode: str | None = None


class ErrorResponse(BaseModel):
    detail: str
