"""Ingestion service: the full parse → chunk → embed → index → persist pipeline.

This is the application-layer orchestration for ``POST /documents``. It is deliberately the
*only* place that knows the ordering of the ingestion steps, keeping the API thin and the
integrations (parser/chunker/embedder/store) reusable.

Metadata strategy (interview-relevant): we store rich metadata on each chunk (document id,
filename, page, section, tenant) for **filtering and citations**, but only prepend a small,
high-signal prefix (the section heading) to the **embedded** text. Embedding every metadata
field would dilute the semantic signal. See ``docs/RAG_DEEP_DIVE.md``.
"""

from __future__ import annotations

import time
import uuid

from app.chunking.strategies import create_chunker
from app.config import Settings
from app.domain import Chunk, DocumentMetadata, DocumentRecord, Principal
from app.embeddings.base import EmbeddingProvider
from app.logging_config import get_logger
from app.parsers import parse_document
from app.persistence.db import session_scope
from app.persistence.repositories import DocumentRepository
from app.vectorstore.base import StoredRecord, VectorStore

logger = get_logger(__name__)


def _embed_text(chunk: Chunk) -> str:
    """High-signal embedding text: prepend section heading only (not all metadata)."""
    if chunk.section:
        return f"{chunk.section}\n{chunk.text}"
    return chunk.text


class IngestionService:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore, settings: Settings) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._settings = settings

    async def ingest(
        self,
        filename: str,
        data: bytes,
        content_type: str,
        principal: Principal,
        strategy: str | None = None,
    ) -> DocumentRecord:
        t0 = time.perf_counter()
        document_id = uuid.uuid4().hex
        strategy = strategy or self._settings.chunk_strategy.value

        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            tenant=principal.tenant,
            content_type=content_type,
        )

        parsed = parse_document(filename, data, metadata)
        t_parse = time.perf_counter()

        chunker = create_chunker(strategy, self._settings)
        chunks = chunker.chunk(parsed)
        if not chunks:
            raise ValueError("Document produced no chunks")
        for ch in chunks:
            ch.embed_text = _embed_text(ch)
        t_chunk = time.perf_counter()

        vectors = await self._embedder.embed([c.text_for_embedding for c in chunks])
        t_embed = time.perf_counter()

        await self._store.upsert([StoredRecord(chunk=c, vector=v) for c, v in zip(chunks, vectors, strict=True)])

        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            tenant=principal.tenant,
            content_type=content_type,
            num_chunks=len(chunks),
            chunk_strategy=strategy,
            title=parsed.metadata.title,
        )
        async with session_scope() as session:
            await DocumentRepository(session).add_document(record, chunks)
        t_done = time.perf_counter()

        logger.info(
            "document ingested",
            extra={
                "extra": {
                    "document_id": document_id,
                    "tenant": principal.tenant,
                    "num_chunks": len(chunks),
                    "strategy": strategy,
                    "parse_ms": round((t_parse - t0) * 1000, 1),
                    "chunk_ms": round((t_chunk - t_parse) * 1000, 1),
                    "embed_ms": round((t_embed - t_chunk) * 1000, 1),
                    "persist_ms": round((t_done - t_embed) * 1000, 1),
                }
            },
        )
        return record

    async def delete(self, principal: Principal, document_id: str) -> bool:
        async with session_scope() as session:
            deleted = await DocumentRepository(session).delete_document(principal.tenant, document_id)
        await self._store.delete_document(principal.tenant, document_id)
        return deleted
