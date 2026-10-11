"""Repositories: the only place that talks to the ORM.

Services depend on these typed methods, not on SQLAlchemy directly. Every query is
tenant-scoped to enforce isolation (see ``docs/SECURITY.md``).
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import Chunk, DocumentRecord, Message, Role
from app.persistence.models import ChunkORM, ConversationORM, DocumentORM, MessageORM


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add_document(self, record: DocumentRecord, chunks: list[Chunk]) -> None:
        doc = DocumentORM(
            id=record.document_id,
            tenant=record.tenant,
            filename=record.filename,
            content_type=record.content_type,
            title=record.title,
            num_chunks=record.num_chunks,
            chunk_strategy=record.chunk_strategy,
        )
        self._s.add(doc)
        for ch in chunks:
            self._s.add(
                ChunkORM(
                    id=ch.chunk_id,
                    document_id=ch.document_id,
                    tenant=ch.tenant,
                    ordinal=ch.ordinal,
                    text=ch.text,
                    page=ch.page,
                    section=ch.section,
                    parent_id=ch.parent_id,
                    filename=ch.filename,
                    strategy=ch.strategy,
                    extra=ch.extra,
                )
            )

    async def list_documents(self, tenant: str, limit: int = 50, offset: int = 0) -> tuple[list[DocumentRecord], int]:
        total = await self._s.scalar(select(func.count()).select_from(DocumentORM).where(DocumentORM.tenant == tenant))
        rows = (
            await self._s.scalars(
                select(DocumentORM)
                .where(DocumentORM.tenant == tenant)
                .order_by(DocumentORM.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        ).all()
        records = [
            DocumentRecord(
                document_id=r.id,
                filename=r.filename,
                tenant=r.tenant,
                content_type=r.content_type,
                num_chunks=r.num_chunks,
                chunk_strategy=r.chunk_strategy,
                title=r.title,
                created_at=r.created_at,
            )
            for r in rows
        ]
        return records, int(total or 0)

    async def get_document(self, tenant: str, document_id: str) -> DocumentRecord | None:
        r = await self._s.get(DocumentORM, document_id)
        if r is None or r.tenant != tenant:
            return None
        return DocumentRecord(
            document_id=r.id,
            filename=r.filename,
            tenant=r.tenant,
            content_type=r.content_type,
            num_chunks=r.num_chunks,
            chunk_strategy=r.chunk_strategy,
            title=r.title,
            created_at=r.created_at,
        )

    async def delete_document(self, tenant: str, document_id: str) -> bool:
        r = await self._s.get(DocumentORM, document_id)
        if r is None or r.tenant != tenant:
            return False
        await self._s.delete(r)
        return True

    async def get_chunks(self, tenant: str, document_id: str) -> list[Chunk]:
        rows = (
            await self._s.scalars(
                select(ChunkORM)
                .where(ChunkORM.tenant == tenant, ChunkORM.document_id == document_id)
                .order_by(ChunkORM.ordinal)
            )
        ).all()
        return [_chunk_from_orm(r) for r in rows]


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(self, tenant: str, subject: str, title: str | None = None) -> str:
        cid = uuid.uuid4().hex
        self._s.add(ConversationORM(id=cid, tenant=tenant, subject=subject, title=title))
        return cid

    async def ensure(self, tenant: str, subject: str, conversation_id: str | None) -> str:
        if conversation_id:
            conv = await self._s.get(ConversationORM, conversation_id)
            if conv is not None and conv.tenant == tenant:
                return conversation_id
        return await self.create(tenant, subject)

    async def add_message(
        self, tenant: str, conversation_id: str, message: Message, citations: list | None = None
    ) -> None:
        ordinal = await self._s.scalar(
            select(func.count()).select_from(MessageORM).where(MessageORM.conversation_id == conversation_id)
        )
        self._s.add(
            MessageORM(
                conversation_id=conversation_id,
                tenant=tenant,
                ordinal=int(ordinal or 0),
                role=message.role.value,
                content=message.content,
                citations=citations or [],
            )
        )

    async def get_messages(self, tenant: str, conversation_id: str, limit: int | None = None) -> list[Message]:
        conv = await self._s.get(ConversationORM, conversation_id)
        if conv is None or conv.tenant != tenant:
            return []
        q = select(MessageORM).where(MessageORM.conversation_id == conversation_id).order_by(MessageORM.ordinal)
        rows = (await self._s.scalars(q)).all()
        msgs = [Message(role=Role(r.role), content=r.content) for r in rows]
        if limit is not None:
            return msgs[-limit:]
        return msgs

    async def delete(self, tenant: str, conversation_id: str) -> bool:
        conv = await self._s.get(ConversationORM, conversation_id)
        if conv is None or conv.tenant != tenant:
            return False
        await self._s.execute(delete(MessageORM).where(MessageORM.conversation_id == conversation_id))
        await self._s.delete(conv)
        return True


def _chunk_from_orm(r: ChunkORM) -> Chunk:
    return Chunk(
        chunk_id=r.id,
        document_id=r.document_id,
        tenant=r.tenant,
        text=r.text,
        ordinal=r.ordinal,
        page=r.page,
        section=r.section,
        parent_id=r.parent_id,
        filename=r.filename,
        strategy=r.strategy,
        extra=r.extra or {},
    )
