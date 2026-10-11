"""SQLAlchemy ORM models for documents, chunks, conversations and messages.

We use portable JSON columns (not Postgres JSONB) so the same schema runs on SQLite and
Postgres (ADR 0002). Tenant is stored on every row to enforce isolation at query time.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.persistence.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


class DocumentORM(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant: Mapped[str] = mapped_column(String(128), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(128), default="text/plain")
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    chunk_strategy: Mapped[str] = mapped_column(String(64), default="recursive")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    chunks: Mapped[list[ChunkORM]] = relationship(back_populates="document", cascade="all, delete-orphan")


class ChunkORM(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    tenant: Mapped[str] = mapped_column(String(128), index=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    filename: Mapped[str] = mapped_column(String(255), default="")
    strategy: Mapped[str] = mapped_column(String(64), default="recursive")
    extra: Mapped[dict] = mapped_column(JSON, default=dict)

    document: Mapped[DocumentORM] = relationship(back_populates="chunks")


class ConversationORM(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant: Mapped[str] = mapped_column(String(128), index=True)
    subject: Mapped[str] = mapped_column(String(128), default="demo-user")
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    messages: Mapped[list[MessageORM]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="MessageORM.ordinal"
    )


class MessageORM(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    tenant: Mapped[str] = mapped_column(String(128), index=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped[ConversationORM] = relationship(back_populates="messages")
