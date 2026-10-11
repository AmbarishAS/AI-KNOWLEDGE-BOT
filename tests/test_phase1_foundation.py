"""Phase 1 — foundation: config, logging, security, persistence."""

from __future__ import annotations

import pytest

from app.config import AppMode, get_settings
from app.core.security import (
    AuthError,
    UploadValidationError,
    authenticate,
    safe_filename,
    validate_upload,
)
from app.domain import Chunk, DocumentRecord, Message, Role
from app.logging_config import redact
from app.persistence.db import session_scope
from app.persistence.repositories import ConversationRepository, DocumentRepository


def test_settings_defaults_to_mock_mode():
    get_settings.cache_clear()
    s = get_settings()
    assert s.app_mode == AppMode.MOCK
    assert s.is_mock is True
    assert ".pdf" in s.allowed_extensions


def test_redaction_masks_api_keys():
    out = redact("my key is sk-ABCDEF123456 and token")
    assert "sk-ABCDEF123456" not in out
    assert "REDACTED" in out


def test_safe_filename_blocks_path_traversal():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("..\\..\\win.ini") == "win.ini"
    assert safe_filename("résumé final.pdf").endswith(".pdf")


def test_authenticate_requires_valid_key():
    s = get_settings()
    with pytest.raises(AuthError):
        authenticate("wrong", None, s)
    p = authenticate(s.demo_api_key, "acme", s)
    assert p.tenant == "acme"


def test_validate_upload_rejects_bad_type_and_size():
    s = get_settings()
    with pytest.raises(UploadValidationError):
        validate_upload("x.exe", 10, s)
    with pytest.raises(UploadValidationError):
        validate_upload("x.pdf", 0, s)
    with pytest.raises(UploadValidationError):
        validate_upload("x.pdf", s.max_upload_bytes + 1, s)
    assert validate_upload("good.pdf", 100, s) == "good.pdf"


async def test_document_repository_roundtrip_and_tenant_isolation():
    rec = DocumentRecord(
        document_id="doc1",
        filename="a.txt",
        tenant="acme",
        content_type="text/plain",
        num_chunks=1,
        chunk_strategy="recursive",
    )
    chunk = Chunk(document_id="doc1", tenant="acme", text="hello", filename="a.txt")
    async with session_scope() as s:
        await DocumentRepository(s).add_document(rec, [chunk])

    async with session_scope() as s:
        repo = DocumentRepository(s)
        docs, total = await repo.list_documents("acme")
        assert total == 1 and docs[0].document_id == "doc1"
        # Different tenant cannot see it.
        other, other_total = await repo.list_documents("evil")
        assert other_total == 0 and other == []


async def test_conversation_history_persists():
    async with session_scope() as s:
        repo = ConversationRepository(s)
        cid = await repo.create("acme", "demo-user")
        await repo.add_message("acme", cid, Message(role=Role.USER, content="hi"))
        await repo.add_message("acme", cid, Message(role=Role.ASSISTANT, content="hello"))

    async with session_scope() as s:
        msgs = await ConversationRepository(s).get_messages("acme", cid)
        assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT]
        # Wrong tenant sees nothing.
        assert await ConversationRepository(s).get_messages("evil", cid) == []
