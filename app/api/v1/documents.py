"""Document endpoints: upload, list (paginated), get, delete."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

from app.api.deps import container_dep, principal_dep, settings_dep
from app.api.schemas import DocumentListResponse, DocumentResponse
from app.config import Settings
from app.container import Container
from app.core.security import UploadValidationError, validate_upload
from app.domain import Principal
from app.parsers import ParserError
from app.persistence.db import session_scope
from app.persistence.repositories import DocumentRepository

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    strategy: str | None = Query(default=None, description="Override chunking strategy"),
    principal: Principal = Depends(principal_dep),
    container: Container = Depends(container_dep),
    settings: Settings = Depends(settings_dep),
) -> DocumentResponse:
    if not principal.can("write"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Write scope required")
    data = await file.read()
    try:
        safe_name = validate_upload(file.filename or "upload", len(data), settings)
    except UploadValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    try:
        record = await container.ingestion.ingest(
            safe_name, data, file.content_type or "application/octet-stream", principal, strategy
        )
    except ParserError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Parse error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return DocumentResponse.from_record(record)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(principal_dep),
) -> DocumentListResponse:
    async with session_scope() as session:
        records, total = await DocumentRepository(session).list_documents(principal.tenant, limit, offset)
    return DocumentListResponse(
        items=[DocumentResponse.from_record(r) for r in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, principal: Principal = Depends(principal_dep)) -> DocumentResponse:
    async with session_scope() as session:
        record = await DocumentRepository(session).get_document(principal.tenant, document_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return DocumentResponse.from_record(record)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_document(
    document_id: str,
    principal: Principal = Depends(principal_dep),
    container: Container = Depends(container_dep),
):
    if not principal.can("write"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Write scope required")
    deleted = await container.ingestion.delete(principal, document_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
