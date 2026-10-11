"""Health and readiness endpoints.

- ``/health`` is a cheap liveness probe (is the process up?).
- ``/ready`` checks dependencies (DB reachable, vector store reachable) so orchestrators
  only route traffic when the app can actually serve requests.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app import __version__
from app.api.deps import container_dep, settings_dep
from app.api.schemas import HealthResponse, ReadyResponse
from app.config import Settings
from app.container import Container
from app.persistence.db import get_engine

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(settings_dep)) -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, mode=settings.app_mode.value)


@router.get("/ready", response_model=ReadyResponse)
async def ready(container: Container = Depends(container_dep)) -> ReadyResponse:
    db_ok = True
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    vs_ok = True
    chunks = 0
    try:
        chunks = await container.vector_store.count()
    except Exception:
        vs_ok = False

    status = "ok" if (db_ok and vs_ok) else "degraded"
    return ReadyResponse(status=status, database=db_ok, vector_store=vs_ok, chunks_indexed=chunks)
