"""FastAPI application factory.

Wires middleware (request-id correlation + latency metric), exception handlers, routers,
and startup/shutdown (DB init). Keeping this in a factory makes it easy to build test apps.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.logging_config import configure_logging, get_logger, request_id_var
from app.observability.metrics import METRICS
from app.persistence.db import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    await init_db()
    logger.info(
        "application started",
        extra={"extra": {"mode": settings.app_mode.value, "version": __version__}},
    )
    yield
    logger.info("application stopping")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise AI Knowledge Copilot",
        version=__version__,
        description="RAG + Agents + MCP, mock-first and Dockerized.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            METRICS.observe(f"http.{request.method}", elapsed)
            METRICS.incr("http.requests")
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error")
        METRICS.incr("http.errors")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # Routers (imported here to avoid circulars and keep factory importable in tests).
    from app.api.v1 import agents as agents_router
    from app.api.v1 import chat, documents, health
    from app.api.v1 import evaluations as evaluations_router

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(agents_router.router)
    app.include_router(evaluations_router.router)

    @app.get("/metrics", tags=["observability"])
    async def metrics() -> dict:
        return METRICS.snapshot()

    return app


app = create_app()
