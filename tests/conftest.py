"""Shared pytest fixtures.

Tests run in deterministic mock mode against a temporary SQLite database and in-memory
vector store, so no external services or credentials are required.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator, Iterator

import pytest

# Force mock mode + isolated temp DB BEFORE importing app modules that read settings.
os.environ.setdefault("APP_MODE", "mock")
os.environ.setdefault("API_AUTH_ENABLED", "true")
os.environ.setdefault("DEMO_API_KEY", "test-key")
_tmp_db = os.path.join(tempfile.gettempdir(), "copilot_test.db")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_tmp_db}"


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def _fresh_db() -> AsyncIterator[None]:
    """Recreate the schema for each test for isolation."""
    from app.persistence.db import Base, get_engine, reset_engine

    # Remove the file so each test starts clean.
    if os.path.exists(_tmp_db):
        try:
            os.remove(_tmp_db)
        except PermissionError:
            pass
    engine = get_engine()
    # Import models to register metadata.
    import app.persistence.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await reset_engine()


@pytest.fixture
def settings():
    from app.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    return get_settings()
