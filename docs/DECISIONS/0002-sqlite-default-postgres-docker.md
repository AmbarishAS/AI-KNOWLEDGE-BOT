# ADR 0002 — SQLite default, PostgreSQL in Docker

- **Status:** Accepted
- **Date:** Phase 0
- **Context:** The spec requires PostgreSQL for metadata, but local development and CI
  benefit from a zero-infrastructure database.
- **Decision:** Use SQLAlchemy 2.0 async with an abstract `DATABASE_URL`. Default to
  `sqlite+aiosqlite` for local runs and tests; Docker Compose sets
  `postgresql+asyncpg://...`. Schema is created via metadata `create_all` for the demo,
  with Alembic noted as the production migration path.
- **Consequences:**
  - (+) Tests run with no DB server; parity with Postgres via the same ORM.
  - (−) A few Postgres-specific features (e.g. JSONB operators) are avoided to keep SQLite
    parity; we use portable JSON columns.

