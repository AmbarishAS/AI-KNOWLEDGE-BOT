# Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `401 Invalid or missing API key` | Missing `X-API-Key` header | Send `X-API-Key: changeme-dev-key` (or your `DEMO_API_KEY`) |
| Chat always abstains | No documents ingested, or query unrelated | Load docs: `poetry run python -m scripts.load_sample_docs`; ask about their content |
| `422 Unsupported file type` | Extension not allowlisted | Use pdf/docx/md/txt or extend `ALLOWED_UPLOAD_EXTENSIONS` |
| `no such table: documents` (scripts) | DB schema not created | The runner calls `init_db()`; for custom scripts call `await init_db()` first |
| `the greenlet library is required` | Missing greenlet for SQLAlchemy async | `poetry run pip install greenlet` (already pinned in `pyproject.toml`) |
| `APP_MODE=real requires OPENAI_API_KEY` | Real mode without key | Set `OPENAI_API_KEY` or use `APP_MODE=mock` |
| Qdrant errors in Docker | Qdrant not ready yet | Compose waits on healthchecks; retry; check `docker compose logs qdrant` |
| Streamlit "API unreachable" | API not running / wrong URL | Start API; set `COPILOT_API_URL=http://localhost:8000` |
| Empty PDF parse error | Scanned/image-only PDF | OCR is out of scope; use a text-based PDF |
| Tests fail on Windows file lock | Temp SQLite file locked | Re-run; conftest tolerates `PermissionError` on cleanup |
| `mcp` import error | Optional SDK not installed | `poetry install --extras mcp` only if you need the stdio transport |

## Diagnostics

- Health: `GET /health` → liveness; `GET /ready` → DB + vector store status.
- Metrics: `GET /metrics` → request counts + latency percentiles.
- Logs are structured JSON with a `request_id` you can grep across the pipeline.

## Resetting state

- Local: delete `./data/copilot.db` (SQLite) to wipe metadata; the in-memory vector store
  resets on process restart.
- Docker: `docker compose down -v` drops Postgres and Qdrant volumes.
