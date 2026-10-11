# Tech Stack — rationale & trade-offs

Python version: **3.13.0** (verified in the build environment). The spec allows 3.12 or
another explicitly verified version; all dependencies below support 3.13.

| Dependency | Why it is used | Alternatives considered | Trade-off |
|-----------|----------------|-------------------------|-----------|
| **FastAPI** | Async REST API, automatic OpenAPI docs, Pydantic integration | Flask, Django REST | Less batteries-included than Django, but ideal for typed async APIs |
| **Pydantic v2 + pydantic-settings** | Request/response validation, typed settings from env | dataclasses + manual validation | Slight learning curve; huge safety/clarity win |
| **SQLAlchemy 2.0 (async)** | ORM + repositories; SQLite for local/tests, Postgres for prod | raw asyncpg, SQLModel | More verbose than SQLModel, but stable and explicit |
| **Uvicorn** | ASGI server | Hypercorn | Standard, well supported |
| **NumPy** | Vector math, cosine similarity, deterministic mock embeddings | pure Python | Fast, ubiquitous |
| **pypdf** | PDF text extraction (pure Python, no system deps) | pdfminer, PyMuPDF | PyMuPDF faster but AGPL/binary; pypdf is simpler for a demo |
| **python-docx** | DOCX parsing | docx2txt | Richer structure (headings) for structure-aware chunking |
| **rank-bm25** (optional) | Sparse BM25 retrieval | custom TF-IDF | We ship a small built-in BM25 so tests need no extra dep |
| **Qdrant client** (optional) | Production vector DB | pgvector, Weaviate, FAISS | Qdrant has great filtering + hybrid; optional so mock mode needs nothing |
| **httpx** | Async HTTP to OpenAI-compatible providers | requests, aiohttp | Async + sync, clean API |
| **LangChain / LangGraph** (optional) | Agent state graph, text splitters reference | custom | Heavy; we provide a dependency-free fallback graph so core works without it |
| **MCP SDK (`mcp`)** (optional) | Model Context Protocol server/client | custom JSON-RPC | Official SDK; guarded import with graceful fallback |
| **Streamlit** | Demo UI | Gradio, React | Fastest path to an upload/chat/citation UI |
| **pytest + pytest-asyncio** | Unit/integration/API tests | unittest | Fixtures + async support |
| **Ruff** | Lint + format (replaces black/flake8/isort) | black+flake8 | Single fast tool |
| **mypy** | Static type checking | pyright | Standard CI type checker |
| **Docker + Compose** | Local multi-service orchestration | Podman | Ubiquitous |
| **GitHub Actions** | CI | GitLab CI | Matches GitHub hosting |

## Dependency strategy

- **Hard runtime deps** are minimal so the tested core runs anywhere: FastAPI, Pydantic,
  SQLAlchemy, NumPy, pypdf, python-docx, httpx.
- **Optional deps** (Qdrant, LangGraph, MCP, Streamlit, RAGAS, OpenTelemetry) are imported
  lazily behind `try/except` with capability flags, so missing optional packages degrade
  gracefully instead of crashing the app or tests.
- Versions are pinned in `pyproject.toml` and locked in `poetry.lock`. Runtime deps live under
  `[tool.poetry.dependencies]`, tooling under the `dev` group, and real/full-mode packages are
  exposed as Poetry extras (`prod`, `agents`, `mcp`, `ui`, `eval`, `observability`, `full`).

## Why mock-first?

Interview demos and CI must be reproducible and free. Deterministic mock providers make
retrieval/generation outputs stable and assertable, while real providers are a one-line
config switch. We **never** present mock output as if it came from a real model — responses
are tagged with the provider name and `mode`.
