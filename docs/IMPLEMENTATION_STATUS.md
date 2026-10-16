# Implementation Status

> Single source of truth for what is actually done, tested, and verified.
> Legend: ✅ done & tested · 🟡 partial · ⬜ pending · 🔬 verified by running

_Last updated: Phases 0–11 implemented; 64 tests passing; lint clean; eval verified._

## Phase tracker

| Phase | Status | Notes |
|------|--------|-------|
| 0 — Plan & architecture docs | ✅ | Plan, architecture, tech stack, status, 4 ADRs |
| 1 — Package, config, logging, domain, DB, test infra | ✅ 🔬 | SQLite async; structured JSON logs; 7 tests |
| 2 — Ingestion (parsers, metadata, chunking) | ✅ 🔬 | 4 parsers, 5 chunk strategies; 11 tests |
| 3 — Embeddings, vector store, retrieval, rerank | ✅ 🔬 | mock+real embeddings, memory+qdrant, dense/sparse/hybrid RRF; 9 tests |
| 4 — RAG generation, abstention, chat API | ✅ 🔬 | grounded answers, citation validation, injection defense; 7 tests |
| 5 — Workflows, agent, tools, approval gates | ✅ 🔬 | bounded agent, safe tools, SQL allowlist; 12 tests |
| 6 — MCP server + client | ✅ 🔬 | tools/list, tools/call, validation, approval; 7 tests |
| 7 — Conversation persistence, memory, auth, tenancy | ✅ 🔬 | bounded history, tenant isolation; Phase 1 + API tests |
| 8 — Evaluation datasets, metrics, ablation | ✅ 🔬 | P/R/MRR/nDCG + generation proxies; CLI + report; 3 tests |
| 9 — Streamlit UI, streaming, observability | ✅ 🟡 | UI + SSE streaming + metrics endpoint; UI not automated-tested |
| 10 — Docker Compose, CI, health checks | ✅ 🟡 | Dockerfiles, compose, CI present; **image build not executed here** |
| 11 — Documentation & interview guide | ✅ | README + docs + 4 ADRs, mapped to source |

## Verification log

| What was run | Result |
|--------------|--------|
| `python --version`, `docker --version` | Python 3.13.0, Docker 28.4.0 |
| `pytest` | **64 passed** |
| `ruff check .` | **All checks passed** |
| `python -m scripts.run_eval` | MRR 0.917→1.0, nDCG 0.939→1.0; abstention 0.875 |

## Known limitations / unverified

- **Docker build / `compose up`**: not executed in the authoring environment (engine present).
  Dockerfiles + compose are complete; verify with `docker compose up --build`.
- **Real-provider calls** (`APP_MODE=real`): implemented with retries/timeouts but not
  exercised (no credentials); opt-in.
- **Optional packages** (`qdrant-client`, `langgraph`, `mcp`, `gradio`, `ragas`): guarded
  imports; absence degrades gracefully. Gradio UI is manual-tested, not in CI.
- **Generation metrics** are deterministic proxies; faithful scoring needs an evaluator model
  (RAGAS) — see `docs/EVALUATION.md`.
- **mypy** is advisory in CI (not blocking) for this demo.
