# Project Plan — Enterprise AI Knowledge Copilot

## Goal

Build a production-style, locally-runnable **Enterprise AI Knowledge Copilot** (RAG +
agents + MCP) that a Java full-stack developer can study to prepare for GenAI
engineering interviews. The system must run end-to-end in **deterministic mock mode**
without any paid API key, and switch to real providers purely through configuration.

## Guiding principles

- **Mock-first**: every external dependency (LLM, embeddings, vector DB) has a
  deterministic local implementation so tests and demos never require network access.
- **Honest status**: `docs/IMPLEMENTATION_STATUS.md` always reflects reality — what is
  implemented, tested, and verified vs. pending.
- **Layered architecture**: API → services → domain → integrations → persistence →
  cross-cutting (config/security/observability).
- **Config over hardcoding**: strategies (chunking, retrieval, reranking) are selectable
  so they can be compared and evaluated, not wired as unconditional steps.

## Phases & acceptance criteria

| Phase | Scope | Acceptance criteria |
|------|-------|---------------------|
| 0 | Inspect workspace, plan, architecture docs, repo structure | Planning docs exist; structure defined; Python/Docker verified |
| 1 | Package structure, config, logging, domain models, DB, lint/type/test infra | `pytest` runs; settings load; logging emits structured JSON |
| 2 | Ingestion: parsers (PDF/DOCX/MD/TXT), metadata, chunking strategies | Parse + chunk tests pass; metadata preserved |
| 3 | Embeddings, vector store (memory/Qdrant), dense/sparse/hybrid retrieval, rerank, citations | Retrieval tests pass; hybrid RRF verified; citations map to chunks |
| 4 | Grounded RAG generation, query transforms, abstention, context mgmt, chat API | RAG returns answer + citations; abstains when no evidence |
| 5 | Deterministic workflow + LangGraph agent, safe tools, approval gates, bounded steps | Agent runs; SQL allowlist enforced; approval required for writes |
| 6 | MCP server + client, agent consumes MCP tools | Client discovers + invokes tools; schema validation works |
| 7 | Conversation persistence, memory, auth, tenant isolation, security tests | History persists; tenant isolation enforced; auth required |
| 8 | Evaluation datasets, retrieval + generation metrics, ablation, reports | `evaluate` command outputs JSON + markdown report |
| 9 | Streamlit UI, streaming, observability, benchmarks | UI uploads/chats/shows citations; metrics captured |
| 10 | Docker Compose, CI, health/readiness, reproducible setup | Compose config valid; CI workflow present |
| 11 | Documentation, acceptance verification, interview guide | All docs match code; interview guide maps to source files |

## Milestones

1. **M1 — Walking skeleton** (Phases 0–1): app boots, config + logging + tests.
2. **M2 — RAG core** (Phases 2–4): upload → retrieve → grounded answer with citations.
3. **M3 — Agents + MCP** (Phases 5–6): bounded agent + MCP tool discovery.
4. **M4 — Enterprise concerns** (Phases 7–8): auth, memory, evaluation.
5. **M5 — Delivery** (Phases 9–11): UI, Docker, CI, docs.

## Dependencies between phases

- Phase 3 depends on Phase 2 chunk output.
- Phase 4 depends on Phase 3 retriever.
- Phase 5 tools reuse Phase 3 retriever (knowledge-base search tool).
- Phase 6 MCP reuses Phase 5 tool implementations.
- Phase 8 evaluation depends on Phase 3/4 pipeline.

## Out of scope (by design)

- Training/fine-tuning a foundation model.
- Redis/Kafka/Kubernetes (not justified for a local demo).
- Real multi-tenant SSO — we provide a clear OAuth2/OIDC extension point instead.
</content>
</invoke>

