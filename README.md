# Enterprise AI Knowledge Copilot 🤖

A production-style **RAG + Agents + MCP** knowledge assistant built to help a Java
full-stack developer prepare for **GenAI engineering interviews**. It runs end-to-end in a
deterministic **mock mode** (no paid API key required) and switches to real OpenAI-compatible
providers via a single config flag.

> Every major GenAI concept — chunking, embeddings, hybrid retrieval, reranking, grounded
> generation with citations, abstention, agents, tools, MCP, memory, evaluation, security,
> and observability — is implemented in real, tested code and mapped to interview questions
> in [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md).

## Features

- 📄 **Ingestion**: PDF / DOCX / Markdown / TXT parsing, metadata extraction, and **five
  configurable chunking strategies** (fixed, recursive, structure-aware, semantic,
  parent-child).
- 🔎 **Retrieval**: dense, sparse (**built-in BM25**), and **hybrid** search with
  **Reciprocal Rank Fusion**, metadata/tenant filtering, configurable top-k, reranking,
  multi-query, and parent-child expansion.
- 💬 **Grounded RAG**: prompt-versioned generation with **citations linked to real chunks**,
  score-threshold **abstention**, context dedup + token budgeting, and prompt-injection
  defenses.
- 🧰 **Agents**: a bounded LangGraph-style state machine with intent routing, **safe tools**
  (calculator, KB search, read-only allowlisted SQL, ticketing), **approval gates** for
  writes, and step/retry limits.
- 🔌 **MCP**: a working Model Context Protocol server + client (tool discovery, invocation,
  schema validation, error handling) with an optional official-SDK stdio transport.
- 🗂️ **Memory**: conversation history in PostgreSQL/SQLite with bounded context and tenant
  isolation.
- 📊 **Evaluation**: reproducible retrieval metrics (Precision@k, Recall@k, MRR, Hit-Rate@k,
  nDCG@k) + generation proxies and a **baseline-vs-improved ablation**.
- 🔐 **Security**: API-key auth (OAuth2/OIDC extension point), tenant authorization, upload
  validation, safe filenames, SQL allowlisting, secret redaction in logs.
- 📈 **Observability**: structured JSON logs with request ids, latency/token/cost metrics.
- 🖥️ **UI**: a Streamlit app for upload, chat with visible citations, and evaluation.
- 🐳 **Ops**: Docker Compose (API + UI + Postgres + Qdrant), GitHub Actions CI, health/ready
  probes.

## Architecture (high level)

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full diagrams.

```
API (FastAPI) ─▶ Services ─▶ Domain/RAG ─▶ Integrations (parsers, embeddings, LLM,
                                            vector store, retrieval, agents, MCP)
                                   │
                           Persistence (Postgres/SQLite, Qdrant/in-memory)
       Cross-cutting: config · structured logging · metrics · security
```

## Quick start (local, no Docker, mock mode)

```powershell
# 1) Install deps with Poetry (creates/manages the virtualenv for you)
poetry install --with dev

# 2) Run the API (mock mode is the default — no API key needed)
poetry run uvicorn app.main:app --reload
#   API docs:   http://localhost:8000/docs
#   Health:     http://localhost:8000/health

# 3) In another terminal: load the sample documents
poetry run python -m scripts.load_sample_docs --api-key changeme-dev-key

# 4) Ask a question
curl -X POST http://localhost:8000/api/v1/chat `
  -H "X-API-Key: changeme-dev-key" -H "Content-Type: application/json" `
  -d '{"question":"How many vacation days do employees accrue?"}'
```

### Streamlit UI

```powershell
$env:COPILOT_API_URL="http://localhost:8000"; poetry run streamlit run ui/streamlit_app.py
# open http://localhost:8501
```

## Quick start (Docker Compose — API + UI + Postgres + Qdrant)

```powershell
copy .env.example .env         # edit if you want real mode / custom keys
docker compose up --build      # builds and starts all services
# API:  http://localhost:8000/docs    UI: http://localhost:8501
docker compose down            # stop
docker compose down -v         # stop + reset volumes (safe demo reset)
```

## Enabling real LLM/embeddings

Mock mode never calls the network. To use a real OpenAI-compatible provider:

```dotenv
APP_MODE=real
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

The app fails fast if `APP_MODE=real` without an API key. Responses are always tagged with
`provider` and `mode` so mock output is never mistaken for real model output.

## Common commands

| Task | Command |
|------|---------|
| Run tests | `poetry run pytest` |
| Coverage | `poetry run pytest --cov=app --cov-report=term-missing` |
| Lint | `poetry run ruff check .` |
| Format | `poetry run ruff format .` |
| Type check | `poetry run mypy app` |
| Load sample docs | `poetry run python -m scripts.load_sample_docs` |
| Run evaluation | `poetry run python -m scripts.run_eval` |
| Run MCP server (stdio) | `poetry install --extras mcp; poetry run python -m scripts.run_mcp_server` |
| API docs | open `http://localhost:8000/docs` |

## REST API (v1)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/documents` | Upload & ingest a document |
| GET | `/api/v1/documents` | List documents (paginated) |
| GET | `/api/v1/documents/{id}` | Get a document |
| DELETE | `/api/v1/documents/{id}` | Delete a document |
| POST | `/api/v1/chat` | Grounded chat answer with citations |
| POST | `/api/v1/chat/stream` | Streaming answer (SSE) |
| GET | `/api/v1/conversations/{id}` | Conversation history |
| DELETE | `/api/v1/conversations/{id}` | Delete a conversation |
| GET | `/api/v1/agent/tools` | List agent tools + schemas |
| POST | `/api/v1/agent/run` | Run the agent (agentic or deterministic) |
| POST | `/api/v1/evaluations/run` | Run the evaluation suite |
| GET | `/health`, `/ready`, `/metrics` | Ops endpoints |

All authenticated endpoints require `X-API-Key` (and optional `X-Tenant`).

## Verified results (this environment)

- ✅ **64 tests pass** (`pytest`) in mock mode; Ruff lint clean; Ruff format applied.
- ✅ **Evaluation ablation** (`python -m scripts.run_eval`), real output:

  | Metric | Baseline (dense) | Improved (hybrid+rerank) |
  |--------|------------------|--------------------------|
  | precision@5 | 0.333 | 0.333 |
  | recall@5 | 1.000 | 1.000 |
  | hit_rate@5 | 1.000 | 1.000 |
  | MRR | 0.917 | **1.000** |
  | nDCG@5 | 0.939 | **1.000** |

  Abstention accuracy **0.875**, avg groundedness **0.76** on the curated 8-case set.
- ⚠️ **Not verified in this environment**: Docker image build / `docker compose up`
  (engine present but not executed here) and real-provider calls (no credentials). Both are
  wired and documented; see [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md).

## Documentation

- [Project plan](docs/PROJECT_PLAN.md) · [Architecture](docs/ARCHITECTURE.md) ·
  [Tech stack](docs/TECH_STACK.md) · [Implementation status](docs/IMPLEMENTATION_STATUS.md)
- [Learning guide](docs/LEARNING_GUIDE.md) · [RAG deep dive](docs/RAG_DEEP_DIVE.md) ·
  [Agents & LangGraph](docs/AGENTS_AND_LANGGRAPH.md) · [MCP guide](docs/MCP_GUIDE.md)
- [Evaluation](docs/EVALUATION.md) · [Security](docs/SECURITY.md) ·
  [Deployment](docs/DEPLOYMENT.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Interview guide](docs/INTERVIEW_GUIDE.md) · [Project walkthrough](docs/PROJECT_WALKTHROUGH.md)

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md). Most common: forgetting the
`X-API-Key` header (→ 401), or running the API without first loading documents (→ the
assistant correctly abstains).

## License

Sample/synthetic data only. No real customer data or production credentials are included.
