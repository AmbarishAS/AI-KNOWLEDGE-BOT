# Architecture — Enterprise AI Knowledge Copilot

## 1. High-level overview

```mermaid
flowchart TB
    subgraph Presentation
        UI[Streamlit UI]
        API[FastAPI REST API v1]
    end
    subgraph Application
        ING[Ingestion Service]
        CHAT[Chat Service]
        AGENT[Agent Orchestrator]
        EVAL[Evaluation Runner]
    end
    subgraph Domain
        MODELS[Domain Models & Rules]
        CHUNK[Chunking Strategies]
        RAGP[RAG Pipeline]
    end
    subgraph Integrations
        PARSE[Document Parsers]
        EMB[Embedding Provider]
        LLM[LLM Provider]
        VEC[Vector Store]
        RERANK[Reranker]
        TOOLS[Business Tools]
        MCPS[MCP Server/Client]
    end
    subgraph Persistence
        PG[(PostgreSQL / SQLite)]
        QD[(Qdrant / In-memory)]
    end
    subgraph CrossCutting
        CFG[Settings]
        LOG[Structured Logging]
        OBS[Metrics/Observability]
        SEC[Security/Auth]
    end

    UI --> API
    API --> ING & CHAT & AGENT & EVAL
    ING --> PARSE --> CHUNK --> EMB --> VEC
    CHAT --> RAGP --> VEC & RERANK & LLM
    AGENT --> TOOLS & MCPS & RAGP
    ING --> PG
    CHAT --> PG
    VEC --> QD
    EMB --> VEC
```

## 2. Layering

1. **API & presentation** (`app/api`, `ui/`): HTTP endpoints, request/response schemas,
   Gradio demo. No business logic here.
2. **Application services** (`app/services`): orchestrate use-cases (ingest a document,
   answer a chat turn). Transaction boundaries and DI live here.
3. **Domain** (`app/domain`, `app/chunking`, `app/rag`): framework-free models and rules.
4. **Integrations** (`app/parsers`, `app/embeddings`, `app/llm`, `app/vectorstore`,
   `app/retrieval`, `app/agents`, `app/mcp`): adapters behind typed interfaces.
5. **Persistence** (`app/persistence`): SQLAlchemy models + repositories.
6. **Cross-cutting** (`app/config.py`, `app/logging_config.py`, `app/observability`,
   `app/core/security.py`): configuration, logging, metrics, auth.

Dependencies point **inward**: integrations depend on domain interfaces, not vice versa.
Providers are selected via **factories** driven by `Settings`, enabling mock ↔ real swaps.

## 3. Document ingestion lifecycle

```mermaid
sequenceDiagram
    participant U as Client
    participant API as POST /documents
    participant ING as IngestionService
    participant P as Parser
    participant C as Chunker
    participant E as Embeddings
    participant V as VectorStore
    participant DB as Metadata DB

    U->>API: upload file (multipart)
    API->>API: validate type/size/filename
    API->>ING: ingest(file, tenant)
    ING->>P: parse() -> ParsedDocument
    ING->>ING: normalize + extract metadata
    ING->>C: chunk(strategy) -> [Chunk]
    ING->>E: embed(chunk texts) [batched]
    ING->>V: upsert(vectors + metadata)
    ING->>DB: save document + chunk records
    ING-->>API: DocumentRecord
    API-->>U: 201 Created
```

## 4. RAG query lifecycle

```mermaid
sequenceDiagram
    participant U as Client
    participant API as POST /chat
    participant CS as ChatService
    participant QT as Query Transform
    participant R as Retriever (dense/sparse/hybrid)
    participant RR as Reranker
    participant CB as Context Builder
    participant LLM as LLM Provider
    participant CV as Citation Validator

    U->>API: question + conversation_id
    API->>CS: answer(request, principal)
    CS->>QT: rewrite / multi-query (optional)
    CS->>R: retrieve(top_k, filters, scope)
    R-->>CS: candidate passages + scores
    CS->>RR: rerank(query, passages)
    CS->>CB: dedupe + token-budget + compress
    alt insufficient evidence
        CS-->>API: abstention response
    else
        CS->>LLM: grounded prompt + context
        LLM-->>CS: answer
        CS->>CV: validate citations map to chunks
        CS-->>API: answer + citations + usage
    end
    API-->>U: 200 OK
```

## 5. Agent state graph

See `docs/AGENTS_AND_LANGGRAPH.md` for the full node/edge description. The agent is a
bounded state machine: classify → retrieve → decide tool → (tool | answer) → validate,
with max-step and retry limits and an approval gate for sensitive (write) tools.

## 6. Data flow & trust boundaries

- **Uploaded documents are untrusted.** They are parsed and treated as data; any
  instructions embedded in them are *not* executed. Prompt-injection defenses wrap
  retrieved context (delimiters + system guardrails) — see `docs/SECURITY.md`.
- **Tool inputs/outputs are validated** with Pydantic schemas. SQL is allowlisted and
  parameterized; the LLM never executes arbitrary SQL/shell/code.
- **Tenant scope** is attached to every stored chunk and enforced at retrieval time.

## 7. Configuration-driven provider selection

| Concern | Mock (default) | Real (opt-in via env) |
|--------|----------------|----------------------|
| LLM | `MockLLM` (deterministic) | OpenAI-compatible `/chat/completions` |
| Embeddings | `MockEmbeddings` (hashed, deterministic) | OpenAI-compatible `/embeddings` |
| Vector store | In-memory | Qdrant |
| Metadata DB | SQLite | PostgreSQL |

`APP_MODE=mock` (default) guarantees no external calls. `APP_MODE=real` requires valid
credentials and documents which env vars are needed.

## 8. Deployment topology (Docker Compose)

Four services run as independent containers so the UI and API can scale separately. The
API talks to Postgres (metadata) and Qdrant (vectors); the Gradio UI talks only to the API
over HTTP. Dependencies are gated by healthchecks.

```mermaid
flowchart LR
    subgraph Host / Docker Compose network
        UI["ui\nGradio\n:7860\n(Dockerfile.gradio)"]
        API["api\nFastAPI\n:8000\n(Dockerfile)"]
        PG[("postgres\n:5432\nvolume: pgdata")]
        QD[("qdrant\n:6333\nvolume: qdrantdata")]
    end
    Browser(("Browser")) -->|HTTP :7860| UI
    UI -->|REST X-API-Key / X-Tenant| API
    API -->|asyncpg| PG
    API -->|qdrant-client| QD
    API -. depends_on: service_healthy .-> PG
    API -. depends_on: service_healthy .-> QD
    UI -. depends_on: service_started .-> API
```

| Service | Build | Port | Depends on |
|---------|-------|------|-----------|
| `ui` | `Dockerfile.gradio` | 7860 | `api` (started) |
| `api` | `Dockerfile` | 8000 | `postgres`, `qdrant` (healthy) |
| `postgres` | `postgres:16-alpine` | 5432 | — |
| `qdrant` | `qdrant/qdrant:v1.12.1` | 6333 | — |


