# Learning Guide (beginner-friendly)

This guide explains every major concept in plain language and points to where it lives in the
code. It's written for a strong engineer new to GenAI (e.g. a Java full-stack dev).

## 0. Mental model

A RAG app answers questions **using your documents** instead of the model's memory. The flow:
cut documents into chunks → turn chunks into vectors → store them → for a question, find the
most relevant chunks → ask the LLM to answer **using only those chunks** and cite them.

## 1. Python & typing you'll see

- **Type hints** everywhere (`list[str]`, `str | None`) — like Java generics/`Optional`.
- **Pydantic v2** models (`app/domain/__init__.py`) validate data at the edges, like strongly
  typed DTOs that validate themselves.
- **Pydantic Settings** (`app/config.py`) load typed config from env vars.
- **async/await** (`async def`) — non-blocking I/O, similar to Java's `CompletableFuture`,
  used for DB and HTTP calls.

## 2. Configuration & secrets — `app/config.py`

One typed `Settings` object is the source of truth. `APP_MODE=mock` means no network calls.
Secrets come from env vars, never hardcoded.

## 3. Documents → text — `app/parsers/`

Each parser turns bytes into a `ParsedDocument` with normalized text and **blocks** (headings,
pages). Blocks let us cite the exact page/section later.

## 4. Chunking — `app/chunking/strategies.py`

Five strategies (fixed, recursive, structure, semantic, parent-child). A **chunk** is a small
retrievable unit that keeps its source references (filename, page, section). See
`docs/RAG_DEEP_DIVE.md` for when to use each.

## 5. Embeddings — `app/embeddings/`

An **embedding** is a list of numbers (a vector) representing meaning. Similar text → nearby
vectors. We measure closeness with **cosine similarity**. The mock provider is deterministic;
the real one calls an OpenAI-compatible API. Key rule: query and documents must use the
**same** embedding model.

## 6. Vector store — `app/vectorstore/`

Stores vectors + metadata and does similarity search. In-memory by default (NumPy), Qdrant in
Docker. Tenant/metadata filters are applied before scoring so you only ever see your data.

## 7. Retrieval — `app/retrieval/`

- **Dense**: search by vector similarity (good for meaning/paraphrase).
- **Sparse/BM25**: search by exact words (good for names, codes). `bm25.py` is a tiny,
  dependency-free implementation.
- **Hybrid**: combine both with **Reciprocal Rank Fusion** (RRF).
- **Reranker**: re-score the shortlist for precision (`reranker.py`).

## 8. Generation & citations — `app/rag/`

We build a prompt (`prompts.py`) containing the question and numbered passages, tell the model
to answer **only** from them and cite like `[1]`, then **validate** the citations
(`citations.py`) so every `[n]` maps to a real passage. If there's no good evidence, we
**abstain** instead of guessing (`pipeline.py`).

## 9. Tokens, context window, cost — `app/llm/tokens.py`

LLMs read/write **tokens** (~4 chars each). The **context window** is the max tokens a model
can consider; we budget context to fit. We estimate **cost** from token counts and a price
table.

## 10. Agents & tools — `app/agents/`

An **agent** decides whether to use a **tool** (calculator, KB search, read-only SQL,
ticketing). Tools validate their inputs; dangerous actions (writes) need **approval**. Bounds
prevent infinite loops. See `docs/AGENTS_AND_LANGGRAPH.md`.

## 11. MCP — `app/mcp/`

The **Model Context Protocol** is a standard way for a client to **discover** and **call**
tools on a server. We implement `tools/list` and `tools/call` with schema validation. See
`docs/MCP_GUIDE.md`.

## 12. Memory — `app/services/chat_service.py`

Conversations are stored in the DB. We only replay the **last few** messages to the model to
control cost and avoid "memory contamination". Memory (chat) is separate from knowledge
(documents).

## 13. Security — `app/core/security.py`

API-key auth, tenant isolation, upload validation, safe filenames, SQL allowlisting, secret
redaction. See `docs/SECURITY.md`.

## 14. Evaluation — `app/evaluation/`

Measure retrieval (Precision@k, Recall@k, MRR, nDCG) and generation (groundedness, abstention)
on a fixed dataset, and compare a baseline vs an improved pipeline. See `docs/EVALUATION.md`.

## 15. API & UI — `app/api/`, `ui/`

FastAPI exposes versioned REST endpoints with typed request/response schemas and automatic
`/docs`. Streamlit provides a simple demo UI.

## 16. How to run things

See the README "Common commands" table. Start with `pytest` to see everything work, then
`uvicorn app.main:app --reload` and the Streamlit UI.

## Suggested reading order of the code

1. `app/config.py`, `app/domain/__init__.py`
2. `app/parsers/` → `app/chunking/strategies.py`
3. `app/embeddings/` → `app/vectorstore/` → `app/retrieval/`
4. `app/rag/` (prompts, citations, pipeline)
5. `app/services/` → `app/api/v1/`
6. `app/agents/` → `app/mcp/`
7. `app/evaluation/`
