# Project Walkthrough

Three scripts for three audiences/time budgets.

## 🎬 5-minute demo (show it works)

1. Start the API: `uvicorn app.main:app --reload` → open `http://localhost:8000/docs`.
2. Hit `GET /health` → shows `mode: mock` (no keys needed).
3. Load sample docs: `python -m scripts.load_sample_docs`.
4. Start the UI: `streamlit run ui/streamlit_app.py` → `http://localhost:8501`.
5. Ask *"How many vacation days do employees accrue?"* → grounded answer **with a citation**
   to `vacation_policy.md`, plus token/cost/latency metrics.
6. Ask *"What is the capital of France?"* → the assistant **abstains** (no evidence) — this is
   the anti-hallucination behavior.
7. In the **Evaluation** tab, click *Run evaluation* → shows the baseline-vs-improved table.

Talking point: "Everything you just saw runs with zero API keys in deterministic mock mode;
flipping `APP_MODE=real` swaps in a real LLM without touching business logic."

## 🧭 15-minute technical walkthrough

1. **Config & modes** (`app/config.py`): one typed settings object; mock-first design (ADR
   0001).
2. **Ingestion** (`app/services/ingestion.py`): parse → chunk → embed → index → persist, with
   the metadata stored-vs-embedded distinction.
3. **Chunking** (`app/chunking/strategies.py`): show the five strategies and when to use each.
4. **Retrieval** (`app/retrieval/`): dense + BM25 + hybrid RRF; metadata/tenant filtering.
5. **RAG pipeline** (`app/rag/pipeline.py`): threshold → rerank → dedup/token-budget →
   generate → **citation validation** → abstention.
6. **Agent** (`app/agents/graph.py`): intent routing, safe tools, **approval gate**, bounds.
7. **MCP** (`app/mcp/`): `tools/list` / `tools/call`, schema validation, in-process vs stdio.
8. **Evaluation** (`app/evaluation/`): metrics + ablation; run `scripts.run_eval` live.
9. **API/tests**: `app/api/v1/*` and 64 passing tests across phases.

## 🏗️ Deeper architecture discussion

- **Layering & DI** (`docs/ARCHITECTURE.md`): API → services → domain → integrations →
  persistence → cross-cutting. Providers chosen by factories from `Settings`.
- **Trust boundaries**: uploaded docs and tool outputs are untrusted; validation, allowlists,
  approvals, and injection defenses enforce safety (`docs/SECURITY.md`).
- **Data model**: tenant on every row/chunk → isolation enforced at query time.
- **Reliability**: timeouts + bounded retries around providers; health/ready probes.
- **Scaling**: swap in-memory store → Qdrant (ANN + payload filters); batch embeddings;
  async I/O; move BM25 to a dedicated search engine at large scale.
- **Honest limitations**: mock providers aren't semantic; real metrics need an evaluator
  model; Docker build not executed in the authoring environment (config complete).

## What to study before an interview (recommended order)

1. `docs/LEARNING_GUIDE.md` (concepts) → 2. `docs/RAG_DEEP_DIVE.md` →
3. `app/rag/pipeline.py` + `app/retrieval/retriever.py` →
4. `docs/AGENTS_AND_LANGGRAPH.md` + `app/agents/graph.py` →
5. `docs/MCP_GUIDE.md` + `app/mcp/server.py` →
6. `docs/EVALUATION.md` + `app/evaluation/runner.py` →
7. `docs/INTERVIEW_GUIDE.md` (rehearse answers against the code).
