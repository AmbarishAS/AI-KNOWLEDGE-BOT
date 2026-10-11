# RAG Deep Dive

This document explains the Retrieval-Augmented Generation pipeline as implemented, with the
exact source files for each stage.

## The pipeline

```
upload → validate → parse → normalize → extract metadata → chunk → embed → index →
query → (rewrite/multi-query) → retrieve (dense/sparse/hybrid) → threshold → rerank →
parent expansion → dedup + token budget → generate → validate citations → answer
```

Code map:

| Stage | File |
|-------|------|
| Parse | `app/parsers/*` |
| Chunk | `app/chunking/strategies.py` |
| Embed | `app/embeddings/*` |
| Index / store | `app/vectorstore/*` |
| Retrieve | `app/retrieval/retriever.py`, `app/retrieval/bm25.py` |
| Rerank | `app/retrieval/reranker.py` |
| Generate + citations | `app/rag/pipeline.py`, `app/rag/prompts.py`, `app/rag/citations.py` |

## Chunking strategies (and when to use them)

Implemented in `app/chunking/strategies.py`; selectable via `CHUNK_STRATEGY` or per-upload
`?strategy=`.

| Strategy | How it works | Use when | Limitations |
|----------|--------------|----------|-------------|
| **fixed** | Fixed char window with overlap | Uniform text, quick baselines | Cuts mid-sentence; ignores structure |
| **recursive** | Split on separators `\n\n → \n → ". " → " "`, packing to size with overlap | General default | Still approximate on tables/code |
| **structure** | One chunk per heading/section/page block, merging small blocks | Well-structured docs (md/docx/PDF) | Depends on good parsing; variable sizes |
| **semantic** | Group adjacent sentences while similar; split at topic shifts | Prose with topic drift | Similarity proxy is lexical by default |
| **parent_child** | Large parents for context + small children for precise retrieval | Long docs where you retrieve small but generate with large context | More chunks to store/index |

**Impact on retrieval.** Smaller chunks improve precision but can lose context; larger
chunks improve recall/context but dilute relevance. Parent-child gets the best of both:
retrieve precise children, then expand to parents for generation (`Retriever.expand_parents`).

## Embeddings

`app/embeddings/`. An embedding maps text → a dense vector; similar meaning → nearby vectors.

- **Mock** (`mock.py`): deterministic hashed bag-of-words, L2-normalized. Not semantic, but
  reproducible and ordered enough for tests/demos.
- **Real** (`openai_compatible.py`): OpenAI-compatible `/embeddings` with batching + retries.
- **Similarity vs relevance**: cosine similarity measures vector closeness; *relevance* is
  whether a passage actually answers the query. High similarity ≠ relevance, which is why we
  **rerank** and apply a **threshold**.
- **Dimensionality & compatibility**: query and index vectors must come from the *same model
  and config*. Changing the embedding model (or its normalization/dimension) **invalidates
  the index** — you must re-embed all chunks. Mixing dims makes similarity meaningless.

## Metadata: stored vs embedded

A deliberately important distinction (`app/services/ingestion.py::_embed_text`):

- **Stored metadata** (document id, filename, page, section, tenant, chunk id) lives on the
  chunk and powers **filtering** and **citations**. It is *not* all crammed into the vector.
- **Embedded text** gets only a small, high-signal prefix (the **section heading**). Embedding
  every metadata field would dilute the semantic signal and hurt retrieval.

## Retrieval: dense, sparse, hybrid

`app/retrieval/retriever.py`.

- **Dense** — embed the query, cosine-search the vector store. Great for paraphrases/synonyms.
- **Sparse (BM25)** — `app/retrieval/bm25.py`, a self-contained BM25 (no external dep). Great
  for exact terms, codes, acronyms, names.
- **Hybrid (RRF)** — run both, fuse with **Reciprocal Rank Fusion**:
  `score(d) = Σ 1/(k + rank_i(d))`, `k=60`. RRF combines rankings **without** needing to
  calibrate the two score scales — robust and parameter-light.

Also supported: **metadata filtering** (and tenant scope) pushed into the store,
**configurable top-k**, **multi-query** (`retrieve_multi`), and **parent-child expansion**.

## Reranking

`app/retrieval/reranker.py`. Retrieval optimizes recall cheaply; the reranker makes a more
precise relevance judgement over the shortlist. We ship a deterministic lexical
cross-encoder stand-in (term coverage + density, stopword-filtered) so it's testable
offline; the interface matches a real cross-encoder/LLM reranker for a drop-in swap.

**Why threshold after rerank?** Raw RRF/dense scores are not comparable across modes, so the
pipeline thresholds on the reranker's calibrated 0–1 relevance (`app/rag/pipeline.py`). If
nothing clears `SCORE_THRESHOLD`, the system **abstains**.

## Context construction

`app/rag/pipeline.py`:

- **Dedup** identical/near-identical passages (`deduplicate`).
- **Token-budget** the context (`fit_token_budget`) using `app/llm/tokens.py` so we never
  blow the context window or waste tokens.
- (Optional) **parent expansion** and **compression** hooks.

## Generation, grounding, and citations

- Prompt is **versioned** (`PROMPT_VERSION` in `app/rag/prompts.py`) and includes
  **prompt-injection defenses**: retrieved text is wrapped as untrusted data and the model is
  told to ignore instructions inside it.
- The LLM must cite passages like `[1]`. We then **validate** citations
  (`app/rag/citations.py`): markers pointing outside the provided passage range are stripped,
  and each remaining marker is mapped to the real source chunk (filename/page/section).
- **Abstention**: if retrieval is empty/below threshold, or the answer isn't grounded, we
  return a safe "insufficient information" response with no citations.

## Failure modes to discuss in interviews

- Retrieval miss (bad chunking / wrong embedding model) → low recall.
- Over-long context → "lost in the middle", higher cost, truncation.
- Hallucinated citations → mitigated by citation validation.
- Prompt injection from documents → mitigated (not eliminated) by delimiters + system rules.
- Stale index after embedding-model change → must re-embed.

## Ablation experiment

`app/evaluation/runner.py` compares **baseline (dense, no rerank)** vs **improved (hybrid +
rerank)** on the same dataset. See [`EVALUATION.md`](EVALUATION.md) for the actual numbers.
