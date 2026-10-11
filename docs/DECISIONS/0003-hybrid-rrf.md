# ADR 0003 — Hybrid retrieval via Reciprocal Rank Fusion

- **Status:** Accepted
- **Context:** Dense retrieval handles semantics; BM25 handles exact terms. We need to combine
  them, but their score scales are not comparable.
- **Decision:** Fuse the two ranked lists with **Reciprocal Rank Fusion** (`score = Σ 1/(k +
  rank)`, `k=60`) in `app/retrieval/retriever.py`. Apply the abstention **threshold after
  reranking** (calibrated 0–1), not on raw RRF/dense scores.
- **Consequences:**
  - (+) No score normalization/calibration needed; robust and simple.
  - (+) Thresholding on reranker relevance gives meaningful abstention.
  - (−) RRF ignores score magnitudes (rank-only); fine for this use case, but weighted fusion
    could help when one retriever is much stronger.
