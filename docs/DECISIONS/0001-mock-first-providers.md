# ADR 0001 — Mock-first provider architecture

- **Status:** Accepted
- **Date:** Phase 0
- **Context:** The copilot must run end-to-end for demos/CI/interviews without paid API
  keys, while still supporting real OpenAI-compatible providers. Tests must be
  deterministic and free.
- **Decision:** Define typed interfaces for LLM, embeddings, and vector store. Provide a
  deterministic mock implementation for each (default `APP_MODE=mock`) and a real
  implementation selected via factories when `APP_MODE=real` and credentials are present.
  Mock outputs are deterministic (hash-seeded) and always tagged with the provider/mode so
  they are never mistaken for real model output.
- **Consequences:**
  - (+) Reproducible tests, free demos, easy provider swaps.
  - (+) Clear separation between business logic and provider SDKs.
  - (−) Mock outputs are not semantically meaningful; evaluation of *generation quality*
    with real metrics requires a real evaluator model (documented in `EVALUATION.md`).
