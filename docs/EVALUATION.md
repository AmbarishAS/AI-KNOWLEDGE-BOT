# Evaluation

## Two families of metrics

1. **Deterministic retrieval metrics** — need only ground-truth relevant items; no LLM,
   fully reproducible. Implemented in `app/evaluation/metrics.py`:
   - **Precision@k** — fraction of top-k that are relevant.
   - **Recall@k** — fraction of relevant items found in top-k.
   - **Hit-Rate@k** — did any relevant item appear in top-k?
   - **MRR** — reciprocal rank of the first relevant item (ranking quality).
   - **nDCG@k** — rank-discounted gain (rewards putting relevant items higher).
2. **Generation / RAG metrics** — harder; some need an evaluator model. We ship deterministic
   **proxies** and clearly label them:
   - **Groundedness** (answer tokens supported by context), **answer relevance**,
     **context relevance**, **citation correctness**, **completeness** vs reference,
     **abstention correctness**.

> Honesty note: proxy scores are directional, not ground truth. Faithful LLM-graded metrics
> (e.g. **RAGAS**) require an evaluator model and are themselves **estimates** — never present
> evaluator scores as unquestionable truth.

## Dataset

Version-controlled at `data/eval/eval_dataset.json`. Each case has a question, expected
answer, expected source documents, and a `should_abstain` flag (including questions that
**must** be refused, e.g. "What is the capital of France?"). Relevance is judged at the
**document** level (robust and reproducible without hand-labeling every chunk).

## Running it

```powershell
python -m scripts.run_eval                 # writes eval_results/report.json + report.md
# or via API:
curl -X POST http://localhost:8000/api/v1/evaluations/run -H "X-API-Key: changeme-dev-key" -d '{}'
```

## Ablation: baseline vs improved

The runner compares **baseline (dense, no rerank)** against **improved (hybrid + rerank)** on
the same questions. Actual output from this environment (mock mode):

| Metric | Baseline (dense) | Improved (hybrid+rerank) |
|--------|------------------|--------------------------|
| precision@5 | 0.333 | 0.333 |
| recall@5 | 1.000 | 1.000 |
| hit_rate@5 | 1.000 | 1.000 |
| MRR | 0.917 | **1.000** |
| nDCG@5 | 0.939 | **1.000** |

Generation: **abstention accuracy 0.875**, **avg groundedness 0.76**.

Interpretation: on this small, clean corpus recall is already saturated, so the win shows up
in **ranking quality** (MRR and nDCG reach 1.0 with hybrid + rerank). Precision@5 is 0.333
because each question has ~1 relevant document out of 5 retrieved — expected at document
granularity.

> Because mock embeddings are not semantic, absolute numbers reflect the deterministic mock
> provider. Switch to `APP_MODE=real` to evaluate a real embedding/LLM stack; the method and
> commands are identical.

## Reproducibility

- Deterministic mock providers → stable scores across runs.
- Environment recorded here: Python 3.13 (local) / 3.12 (Docker/CI), in-memory vector store,
  k=5, dataset `enterprise-policies-eval-v1`.
- Do not invent benchmark numbers — always re-run `scripts.run_eval` and cite the environment.

## Tests

`tests/test_phase8_eval.py` covers metric correctness and asserts the ablation report
contains both configurations and strong abstention accuracy.
