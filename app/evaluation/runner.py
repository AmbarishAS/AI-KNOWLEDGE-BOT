"""Evaluation runner: reproducible retrieval + generation evaluation with ablation.

Loads the version-controlled dataset, ingests the sample corpus into an isolated eval
tenant, then compares a **baseline** (dense-only, no rerank) against an **improved**
configuration (hybrid + rerank) on the same questions. Emits machine-readable JSON and a
human-readable Markdown report.

Relevance is judged at the **document** level (expected source filenames), which is robust
and reproducible without hand-labeling every chunk id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.config import RetrievalMode, Settings, get_settings
from app.domain import Principal
from app.embeddings import create_embedding_provider
from app.evaluation import metrics as M
from app.llm import create_llm
from app.rag import RAGConfig, RAGPipeline
from app.retrieval import Retriever, create_reranker
from app.services.ingestion import IngestionService
from app.vectorstore.memory import InMemoryVectorStore

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SAMPLE_DOCS = DATA_DIR / "sample_docs"
DEFAULT_DATASET = DATA_DIR / "eval" / "eval_dataset.json"
EVAL_TENANT = "eval"
K = 5


def load_dataset(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else DEFAULT_DATASET
    return json.loads(Path(p).read_text(encoding="utf-8"))


class EvaluationRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def _setup(self) -> tuple[Retriever, RAGPipeline]:
        from app.persistence.db import init_db

        await init_db()  # ensure metadata tables exist (CLI runs outside the app lifespan)
        embedder = create_embedding_provider(self.settings)
        store = InMemoryVectorStore()  # isolated store for eval
        svc = IngestionService(embedder, store, self.settings)
        principal = Principal(tenant=EVAL_TENANT)
        for doc in load_dataset()["documents"]:
            path = SAMPLE_DOCS / doc
            data = path.read_bytes()
            await svc.ingest(doc, data, "text/markdown", principal)
        retriever = Retriever(store, embedder, self.settings)
        pipeline = RAGPipeline(retriever, create_llm(self.settings), self.settings, create_reranker())
        return retriever, pipeline

    async def _retrieval_case(
        self,
        retriever: Retriever,
        question: str,
        relevant: set[str],
        mode: RetrievalMode,
        rerank: bool,
    ) -> dict:
        results = await retriever.retrieve(question, EVAL_TENANT, top_k=K, mode=mode)
        if rerank:
            results = create_reranker().rerank(question, results, K)
        retrieved_docs = [r.chunk.filename for r in results]
        return {
            "precision@k": M.precision_at_k(retrieved_docs, relevant, K),
            "recall@k": M.recall_at_k(retrieved_docs, relevant, K),
            "hit_rate@k": M.hit_rate_at_k(retrieved_docs, relevant, K),
            "mrr": M.reciprocal_rank(retrieved_docs, relevant),
            "ndcg@k": M.ndcg_at_k(retrieved_docs, relevant, K),
        }

    async def run(self, dataset_path: str | None = None) -> dict:
        dataset = load_dataset(dataset_path)
        retriever, pipeline = await self._setup()

        baseline_rows: list[dict] = []
        improved_rows: list[dict] = []
        generation_rows: list[dict] = []

        for case in dataset["cases"]:
            relevant = set(case.get("expected_documents", []))
            q = case["question"]

            if not case.get("should_abstain"):
                baseline_rows.append(
                    await self._retrieval_case(retriever, q, relevant, RetrievalMode.DENSE, rerank=False)
                )
                improved_rows.append(
                    await self._retrieval_case(retriever, q, relevant, RetrievalMode.HYBRID, rerank=True)
                )

            # Generation + abstention evaluation on the improved pipeline.
            answer = await pipeline.answer(q, EVAL_TENANT, config=RAGConfig(top_k=K))
            contexts = [p.chunk.text for p in answer.retrieved]
            cited = [c.filename for c in answer.citations]
            should_abstain = bool(case.get("should_abstain"))
            abstain_correct = answer.abstained == should_abstain
            generation_rows.append(
                {
                    "id": case["id"],
                    "should_abstain": should_abstain,
                    "abstained": answer.abstained,
                    "abstention_correct": abstain_correct,
                    "groundedness": M.groundedness(answer.text, contexts),
                    "answer_relevance": M.answer_relevance(answer.text, q),
                    "context_relevance": M.context_relevance(contexts, q),
                    "citation_correctness": (M.citation_correctness(cited, relevant) if not should_abstain else None),
                    "completeness": (
                        M.completeness(answer.text, case.get("expected_answer", "")) if not should_abstain else None
                    ),
                }
            )

        abstention_acc = round(sum(1 for r in generation_rows if r["abstention_correct"]) / len(generation_rows), 4)
        report = {
            "dataset": dataset.get("name"),
            "k": K,
            "mode": self.settings.app_mode.value,
            "num_cases": len(dataset["cases"]),
            "retrieval": {
                "baseline_dense": M.aggregate_retrieval(baseline_rows),
                "improved_hybrid_rerank": M.aggregate_retrieval(improved_rows),
            },
            "generation": {
                "abstention_accuracy": abstention_acc,
                "avg_groundedness": round(sum(r["groundedness"] for r in generation_rows) / len(generation_rows), 4),
            },
            "per_case": generation_rows,
        }
        return report


def render_markdown(report: dict) -> str:
    r = report["retrieval"]
    b, i = r["baseline_dense"], r["improved_hybrid_rerank"]
    lines = [
        f"# Evaluation Report — {report.get('dataset')}",
        "",
        f"- Mode: `{report['mode']}`  |  k = {report['k']}  |  cases = {report['num_cases']}",
        "",
        "## Retrieval (ablation: baseline vs improved)",
        "",
        "| Metric | Baseline (dense) | Improved (hybrid+rerank) |",
        "|--------|------------------|--------------------------|",
    ]
    for key in ["precision@k", "recall@k", "hit_rate@k", "mrr", "ndcg@k"]:
        lines.append(f"| {key} | {b.get(key, 0)} | {i.get(key, 0)} |")
    lines += [
        "",
        "## Generation",
        "",
        f"- Abstention accuracy: **{report['generation']['abstention_accuracy']}**",
        f"- Avg groundedness: **{report['generation']['avg_groundedness']}**",
        "",
        "> Note: generation metrics here are deterministic proxies. Faithful LLM-graded",
        "> scores (e.g. RAGAS) require an evaluator model and are estimates, not ground truth.",
    ]
    return "\n".join(lines)


async def run_and_save(out_dir: str = "eval_results", dataset_path: str | None = None) -> dict:
    report = await EvaluationRunner().run(dataset_path)
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report
