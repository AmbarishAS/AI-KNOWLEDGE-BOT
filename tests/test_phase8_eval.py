"""Phase 8 — evaluation metrics and the reproducible runner (ablation)."""

from __future__ import annotations

from app.config import get_settings
from app.evaluation import metrics as M
from app.evaluation.runner import EvaluationRunner, render_markdown
from app.vectorstore import reset_vector_store


def test_precision_recall_mrr_ndcg():
    retrieved = ["a.md", "b.md", "c.md"]
    relevant = {"b.md"}
    assert M.precision_at_k(retrieved, relevant, 3) == pytest_approx(1 / 3)
    assert M.recall_at_k(retrieved, relevant, 3) == 1.0
    assert M.hit_rate_at_k(retrieved, relevant, 3) == 1.0
    assert M.reciprocal_rank(retrieved, relevant) == 0.5
    assert 0.0 < M.ndcg_at_k(retrieved, relevant, 3) <= 1.0


def test_abstention_and_groundedness_proxies():
    assert M.groundedness("vacation days", ["employees get vacation days"]) > 0.0
    assert M.answer_relevance("vacation days accrue", "how many vacation days") > 0.0


async def test_runner_produces_ablation_report():
    get_settings.cache_clear()
    await reset_vector_store()
    report = await EvaluationRunner().run()
    assert report["retrieval"]["baseline_dense"]
    assert report["retrieval"]["improved_hybrid_rerank"]
    # Abstention accuracy should be strong on the curated dataset.
    assert report["generation"]["abstention_accuracy"] >= 0.75
    md = render_markdown(report)
    assert "Evaluation Report" in md
    await reset_vector_store()


def pytest_approx(value, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - value) < tol

    return _A()
