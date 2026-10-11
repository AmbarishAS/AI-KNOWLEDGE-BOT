"""Evaluation: deterministic retrieval metrics + generation proxies + runner."""

from app.evaluation import metrics
from app.evaluation.runner import EvaluationRunner, load_dataset, render_markdown, run_and_save

__all__ = [
    "metrics",
    "EvaluationRunner",
    "load_dataset",
    "render_markdown",
    "run_and_save",
]
