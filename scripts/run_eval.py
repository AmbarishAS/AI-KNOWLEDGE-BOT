"""Run the reproducible evaluation and write machine + human readable reports.

Usage:
    python -m scripts.run_eval            # writes eval_results/report.json and report.md
    python -m scripts.run_eval --dataset data/eval/eval_dataset.json
"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.evaluation.runner import render_markdown, run_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation + ablation")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default="eval_results")
    args = parser.parse_args()

    report = asyncio.run(run_and_save(args.out, args.dataset))
    print(json.dumps(report["retrieval"], indent=2))
    print("\n" + render_markdown(report))
    print(f"\nSaved JSON + Markdown reports to ./{args.out}/")


if __name__ == "__main__":
    main()
