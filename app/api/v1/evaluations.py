"""Evaluation endpoint: run the reproducible evaluation and return the report."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import principal_dep
from app.api.schemas import EvaluationRequest
from app.domain import Principal
from app.evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluation"])


@router.post("/run")
async def run_evaluation(
    req: EvaluationRequest,
    principal: Principal = Depends(principal_dep),
) -> dict:
    report = await EvaluationRunner().run(req.dataset)
    return report
