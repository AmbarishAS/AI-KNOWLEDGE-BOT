"""Agent endpoints: list tools, run the agentic workflow, run deterministic workflow."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.agents.graph import AgentResult
from app.api.deps import container_dep, principal_dep
from app.container import Container
from app.domain import Principal

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    approved: bool = False
    deterministic: bool = False


@router.get("/tools")
async def list_tools(container: Container = Depends(container_dep)) -> dict:
    return {"tools": container.tool_registry.describe()}


@router.post("/run", response_model=AgentResult)
async def run_agent(
    req: AgentRequest,
    principal: Principal = Depends(principal_dep),
    container: Container = Depends(container_dep),
) -> AgentResult:
    if req.deterministic:
        return await container.agent.run_deterministic(req.question, principal)
    return await container.agent.run(req.question, principal, approved=req.approved)
