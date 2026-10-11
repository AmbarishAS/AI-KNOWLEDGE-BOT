"""Bounded agent workflow.

We model the agent as an explicit state machine (nodes + edges + typed state), which is the
same mental model as LangGraph. If ``langgraph`` is installed we expose a compiled graph via
``build_langgraph`` (optional); the dependency-free :class:`Agent` below is the default so
the system always runs. See ``docs/AGENTS_AND_LANGGRAPH.md``.

Safety rails:
- ``max_steps`` prevents infinite loops.
- ``max_retries`` bounds tool retries.
- sensitive (write) tools require explicit approval (human-in-the-loop).
- all tool inputs are validated by the tool's Pydantic schema before execution.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.agents.tools import ApprovalRequired, ToolError, ToolRegistry
from app.config import Settings
from app.domain import Principal
from app.logging_config import get_logger
from app.rag import RAGPipeline

logger = get_logger(__name__)

_TICKET_RE = re.compile(r"\bTICK-\d+\b", re.IGNORECASE)


class AgentStep(BaseModel):
    node: str
    detail: str = ""
    tool: str | None = None
    ok: bool = True


class AgentState(BaseModel):
    question: str
    tenant: str
    intent: str = "unknown"
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: dict[str, Any] | None = None
    final_answer: str = ""
    steps: list[AgentStep] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    pending_approval: bool = False
    approved: bool = False
    done: bool = False


class AgentResult(BaseModel):
    answer: str
    intent: str
    tool_used: str | None
    tool_result: dict[str, Any] | None
    pending_approval: bool
    trace: list[AgentStep]
    steps_taken: int


class Agent:
    def __init__(
        self,
        registry: ToolRegistry,
        pipeline: RAGPipeline,
        settings: Settings,
    ) -> None:
        self._registry = registry
        self._pipeline = pipeline
        self._settings = settings

    # ---- Node: classify intent & choose tool (agentic routing) -----------------
    def classify(self, state: AgentState) -> AgentState:
        q = state.question
        ql = q.lower()
        if _TICKET_RE.search(q) and ("look" in ql or "status" in ql or "find" in ql):
            state.intent = "ticket_lookup"
            state.tool_name = "ticket_lookup"
            state.tool_args = {"ticket_id": _TICKET_RE.search(q).group(0).upper()}
        elif ("create" in ql or "open" in ql or "raise" in ql) and "ticket" in ql:
            state.intent = "ticket_create"
            state.tool_name = "ticket_create"
            subject = re.sub(r".*ticket (for|about|:)?", "", q, flags=re.IGNORECASE).strip()
            state.tool_args = {"subject": subject or "New ticket from agent"}
        elif re.search(r"[0-9].*[-+*/].*[0-9]|calculate|compute|sum of|product of", ql):
            state.intent = "calculation"
            state.tool_name = "calculator"
            state.tool_args = {"expression": _extract_expression(q)}
        elif any(w in ql for w in ("employee", "department", "how many employees", "staff headcount", "budget")):
            state.intent = "structured_data"
            state.tool_name = "sql_query"
            state.tool_args = {"sql": _nl_to_sql(ql)}
        else:
            state.intent = "knowledge"
            state.tool_name = "knowledge_base_search"
            state.tool_args = {"query": q, "top_k": self._settings.top_k}
        state.steps.append(AgentStep(node="classify", detail=state.intent, tool=state.tool_name))
        return state

    # ---- Node: approval gate ---------------------------------------------------
    def check_approval(self, state: AgentState) -> AgentState:
        tool = self._registry.get(state.tool_name)  # type: ignore[arg-type]
        if tool.requires_approval and not state.approved:
            state.pending_approval = True
            state.done = True
            state.final_answer = (
                f"The action '{tool.name}' requires human approval before it can run. "
                f"Re-submit with approval to proceed."
            )
            state.steps.append(AgentStep(node="approval", detail="approval required", tool=tool.name, ok=False))
        return state

    # ---- Node: execute tool with bounded retries -------------------------------
    async def execute(self, state: AgentState, principal: Principal) -> AgentState:
        tool = self._registry.get(state.tool_name)  # type: ignore[arg-type]
        for attempt in range(self._settings.agent_max_retries + 1):
            try:
                state.tool_result = await tool.run(state.tool_args, principal, approved=state.approved)
                state.steps.append(AgentStep(node="execute", detail=f"attempt {attempt + 1}", tool=tool.name))
                return state
            except ApprovalRequired as exc:
                state.pending_approval = True
                state.errors.append(str(exc))
                state.steps.append(AgentStep(node="execute", detail=str(exc), tool=tool.name, ok=False))
                return state
            except (ToolError, Exception) as exc:  # noqa: BLE001 - bounded, logged
                state.errors.append(f"{tool.name}: {exc}")
                state.steps.append(AgentStep(node="execute", detail=str(exc), tool=tool.name, ok=False))
        return state

    # ---- Node: finalize answer -------------------------------------------------
    async def finalize(self, state: AgentState, principal: Principal) -> AgentState:
        if state.intent == "knowledge":
            answer = await self._pipeline.answer(state.question, state.tenant)
            state.final_answer = answer.text
            state.tool_result = {
                "citations": [c.model_dump() for c in answer.citations],
                "abstained": answer.abstained,
            }
        elif state.tool_result is not None:
            state.final_answer = _summarize(state.intent, state.tool_result)
        elif state.errors:
            state.final_answer = f"I couldn't complete the request: {state.errors[-1]}"
        else:
            state.final_answer = "I couldn't determine how to help with that."
        state.steps.append(AgentStep(node="finalize", detail=state.intent))
        state.done = True
        return state

    # ---- Orchestration (bounded) ----------------------------------------------
    async def run(self, question: str, principal: Principal, approved: bool = False) -> AgentResult:
        state = AgentState(question=question, tenant=principal.tenant, approved=approved)
        max_steps = self._settings.agent_max_steps
        state = self.classify(state)
        state = self.check_approval(state)
        if not state.done:
            state = await self.execute(state, principal)
        if not state.done and len(state.steps) <= max_steps:
            state = await self.finalize(state, principal)
        logger.info(
            "agent run",
            extra={
                "extra": {
                    "intent": state.intent,
                    "tool": state.tool_name,
                    "steps": len(state.steps),
                    "errors": len(state.errors),
                    "pending_approval": state.pending_approval,
                }
            },
        )
        return AgentResult(
            answer=state.final_answer,
            intent=state.intent,
            tool_used=state.tool_name,
            tool_result=state.tool_result,
            pending_approval=state.pending_approval,
            trace=state.steps,
            steps_taken=len(state.steps),
        )

    async def run_deterministic(self, question: str, principal: Principal) -> AgentResult:
        """A fixed workflow (no tool choice): always KB-search → grounded answer.

        Use this when the sequence of steps is known in advance — it is cheaper and more
        predictable than letting the model choose tools (see docs).
        """
        answer = await self._pipeline.answer(question, principal.tenant)
        trace = [
            AgentStep(node="retrieve", detail="knowledge_base_search", tool="knowledge_base_search"),
            AgentStep(node="generate", detail="grounded answer"),
        ]
        return AgentResult(
            answer=answer.text,
            intent="knowledge",
            tool_used="knowledge_base_search",
            tool_result={"citations": [c.model_dump() for c in answer.citations]},
            pending_approval=False,
            trace=trace,
            steps_taken=len(trace),
        )


# ---- helpers ---------------------------------------------------------------

_EXPR_RE = re.compile(r"[-+*/().\d\s]+")


def _extract_expression(text: str) -> str:
    matches = _EXPR_RE.findall(text)
    candidate = max(matches, key=len).strip() if matches else ""
    return candidate or "0"


def _nl_to_sql(ql: str) -> str:
    """Tiny, safe natural-language → SQL mapping for the demo (allowlisted only)."""
    if "how many employees" in ql or "number of employees" in ql:
        return "SELECT COUNT(*) AS employee_count FROM employees"
    if "department" in ql and "budget" in ql:
        return "SELECT name, budget FROM departments"
    if "vacation" in ql:
        return "SELECT name, vacation_days FROM employees ORDER BY vacation_days DESC"
    return "SELECT name, department FROM employees"


def _summarize(intent: str, result: dict) -> str:
    if intent == "calculation":
        return f"The result is {result.get('result')}."
    if intent == "structured_data":
        rows = result.get("rows", [])
        return f"Query returned {len(rows)} row(s): {rows[:5]}"
    if intent == "ticket_lookup":
        t = result.get("ticket", {})
        return f"Ticket {t.get('id')} is '{t.get('subject')}' (status: {t.get('status')})."
    if intent == "ticket_create":
        t = result.get("ticket", {})
        return f"Created ticket {t.get('id')}: '{t.get('subject')}'."
    return str(result)
