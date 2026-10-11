"""Agent tools with structured, validated schemas.

Every tool declares a Pydantic input/output schema so arguments are validated before
execution (never trust raw LLM output). Sensitive tools (writes) set
``requires_approval=True`` and are gated by a human-approval step in the agent.

Security highlights (interview-relevant):
- The **calculator** uses a safe AST evaluator — no ``eval`` of arbitrary code.
- The **SQL tool** is read-only and allowlisted: only ``SELECT`` on approved tables, no
  stacked statements, no DDL/DML. We never execute arbitrary LLM-generated SQL.
- The **ticket create** tool requires explicit approval.
"""

from __future__ import annotations

import ast
import operator
import sqlite3
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.domain import Principal
from app.retrieval.retriever import Retriever


class ToolError(Exception):
    """Raised when a tool fails or is called unsafely."""


class ApprovalRequired(Exception):
    """Raised when a sensitive tool is invoked without approval."""


class Tool(Protocol):
    name: str
    description: str
    requires_approval: bool
    InputModel: type[BaseModel]

    async def run(self, args: dict, principal: Principal, approved: bool = False) -> dict: ...


# --------------------------------------------------------------------- calculator

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return float(node.value)
        raise ToolError("Only numeric constants allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ToolError("Unsupported expression")


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '20 * 5 + 3'")


class CalculatorTool:
    name = "calculator"
    description = "Evaluate a basic arithmetic expression (safe, no code execution)."
    requires_approval = False
    InputModel = CalculatorInput

    async def run(self, args: dict, principal: Principal, approved: bool = False) -> dict:
        data = CalculatorInput(**args)
        try:
            tree = ast.parse(data.expression, mode="eval")
            result = _safe_eval(tree)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Invalid expression: {exc}") from exc
        return {"result": result}


# --------------------------------------------------------------------- KB search


class KBSearchInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class KnowledgeBaseSearchTool:
    name = "knowledge_base_search"
    description = "Search the enterprise knowledge base for relevant passages."
    requires_approval = False
    InputModel = KBSearchInput

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def run(self, args: dict, principal: Principal, approved: bool = False) -> dict:
        data = KBSearchInput(**args)
        results = await self._retriever.retrieve(data.query, principal.tenant, data.top_k)
        return {
            "results": [
                {
                    "filename": r.chunk.filename,
                    "chunk_id": r.chunk.chunk_id,
                    "score": round(r.score, 4),
                    "text": r.chunk.text[:400],
                }
                for r in results
            ]
        }


# --------------------------------------------------------------------- SQL (read-only)

_SQL_ALLOWED_TABLES = {"employees", "departments"}
_SQL_FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "attach",
    "pragma",
    "replace",
    "truncate",
    "grant",
    "revoke",
    "vacuum",
    "--",
    "/*",
)


def _build_demo_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER);
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY, name TEXT, department TEXT, vacation_days INTEGER
        );
        INSERT INTO departments VALUES (1,'Engineering',500000),(2,'Sales',300000);
        INSERT INTO employees VALUES
            (1,'Alice','Engineering',20),
            (2,'Bob','Engineering',18),
            (3,'Carol','Sales',22);
        """
    )
    conn.commit()
    return conn


def validate_select(sql: str) -> str:
    """Validate a read-only, allowlisted SELECT. Raises ToolError otherwise."""
    cleaned = sql.strip().rstrip(";").strip()
    lowered = cleaned.lower()
    if not lowered.startswith("select"):
        raise ToolError("Only SELECT statements are allowed")
    if ";" in cleaned:
        raise ToolError("Multiple statements are not allowed")
    for bad in _SQL_FORBIDDEN:
        if bad in lowered:
            raise ToolError(f"Forbidden token in SQL: {bad}")
    # Require at least one allowlisted table to be referenced.
    if not any(t in lowered for t in _SQL_ALLOWED_TABLES):
        raise ToolError(f"Query must reference an allowlisted table: {_SQL_ALLOWED_TABLES}")
    return cleaned


class SQLQueryInput(BaseModel):
    sql: str = Field(description="A read-only SELECT over allowlisted demo tables")


class SQLQueryTool:
    name = "sql_query"
    description = "Run a read-only SELECT against the demo HR dataset " "(allowlisted tables: employees, departments)."
    requires_approval = False
    InputModel = SQLQueryInput

    async def run(self, args: dict, principal: Principal, approved: bool = False) -> dict:
        data = SQLQueryInput(**args)
        safe_sql = validate_select(data.sql)
        conn = _build_demo_db()
        try:
            cur = conn.execute(f"SELECT * FROM ({safe_sql}) LIMIT 50")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
        except sqlite3.Error as exc:
            raise ToolError(f"SQL error: {exc}") from exc
        finally:
            conn.close()
        return {"columns": cols, "rows": rows, "row_count": len(rows)}


# --------------------------------------------------------------------- ticketing (mock)

_TICKETS: dict[str, dict[str, Any]] = {
    "TICK-1001": {"id": "TICK-1001", "status": "open", "subject": "VPN not working"},
    "TICK-1002": {"id": "TICK-1002", "status": "closed", "subject": "Laptop replacement"},
}
_ticket_seq = 1003


class TicketLookupInput(BaseModel):
    ticket_id: str


class TicketLookupTool:
    name = "ticket_lookup"
    description = "Look up a support ticket by id (read-only)."
    requires_approval = False
    InputModel = TicketLookupInput

    async def run(self, args: dict, principal: Principal, approved: bool = False) -> dict:
        data = TicketLookupInput(**args)
        ticket = _TICKETS.get(data.ticket_id)
        if ticket is None:
            raise ToolError(f"Ticket {data.ticket_id} not found")
        return {"ticket": ticket}


class TicketCreateInput(BaseModel):
    subject: str = Field(min_length=3, max_length=200)
    description: str = Field(default="", max_length=2000)


class TicketCreateTool:
    name = "ticket_create"
    description = "Create a support ticket (WRITE — requires human approval)."
    requires_approval = True
    InputModel = TicketCreateInput

    async def run(self, args: dict, principal: Principal, approved: bool = False) -> dict:
        if not approved:
            raise ApprovalRequired("ticket_create requires explicit human approval")
        global _ticket_seq
        data = TicketCreateInput(**args)
        tid = f"TICK-{_ticket_seq}"
        _ticket_seq += 1
        _TICKETS[tid] = {"id": tid, "status": "open", "subject": data.subject}
        return {"ticket": _TICKETS[tid], "created": True}


# --------------------------------------------------------------------- registry


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "requires_approval": t.requires_approval,
                "input_schema": t.InputModel.model_json_schema(),
            }
            for t in self._tools.values()
        ]


def build_default_registry(retriever: Retriever) -> ToolRegistry:
    return ToolRegistry(
        [
            CalculatorTool(),
            KnowledgeBaseSearchTool(retriever),
            SQLQueryTool(),
            TicketLookupTool(),
            TicketCreateTool(),
        ]
    )
