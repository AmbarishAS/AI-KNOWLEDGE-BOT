"""Phase 5 — agent workflow, safe tools, approval gates, SQL allowlisting."""

from __future__ import annotations

import pytest

from app.agents import ToolError, build_default_registry, validate_select
from app.agents.graph import Agent
from app.agents.tools import ApprovalRequired, CalculatorTool, SQLQueryTool, TicketCreateTool
from app.config import get_settings
from app.domain import Principal
from app.embeddings import create_embedding_provider
from app.llm import create_llm
from app.rag import RAGPipeline
from app.retrieval import Retriever, create_reranker
from app.services.ingestion import IngestionService
from app.vectorstore import create_vector_store, reset_vector_store

POLICY = b"""# Vacation Policy

Full-time employees accrue 20 days of paid vacation per year.
"""


@pytest.fixture
async def agent():
    get_settings.cache_clear()
    settings = get_settings()
    await reset_vector_store()
    embedder = create_embedding_provider(settings)
    store = create_vector_store(settings, embedder.dim)
    await IngestionService(embedder, store, settings).ingest(
        "vacation.md", POLICY, "text/markdown", Principal(tenant="acme")
    )
    retriever = Retriever(store, embedder, settings)
    pipeline = RAGPipeline(retriever, create_llm(settings), settings, create_reranker())
    registry = build_default_registry(retriever)
    yield Agent(registry, pipeline, settings)
    await reset_vector_store()


# ---- tool-level safety ----


async def test_calculator_rejects_code_execution():
    tool = CalculatorTool()
    with pytest.raises(ToolError):
        await tool.run({"expression": "__import__('os').system('ls')"}, Principal())
    out = await tool.run({"expression": "20 * 5 + 3"}, Principal())
    assert out["result"] == 103


def test_sql_validator_blocks_mutations_and_stacking():
    with pytest.raises(ToolError):
        validate_select("DROP TABLE employees")
    with pytest.raises(ToolError):
        validate_select("SELECT * FROM employees; DELETE FROM employees")
    with pytest.raises(ToolError):
        validate_select("SELECT * FROM secret_table")  # not allowlisted
    assert "employees" in validate_select("SELECT name FROM employees").lower()


async def test_sql_tool_runs_readonly_query():
    out = await SQLQueryTool().run({"sql": "SELECT name FROM employees"}, Principal())
    assert out["row_count"] == 3


async def test_ticket_create_requires_approval():
    tool = TicketCreateTool()
    with pytest.raises(ApprovalRequired):
        await tool.run({"subject": "New laptop"}, Principal(), approved=False)
    out = await tool.run({"subject": "New laptop"}, Principal(), approved=True)
    assert out["created"] is True


# ---- agent-level routing ----


async def test_agent_routes_calculation(agent):
    result = await agent.run("What is 12 * 4?", Principal(tenant="acme"))
    assert result.intent == "calculation"
    assert "48" in result.answer


async def test_agent_routes_knowledge_and_grounds(agent):
    result = await agent.run("How many vacation days do I get?", Principal(tenant="acme"))
    assert result.intent == "knowledge"
    assert result.answer


async def test_agent_sql_intent(agent):
    result = await agent.run("How many employees are there?", Principal(tenant="acme"))
    assert result.intent == "structured_data"
    assert result.tool_result is not None


async def test_agent_write_tool_pends_approval(agent):
    result = await agent.run("Please create a ticket for broken monitor", Principal(tenant="acme"))
    assert result.intent == "ticket_create"
    assert result.pending_approval is True


async def test_agent_write_tool_executes_with_approval(agent):
    result = await agent.run("Please create a ticket for broken monitor", Principal(tenant="acme"), approved=True)
    assert result.pending_approval is False
    assert result.tool_result and result.tool_result.get("created") is True


async def test_agent_is_bounded(agent):
    result = await agent.run("How many vacation days?", Principal(tenant="acme"))
    assert result.steps_taken <= get_settings().agent_max_steps + 2


async def test_deterministic_workflow(agent):
    result = await agent.run_deterministic("How many vacation days?", Principal(tenant="acme"))
    assert result.tool_used == "knowledge_base_search"
    assert len(result.trace) == 2
