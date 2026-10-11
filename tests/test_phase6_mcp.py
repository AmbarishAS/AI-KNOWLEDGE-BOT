"""Phase 6 — MCP server/client: discovery, invocation, validation, errors, approval."""

from __future__ import annotations

import pytest

from app.agents import build_default_registry
from app.config import get_settings
from app.domain import Principal
from app.embeddings import create_embedding_provider
from app.mcp import InProcessTransport, McpApprovalRequired, McpClient, McpClientError, McpServer
from app.retrieval import Retriever
from app.services.ingestion import IngestionService
from app.vectorstore import create_vector_store, reset_vector_store

POLICY = b"# Vacation\n\nEmployees accrue 20 vacation days per year."


@pytest.fixture
async def mcp_client():
    get_settings.cache_clear()
    settings = get_settings()
    await reset_vector_store()
    embedder = create_embedding_provider(settings)
    store = create_vector_store(settings, embedder.dim)
    await IngestionService(embedder, store, settings).ingest(
        "vacation.md", POLICY, "text/markdown", Principal(tenant="acme")
    )
    retriever = Retriever(store, embedder, settings)
    registry = build_default_registry(retriever)
    server = McpServer(registry, principal=Principal(tenant="acme"))
    client = McpClient(InProcessTransport(server))
    yield client
    await reset_vector_store()


async def test_initialize_and_tool_discovery(mcp_client):
    init = await mcp_client.initialize()
    assert "protocolVersion" in init
    tools = await mcp_client.list_tools()
    names = {t["name"] for t in tools}
    assert {"calculator", "knowledge_base_search", "sql_query"} <= names
    # Each tool advertises a JSON input schema (discovery contract).
    assert all("input_schema" in t for t in tools)


async def test_invoke_calculator_tool(mcp_client):
    out = await mcp_client.call_tool("calculator", {"expression": "20 + 22"})
    assert out["result"] == 42


async def test_invoke_knowledge_search_tool(mcp_client):
    out = await mcp_client.call_tool("knowledge_base_search", {"query": "vacation days", "top_k": 2})
    assert out["results"]


async def test_invalid_arguments_rejected(mcp_client):
    with pytest.raises(McpClientError):
        await mcp_client.call_tool("calculator", {"wrong_field": "x"})


async def test_unknown_tool_errors(mcp_client):
    with pytest.raises(McpClientError):
        await mcp_client.call_tool("does_not_exist", {})


async def test_sql_tool_blocks_mutation_via_mcp(mcp_client):
    with pytest.raises(McpClientError):
        await mcp_client.call_tool("sql_query", {"sql": "DROP TABLE employees"})


async def test_write_tool_requires_approval_via_mcp(mcp_client):
    with pytest.raises(McpApprovalRequired):
        await mcp_client.call_tool("ticket_create", {"subject": "New monitor"})
    out = await mcp_client.call_tool("ticket_create", {"subject": "New monitor"}, approved=True)
    assert out["created"] is True
