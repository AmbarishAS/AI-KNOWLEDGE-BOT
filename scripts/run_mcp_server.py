"""Entrypoint to run the MCP server over stdio (requires the optional ``mcp`` package).

Usage:
    poetry install --extras mcp
    poetry run python -m scripts.run_mcp_server

Exposes the Copilot's safe tools (calculator, knowledge_base_search, sql_query,
ticket_lookup, ticket_create) to any MCP-compatible client over stdio.
"""

from __future__ import annotations

from app.container import get_container
from app.mcp import serve_stdio


def main() -> None:
    container = get_container()
    serve_stdio(container.tool_registry)


if __name__ == "__main__":
    main()
