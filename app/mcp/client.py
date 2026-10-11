"""MCP client.

Discovers tools from an MCP server and invokes them over a transport. The client is what an
agent uses to consume MCP tools — demonstrating that tool discovery/invocation is decoupled
from the agent's own logic (the whole point of MCP).
"""

from __future__ import annotations

import itertools
from typing import Any

from app.logging_config import get_logger
from app.mcp.server import InProcessTransport

logger = get_logger(__name__)


class McpClient:
    def __init__(self, transport: InProcessTransport) -> None:
        self._transport = transport
        self._ids = itertools.count(1)
        self._tools_cache: list[dict] | None = None

    async def initialize(self) -> dict:
        resp = await self._transport.send({"jsonrpc": "2.0", "id": next(self._ids), "method": "initialize"})
        return resp.get("result", {})

    async def list_tools(self) -> list[dict]:
        resp = await self._transport.send({"jsonrpc": "2.0", "id": next(self._ids), "method": "tools/list"})
        if "error" in resp:
            raise McpClientError(resp["error"]["message"])
        self._tools_cache = resp["result"]["tools"]
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict[str, Any], approved: bool = False) -> dict:
        resp = await self._transport.send(
            {
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments, "approved": approved},
            }
        )
        if "error" in resp:
            raise McpClientError(resp["error"]["message"], data=resp["error"].get("data"))
        result = resp["result"]
        if result.get("approval_required"):
            raise McpApprovalRequired(result["content"][0]["text"])
        if result.get("isError"):
            raise McpClientError(result["content"][0]["text"])
        # Unwrap the first json content block for convenience.
        for block in result.get("content", []):
            if block.get("type") == "json":
                return block["json"]
        return result


class McpClientError(Exception):
    def __init__(self, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.data = data


class McpApprovalRequired(Exception):
    pass
