"""MCP server.

Model Context Protocol (MCP) standardizes how clients **discover** and **invoke** tools and
resources over a transport. This module exposes our existing :class:`ToolRegistry` through an
MCP-style JSON-RPC surface with two methods:

- ``tools/list``  → returns tool names, descriptions, and JSON input schemas.
- ``tools/call``  → validates arguments against the tool schema, executes, returns a result
  or a structured error (including an ``approval_required`` signal for sensitive tools).

Two transports are provided:
- :class:`InProcessTransport` (default) — used by tests and the agent; no external process.
- An optional **stdio** server using the official ``mcp`` SDK (see ``serve_stdio``), guarded
  so the package is only required when you actually run it.

MCP vs. function calling (interview point): *function calling* is the model-facing mechanism
for requesting a tool; *MCP* is the protocol that lets any compatible client discover and
call tools/resources across process boundaries. See ``docs/MCP_GUIDE.md``.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.agents.tools import ApprovalRequired, ToolError, ToolRegistry
from app.domain import Principal
from app.logging_config import get_logger

logger = get_logger(__name__)

PROTOCOL_VERSION = "mcp-demo-1.0"


class McpServer:
    """Transport-agnostic MCP request handler over a ToolRegistry."""

    def __init__(self, registry: ToolRegistry, principal: Principal | None = None) -> None:
        self._registry = registry
        self._principal = principal or Principal()

    async def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a single JSON-RPC-style MCP request."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        try:
            if method == "initialize":
                result = {"protocolVersion": PROTOCOL_VERSION, "serverName": "copilot-mcp"}
            elif method == "tools/list":
                result = {"tools": self._registry.describe()}
            elif method == "tools/call":
                result = await self._call_tool(params)
            else:
                return self._error(req_id, -32601, f"Method not found: {method}")
        except _McpError as exc:
            return self._error(req_id, exc.code, exc.message, exc.data)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    async def _call_tool(self, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        approved = bool(params.get("approved", False))
        try:
            tool = self._registry.get(name)
        except ToolError as exc:
            raise _McpError(-32602, str(exc)) from exc
        # Validate arguments against the tool's declared schema BEFORE execution.
        try:
            tool.InputModel(**arguments)
        except ValidationError as exc:
            raise _McpError(-32602, "Invalid tool arguments", {"errors": exc.errors()}) from exc
        try:
            output = await tool.run(arguments, self._principal, approved=approved)
        except ApprovalRequired as exc:
            return {
                "isError": True,
                "approval_required": True,
                "content": [{"type": "text", "text": str(exc)}],
            }
        except ToolError as exc:
            raise _McpError(-32000, f"Tool execution failed: {exc}") from exc
        return {"isError": False, "content": [{"type": "json", "json": output}]}

    @staticmethod
    def _error(req_id, code: int, message: str, data: Any = None) -> dict:
        err = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}


class _McpError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class InProcessTransport:
    """Directly connects a client to a server object (no serialization boundary).

    Mirrors the MCP request/response lifecycle for tests and the in-app agent, without
    spawning a subprocess. A real deployment would use stdio or HTTP transport instead.
    """

    def __init__(self, server: McpServer) -> None:
        self._server = server

    async def send(self, request: dict) -> dict:
        return await self._server.handle(request)


def serve_stdio(registry: ToolRegistry) -> None:  # pragma: no cover - requires mcp SDK + runtime
    """Run a real MCP server over stdio using the official ``mcp`` SDK, if installed.

    This is the production transport. We keep it optional and guarded so the default demo
    and tests do not require the SDK.
    """
    try:
        import anyio
        import mcp.types as types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise RuntimeError("The 'mcp' package is required for stdio transport. Install with: poetry install --extras mcp") from exc

    server = Server("copilot-mcp")
    bridge = McpServer(registry)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=t["name"], description=t["description"], inputSchema=t["input_schema"])
            for t in registry.describe()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        resp = await bridge.handle({"id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        import json

        return [types.TextContent(type="text", text=json.dumps(resp.get("result", resp)))]

    async def _main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_main)
