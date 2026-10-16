# ADR 0004 — Dependency-free agent + optional LangGraph/MCP SDK

- **Status:** Accepted
- **Context:** The spec asks for LangGraph agents and the MCP SDK, but we also require the
  system to always run (tests/CI/demo) without heavy optional dependencies.
- **Decision:** Implement the agent as an explicit, typed **state machine** with the same
  node/edge/state semantics as LangGraph (`app/agents/graph.py`), and implement MCP with an
  **in-process transport** plus an **optional** official-SDK stdio server
  (`app/mcp/server.py::serve_stdio`). Optional packages (`langgraph`, `mcp`, `qdrant-client`,
  `gradio`, `ragas`) are guarded so their absence degrades gracefully.
- **Consequences:**
  - (+) The core always runs and is fully testable offline.
  - (+) Clear upgrade path to the real libraries without rearchitecting.
  - (−) The bundled agent/MCP are intentionally simpler than the full frameworks; documented
    as such so nothing is overclaimed.
