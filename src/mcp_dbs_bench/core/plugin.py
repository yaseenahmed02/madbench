"""MCPServerPlugin Protocol — the contract for MCP server adapters.

A class is a valid plugin if and only if it has:

  - `name: str` and `version: str` class/instance attributes
  - `capabilities() -> ServerCapabilities`            (sync, pure)
  - `connect(backend, phase) -> AsyncContextManager`  (async ctxmgr)
  - `discover(conn, phase) -> ToolManifest`            (async)
  - `call_tool(conn, request, phase) -> ToolCallResponse`  (async)

No inheritance is required: plugins satisfy a Protocol rather than an ABC.
The Protocol is
`@runtime_checkable` so the registry can reject malformed plugins at
discovery time with a useful error message.
"""

from __future__ import annotations

from typing import AsyncContextManager, Protocol, runtime_checkable

from mcp_dbs_bench.core.phase_context import PhaseContext
from mcp_dbs_bench.core.types import (
    ConnectionHandle,
    ServerCapabilities,
    ToolCallRequest,
    ToolCallResponse,
    ToolManifest,
)


@runtime_checkable
class MCPServerPlugin(Protocol):
    """Contract for any MCP server registered with MCP-DBS-Bench.

    The harness calls the lifecycle in this order, once per run:

        1. capabilities()              — pure, no side effects.
        2. async with connect():       — establishes lifecycle.
        3. await discover():           — issues tools/list.
        4. await call_tool(...):       — issues tools/call (n times).
        5. context exit:               — closes lifecycle.

    Plugins MUST NOT call OTel APIs directly. The `phase` argument is a
    `PhaseContext` (state machine); the plugin records phase boundaries on
    it, and the harness translates those records into spans. This is the
    only design that preserves Property 2 (exclusivity) across plugins
    written by different authors.
    """

    name: str
    version: str

    def capabilities(self) -> ServerCapabilities: ...

    def connect(
        self,
        backend: object,  # DatabaseBackend; typed as object to avoid forward ref cycle
        phase: PhaseContext,
    ) -> AsyncContextManager[ConnectionHandle]: ...

    async def discover(
        self,
        conn: ConnectionHandle,
        phase: PhaseContext,
    ) -> ToolManifest: ...

    async def call_tool(
        self,
        conn: ConnectionHandle,
        request: ToolCallRequest,
        phase: PhaseContext,
    ) -> ToolCallResponse: ...
