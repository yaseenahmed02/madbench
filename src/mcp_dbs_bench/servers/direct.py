"""Direct-DB plugin — no-MCP engine-truth baseline.

This adapter implements the MCP server Protocol but does NOT speak MCP. It
executes SQL directly against the configured DatabaseBackend with zero
protocol layering: no subprocess, no JSON-RPC envelope, no stdio transport,
no tool discovery handshake. The result is a clean per-engine T_db ground
truth that other MCP servers can be compared against.

Why this matters
----------------
Each MCP server's instrumentation reports T_db with a slightly different
scope (mcp-alchemy includes SQLAlchemy result hydration; anthropic-pg
includes Node pg row materialization; DBHub measures more tightly). When
computing PTR = PTax / T_db across servers, the denominator is not
apples-to-apples. The DirectDBPlugin produces a canonical T_db measurement
using only the driver-level execute call, eliminating server-specific
attribution asymmetry.

Phase-span behavior
-------------------
The plugin still emits all eight phase spans so traces remain comparable to
MCP server runs:

  - DISCOVERY:  ~0 ms (synthetic, no real tools/list traffic)
  - LIFECYCLE:  driver connection setup (~few ms)
  - HANDOVER:   ~0 ms (no marshalling — argument passed natively)
  - SERIALIZE:  wraps the database execute call; t_db_ms captured on this span
  - PARSE:      ~0 ms (no JSON parsing — rows returned natively)
  - TRANSPORT:  ~0 ms (no IPC — in-process function call)

PTR for this plugin should approach 0 — that IS the engine truth.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp_dbs_bench.core.backends import DatabaseBackend
from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseContext
from mcp_dbs_bench.core.types import (
    ConnectionHandle,
    ConnectionModel,
    OutputFormat,
    ServerCapabilities,
    ToolCallRequest,
    ToolCallResponse,
    ToolManifest,
)

_DIRECT_TOOLS = [
    {
        "name": "execute_sql",
        "description": "Execute SQL directly against the underlying database driver (no MCP layer).",
        "inputSchema": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
]


class DirectDBPlugin:
    """Direct-DB plugin — no MCP, ground-truth engine measurement.

    Backends supported: postgres, mysql, duckdb. The plugin holds a single
    DatabaseBackend handle and forwards every execute_sql request to it.
    """

    name = "direct"
    version = "0.1.0"
    MEASUREMENT_REQUIRES_SUBPROCESS: bool = False

    def __init__(self, *, use_subprocess: bool = False) -> None:
        # use_subprocess is accepted but ignored — there is no subprocess
        # mode for this plugin by definition. Argument is accepted so the
        # plugin loader can call the constructor uniformly with other plugins.
        del use_subprocess

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            output_format=OutputFormat.JSON_FLAT_TABLE,
            connection_model=ConnectionModel.DIRECT,
            declared_tool_count=len(_DIRECT_TOOLS),
            supports_streaming=False,
            safety_mechanism="none",
            schema_depth="none",
        )

    @asynccontextmanager
    async def connect(
        self, backend: DatabaseBackend, phase: PhaseContext,
    ) -> AsyncIterator[ConnectionHandle]:
        # LIFECYCLE: actual driver connection establishment.
        with phase.span(Phase.LIFECYCLE, cold_start=True, lifecycle_calls=1):
            backend.execute("SELECT 1")
        try:
            yield ConnectionHandle(value=backend)
        finally:
            # Connection is owned by the harness; close happens at run teardown.
            pass

    async def discover(self, conn: ConnectionHandle, phase: PhaseContext) -> ToolManifest:
        # DISCOVERY: synthetic. No real tools/list handshake occurs.
        with phase.span(Phase.DISCOVERY) as _:
            await asyncio.sleep(0)
        return ToolManifest(tools=list(_DIRECT_TOOLS), raw_bytes=b"")

    async def call_tool(
        self, conn: ConnectionHandle, request: ToolCallRequest, phase: PhaseContext,
    ) -> ToolCallResponse:
        if request.name != "execute_sql":
            raise ValueError(
                f"direct plugin only handles 'execute_sql'; got {request.name!r}"
            )
        sql = request.arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("execute_sql requires a non-empty 'sql' argument")

        backend: DatabaseBackend = conn.value

        # HANDOVER: no marshalling — argument is a native Python string.
        with phase.span(Phase.HANDOVER, request_bytes=request.byte_size()):
            await asyncio.sleep(0)

        # Engine truth: this is the only real work the plugin does.
        rows, columns, t_db_ms = backend.execute(sql)

        # SERIALIZE: span exists so trace shape matches MCP server plugins,
        # but no JSON-RPC envelope is constructed. T_db is attached here per
        # the harness's t_db extraction contract.
        with phase.span(
            Phase.SERIALIZE, row_count=len(rows), t_db_ms=t_db_ms, output_bytes=0
        ) as _:
            await asyncio.sleep(0)

        # PARSE: no JSON parsing occurs.
        with phase.span(Phase.PARSE, input_bytes=0):
            await asyncio.sleep(0)

        # TRANSPORT: in-process function call, no IPC.
        with phase.span(
            Phase.TRANSPORT, payload_bytes=0, transport_mode="none"
        ):
            await asyncio.sleep(0)

        return ToolCallResponse(
            rows=rows,
            raw_bytes=b"",
            column_names=tuple(columns),
        )


__all__ = ["DirectDBPlugin"]
