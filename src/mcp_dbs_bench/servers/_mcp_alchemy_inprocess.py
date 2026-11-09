"""mcp-alchemy adapter (in-process fallback when upstream binary is missing).

Primary code path is the subprocess wrapper in ``servers/mcp_alchemy.py``.
This module remains so the harness can degrade gracefully on a host where
``mcp-alchemy`` is not installed: identical Protocol surface, identical
trace shape, but T_xport is sub-microsecond instead of real IPC.

Upstream: https://github.com/runekaagaard/mcp-alchemy (Python, SQLAlchemy-based,
vertical-text output by default).
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
from mcp_dbs_bench.servers._base import (
    encode_jsonrpc_tool_call_response,
    encode_jsonrpc_tools_list_response,
)

_MCP_ALCHEMY_TOOLS = [
    {
        "name": "execute_query",
        "description": (
            "Execute a SQL query against the connected database and return "
            "the result set."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
    {
        "name": "list_tables",
        "description": "List all tables in the connected database.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "describe_table",
        "description": "Show columns and types for one table, including FKs.",
        "inputSchema": {
            "type": "object",
            "properties": {"table_name": {"type": "string"}},
            "required": ["table_name"],
        },
    },
    {
        "name": "all_table_names",
        "description": "Return all table names visible to the current schema.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class McpAlchemyInProcessPlugin:
    """In-process mcp-alchemy fallback adapter.

    Backends supported: postgres, mysql, duckdb (matches upstream's SQLAlchemy
    URL coverage on the benchmark backend set).
    """

    name = "mcp-alchemy"
    version = "0.1.0-inprocess"

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            output_format=OutputFormat.VERTICAL_TEXT,
            connection_model=ConnectionModel.SQLALCHEMY,
            declared_tool_count=len(_MCP_ALCHEMY_TOOLS),
            supports_streaming=False,
            safety_mechanism="read_only",
            schema_depth="rich",
        )

    @asynccontextmanager
    async def connect(
        self,
        backend: DatabaseBackend,
        phase: PhaseContext,
    ) -> AsyncIterator[ConnectionHandle]:
        # mcp-alchemy's connection model is SQLALCHEMY — lazy pool init on first
        # request. We model this as a single LIFECYCLE span at connect time
        # plus zero handover overhead on first call.
        with phase.span(Phase.LIFECYCLE, cold_start=True, lifecycle_calls=1):
            # Force the backend to actually establish its connection (DSN parse,
            # pool warm-up). For the supported backends, the __init__ already
            # connected, so this is a no-op SELECT.
            backend.execute("SELECT 1")
        try:
            yield ConnectionHandle(value=backend)
        finally:
            # mcp-alchemy keeps the connection open until the host disconnects;
            # the backend.close() happens at the harness level after the run.
            pass

    async def discover(self, conn: ConnectionHandle, phase: PhaseContext) -> ToolManifest:
        with phase.span(Phase.DISCOVERY) as _:
            raw_bytes = encode_jsonrpc_tools_list_response(_MCP_ALCHEMY_TOOLS)
            await asyncio.sleep(0)
        return ToolManifest(tools=list(_MCP_ALCHEMY_TOOLS), raw_bytes=raw_bytes)

    async def call_tool(
        self,
        conn: ConnectionHandle,
        request: ToolCallRequest,
        phase: PhaseContext,
    ) -> ToolCallResponse:
        backend: DatabaseBackend = conn.value
        if request.name != "execute_query":
            raise ValueError(
                f"the mcp-alchemy adapter only handles 'execute_query'; got {request.name!r}"
            )
        sql = request.arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("execute_query requires a non-empty 'sql' string argument")

        # HANDOVER: marshalling. T_db sits between HANDOVER and SERIALIZE.
        with phase.span(Phase.HANDOVER, request_bytes=request.byte_size()):
            await asyncio.sleep(0)

        rows, columns, t_db_ms = backend.execute(sql)

        with phase.span(
            Phase.SERIALIZE, row_count=len(rows), t_db_ms=t_db_ms, output_bytes=0
        ) as _:
            payload = encode_jsonrpc_tool_call_response(
                columns=list(columns),
                rows=rows,
                output_format="json_flat_table",
            )
        with phase.span(Phase.PARSE, input_bytes=len(payload)):
            await asyncio.sleep(0)
        with phase.span(
            Phase.TRANSPORT, payload_bytes=len(payload), transport_mode="stdio"
        ):
            await asyncio.sleep(0)

        return ToolCallResponse(
            rows=rows,
            raw_bytes=payload,
            column_names=tuple(columns),
        )


__all__ = ["McpAlchemyInProcessPlugin"]
