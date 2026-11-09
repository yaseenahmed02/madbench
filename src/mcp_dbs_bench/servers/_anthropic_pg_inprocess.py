"""Anthropic PG adapter (in-process fallback when upstream binary is missing).

Primary code path is the subprocess wrapper in ``servers/anthropic_pg.py``.
This module remains so the harness can degrade gracefully on a host where
``mcp-server-postgres`` is not installed.

Upstream: https://github.com/modelcontextprotocol/server-postgres (Node).
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

_TOOLS = [
    {
        "name": "query",
        "description": (
            "Run a read-only SQL query against the connected PostgreSQL "
            "instance. Wrapped in a READ ONLY transaction."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
]


class AnthropicPgInProcessPlugin:
    """In-process Anthropic-PG fallback adapter. Single tool, JSON flat-table output."""

    name = "anthropic-pg"
    version = "0.1.0-inprocess"

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            output_format=OutputFormat.JSON_FLAT_TABLE,
            connection_model=ConnectionModel.IMPLICIT_POOL,
            declared_tool_count=len(_TOOLS),
            supports_streaming=False,
            safety_mechanism="read_only_tx",
            schema_depth="basic",
        )

    @asynccontextmanager
    async def connect(
        self, backend: DatabaseBackend, phase: PhaseContext,
    ) -> AsyncIterator[ConnectionHandle]:
        if backend.name != "postgres":
            raise ValueError(
                f"anthropic-pg only supports postgres; got {backend.name!r}"
            )
        with phase.span(Phase.LIFECYCLE, cold_start=True, lifecycle_calls=1):
            backend.execute("SELECT 1")
        yield ConnectionHandle(value=backend)

    async def discover(self, conn: ConnectionHandle, phase: PhaseContext) -> ToolManifest:
        with phase.span(Phase.DISCOVERY):
            raw = encode_jsonrpc_tools_list_response(_TOOLS)
            await asyncio.sleep(0)
        return ToolManifest(tools=list(_TOOLS), raw_bytes=raw)

    async def call_tool(
        self, conn: ConnectionHandle, request: ToolCallRequest, phase: PhaseContext,
    ) -> ToolCallResponse:
        backend: DatabaseBackend = conn.value
        if request.name != "query":
            raise ValueError(f"anthropic-pg only handles 'query'; got {request.name!r}")
        sql = request.arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("query requires a non-empty 'sql' string argument")

        with phase.span(Phase.HANDOVER, request_bytes=request.byte_size()):
            # Open the BEGIN READ ONLY transaction that upstream wraps every query in.
            backend.execute("BEGIN READ ONLY")

        rows, columns, t_db_ms = backend.execute(sql)
        backend.execute("COMMIT")

        with phase.span(Phase.SERIALIZE, row_count=len(rows), t_db_ms=t_db_ms):
            payload = encode_jsonrpc_tool_call_response(
                columns=list(columns), rows=rows, output_format="json_flat_table",
            )
        with phase.span(Phase.PARSE, input_bytes=len(payload)):
            await asyncio.sleep(0)
        with phase.span(Phase.TRANSPORT, payload_bytes=len(payload), transport_mode="stdio"):
            await asyncio.sleep(0)

        return ToolCallResponse(
            rows=rows, raw_bytes=payload, column_names=tuple(columns),
        )


__all__ = ["AnthropicPgInProcessPlugin"]
