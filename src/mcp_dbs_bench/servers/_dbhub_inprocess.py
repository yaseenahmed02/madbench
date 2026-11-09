"""DBHub adapter (in-process fallback when upstream binary is missing).

Primary code path is the subprocess wrapper in ``servers/dbhub.py``. This
module remains so the harness can degrade gracefully on a host where
``dbhub`` is not installed.

Upstream: https://github.com/bytebase/dbhub (Node/TS, gateway architecture).
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
        "name": "execute_sql",
        "description": (
            "Run a SQL query against the connected database. Statements "
            "matching the keyword filter (DROP, TRUNCATE, ALTER, GRANT) "
            "are rejected unless write_enabled=true."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "write_enabled": {"type": "boolean", "default": False},
            },
            "required": ["sql"],
        },
    },
    {
        "name": "list_schemas",
        "description": "List all schemas, tables, and views in the connected DB.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

_FORBIDDEN_KEYWORDS = ("DROP ", "TRUNCATE ", "ALTER ", "GRANT ")


class DBHubInProcessPlugin:
    """In-process DBHub fallback adapter. Covers postgres, mysql, duckdb."""

    name = "dbhub"
    version = "0.1.0-inprocess"

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            output_format=OutputFormat.JSON_FLAT_TABLE,
            connection_model=ConnectionModel.GATEWAY,
            declared_tool_count=len(_TOOLS),
            supports_streaming=False,
            safety_mechanism="keyword_filter",
            schema_depth="basic",
        )

    @asynccontextmanager
    async def connect(
        self, backend: DatabaseBackend, phase: PhaseContext,
    ) -> AsyncIterator[ConnectionHandle]:
        # DBHub uses a gateway connection: one persistent backend handle.
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
        if request.name != "execute_sql":
            raise ValueError(f"the dbhub adapter only handles 'execute_sql'; got {request.name!r}")
        sql = request.arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("execute_sql requires a non-empty 'sql' string argument")
        write_enabled = bool(request.arguments.get("write_enabled", False))

        with phase.span(Phase.HANDOVER, request_bytes=request.byte_size()):
            self._enforce_keyword_filter(sql, write_enabled=write_enabled)

        rows, columns, t_db_ms = backend.execute(sql)

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

    @staticmethod
    def _enforce_keyword_filter(sql: str, *, write_enabled: bool) -> None:
        if write_enabled:
            return
        upper = sql.upper()
        for kw in _FORBIDDEN_KEYWORDS:
            if kw in upper:
                raise ValueError(
                    f"dbhub keyword filter rejected query: contains '{kw.strip()}'. "
                    "Pass write_enabled=true to override."
                )


__all__ = ["DBHubInProcessPlugin"]
