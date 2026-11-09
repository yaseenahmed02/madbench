"""DBHub adapter — subprocess wrapper (primary) + in-process fallback.

Upstream: https://github.com/bytebase/dbhub (Node/TS, gateway architecture).

Subprocess mode spawns the upstream ``dbhub`` binary with
``--transport=stdio`` and a DSN, as captured from the upstream tools/list. Two tools are exposed in tools/list:
``execute_sql`` and ``search_objects``.

Note on keyword filter: dbhub itself does not advertise a keyword filter;
the original safety_mechanism was the harness-side rejection. The
subprocess wrapper preserves the same harness-side filter so the
safety-mechanism capability remains accurate.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp_dbs_bench.core.backends import DatabaseBackend
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
from mcp_dbs_bench.servers._dbhub_inprocess import DBHubInProcessPlugin
from mcp_dbs_bench.servers._response_parsers import parse_dbhub_payload
from mcp_dbs_bench.servers._subprocess import SubprocessMCPClient
from mcp_dbs_bench.servers._subprocess_adapter import (
    _SubprocessConnState,
    perform_discovery,
    perform_initialize,
    perform_tool_call,
)

_LOG = logging.getLogger(__name__)
_DEFAULT_BINARY_NAME = "dbhub"
_FORBIDDEN_KEYWORDS = ("DROP ", "TRUNCATE ", "ALTER ", "GRANT ")


def _translate_request(request: ToolCallRequest) -> tuple[str, dict[str, Any]]:
    """Map ``execute_sql`` request -> upstream ``execute_sql`` (arg ``sql``)."""
    if request.name != "execute_sql":
        raise ValueError(
            f"dbhub subprocess wrapper only handles 'execute_sql'; got {request.name!r}"
        )
    return "execute_sql", {"sql": request.arguments["sql"]}


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


class DBHubPlugin:
    """DBHub plugin — subprocess for measurement, in-process for tests.

    ``use_subprocess=True`` is the measured mode; the in-process fallback
    exists for test environments without the upstream binary. See :class:`McpAlchemyPlugin` for the
    full rationale; the same convention applies here.

    Args:
        use_subprocess: When True, spawn the real ``dbhub`` upstream.
        dsn: Database DSN (e.g. ``sqlite:///:memory:``). Required when
            ``use_subprocess=True``.
        binary: Path to the ``dbhub`` binary; defaults to ``shutil.which``.
        stderr_log_path: Optional path for capturing the child's stderr.
    """

    name = "dbhub"
    version = "0.1.0"
    MEASUREMENT_REQUIRES_SUBPROCESS: bool = True

    def __init__(
        self,
        *,
        use_subprocess: bool = False,
        dsn: str | None = None,
        binary: str | None = None,
        stderr_log_path: str | None = None,
    ) -> None:
        self._use_subprocess = use_subprocess
        self._dsn = dsn
        from mcp_dbs_bench.servers._binary_discovery import find_binary
        self._binary = find_binary(_DEFAULT_BINARY_NAME, explicit=binary)
        self._stderr_log_path = stderr_log_path
        self._fallback = DBHubInProcessPlugin()

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            output_format=OutputFormat.JSON_FLAT_TABLE,
            connection_model=ConnectionModel.GATEWAY,
            declared_tool_count=2,  # execute_sql + search_objects per probe
            supports_streaming=False,
            safety_mechanism="keyword_filter",
            schema_depth="basic",
        )

    @asynccontextmanager
    async def connect(
        self, backend: DatabaseBackend, phase: PhaseContext,
    ) -> AsyncIterator[ConnectionHandle]:
        if not self._use_subprocess:
            async with self._fallback.connect(backend, phase) as handle:
                yield handle
            return

        if not self._dsn:
            raise ValueError("dbhub subprocess wrapper requires a 'dsn' constructor argument")
        from pathlib import Path

        client = SubprocessMCPClient(
            self._binary,
            args=["--transport=stdio", f"--dsn={self._dsn}"],
            stderr_log_path=Path(self._stderr_log_path) if self._stderr_log_path else None,
        )
        await perform_initialize(client, phase)
        try:
            yield ConnectionHandle(value=_SubprocessConnState(client=client, backend=backend))
        finally:
            await client.close()

    async def discover(self, conn: ConnectionHandle, phase: PhaseContext) -> ToolManifest:
        if not self._use_subprocess:
            return await self._fallback.discover(conn, phase)
        state: _SubprocessConnState = conn.value
        return await perform_discovery(state.client, phase)

    async def call_tool(
        self, conn: ConnectionHandle, request: ToolCallRequest, phase: PhaseContext,
    ) -> ToolCallResponse:
        if not self._use_subprocess:
            return await self._fallback.call_tool(conn, request, phase)
        sql = request.arguments.get("sql")
        if isinstance(sql, str):
            _enforce_keyword_filter(
                sql, write_enabled=bool(request.arguments.get("write_enabled", False))
            )
        state: _SubprocessConnState = conn.value
        return await perform_tool_call(
            client=state.client,
            backend=state.backend,
            request=request,
            phase=phase,
            request_translator=_translate_request,
            response_parser=parse_dbhub_payload,
        )


__all__ = ["DBHubPlugin"]
