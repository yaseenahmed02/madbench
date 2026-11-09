"""Anthropic PG adapter — subprocess wrapper (primary) + in-process fallback.

Upstream: ``@modelcontextprotocol/server-postgres`` (Node, the community-
maintained successor of the archived Anthropic PG server).

Subprocess mode spawns the upstream binary with the postgres DSN passed
positionally. The upstream exposes a single
tool ``query``.
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
from mcp_dbs_bench.servers._anthropic_pg_inprocess import AnthropicPgInProcessPlugin
from mcp_dbs_bench.servers._response_parsers import parse_anthropic_pg_payload
from mcp_dbs_bench.servers._subprocess import SubprocessMCPClient
from mcp_dbs_bench.servers._subprocess_adapter import (
    _SubprocessConnState,
    perform_discovery,
    perform_initialize,
    perform_tool_call,
)

_LOG = logging.getLogger(__name__)
_DEFAULT_BINARY_NAME = "mcp-server-postgres"


def _translate_request(request: ToolCallRequest) -> tuple[str, dict[str, Any]]:
    if request.name != "query":
        raise ValueError(
            f"anthropic-pg subprocess wrapper only handles 'query'; got {request.name!r}"
        )
    return "query", {"sql": request.arguments["sql"]}


class AnthropicPgPlugin:
    """Anthropic PG plugin — subprocess for measurement, in-process for tests.

    ``use_subprocess=True`` is the measured mode; the in-process fallback exists for test environments
    without the upstream binary. See :class:`McpAlchemyPlugin` for the
    full rationale.

    Args:
        use_subprocess: When True, spawn the real upstream PG MCP server.
        dsn: PostgreSQL DSN (``postgresql://user:pass@host:port/db``).
            Required when ``use_subprocess=True``.
        binary: Path to the ``mcp-server-postgres`` binary.
        stderr_log_path: Optional path for capturing the child's stderr.
    """

    name = "anthropic-pg"
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
        self._fallback = AnthropicPgInProcessPlugin()

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            output_format=OutputFormat.JSON_FLAT_TABLE,
            connection_model=ConnectionModel.IMPLICIT_POOL,
            declared_tool_count=1,
            supports_streaming=False,
            safety_mechanism="read_only_tx",
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
            raise ValueError("anthropic-pg subprocess wrapper requires a 'dsn' constructor argument")
        if backend.name != "postgres":
            raise ValueError(
                f"anthropic-pg only supports postgres; got {backend.name!r}"
            )
        from pathlib import Path

        client = SubprocessMCPClient(
            self._binary,
            args=[self._dsn],
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
        state: _SubprocessConnState = conn.value
        return await perform_tool_call(
            client=state.client,
            backend=state.backend,
            request=request,
            phase=phase,
            request_translator=_translate_request,
            response_parser=parse_anthropic_pg_payload,
        )


__all__ = ["AnthropicPgPlugin"]
