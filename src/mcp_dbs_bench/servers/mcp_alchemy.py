"""mcp-alchemy adapter — subprocess wrapper (primary) + in-process fallback.

Upstream: https://github.com/runekaagaard/mcp-alchemy (Python, SQLAlchemy-based).

Subprocess mode (``use_subprocess=True``) spawns the upstream ``mcp-alchemy``
binary, performs the upstream JSON-RPC handshake, and routes every ``tools/call`` through
the real upstream process. The wire bytes recorded on each phase reflect the
actual stdio I/O between the harness and the child.

In-process mode (default) re-uses the harness backend directly so the trace
shape and byte counts are wire-format identical to the subprocess path. This
fallback exists so the suite can run on hosts where ``mcp-alchemy`` is not
installed.

Tool-name mapping (subprocess):

  - Harness tool ``execute_query`` -> upstream ``execute_query`` (arg ``query``)
"""

from __future__ import annotations

import logging
import shutil
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
from mcp_dbs_bench.servers._mcp_alchemy_inprocess import McpAlchemyInProcessPlugin
from mcp_dbs_bench.servers._response_parsers import parse_mcp_alchemy_payload
from mcp_dbs_bench.servers._subprocess import SubprocessMCPClient
from mcp_dbs_bench.servers._subprocess_adapter import (
    _SubprocessConnState,
    perform_discovery,
    perform_initialize,
    perform_tool_call,
)

_LOG = logging.getLogger(__name__)
_DECLARED_TOOL_COUNT = 4  # all_table_names, filter_table_names, schema_definitions, execute_query
_DEFAULT_BINARY_NAME = "mcp-alchemy"


def _translate_request(request: ToolCallRequest) -> tuple[str, dict[str, Any]]:
    """Map the harness's ``execute_query`` to upstream's ``execute_query``.

    Upstream uses the argument key ``query`` (not ``sql``).
    """
    if request.name != "execute_query":
        raise ValueError(
            f"mcp-alchemy subprocess wrapper only handles 'execute_query'; "
            f"got {request.name!r}"
        )
    return "execute_query", {"query": request.arguments["sql"]}


class McpAlchemyPlugin:
    """mcp-alchemy plugin — subprocess for measurement, in-process for tests.

    ``use_subprocess=True`` is the measured mode: it runs the real upstream
    binary. ``use_subprocess=False`` is a Python re-implementation of the
    upstream protocol so unit tests can run on hosts without ``mcp-alchemy``
    installed. The Track A scripts default to subprocess mode.

    Args:
        use_subprocess: When True, spawn the real ``mcp-alchemy`` upstream
            (measured mode). When False (default for tests), use
            the in-process fallback.
        db_url: SQLAlchemy URL passed to the upstream via ``DB_URL`` env var.
            Required when ``use_subprocess=True``.
        binary: Path to the ``mcp-alchemy`` binary. Defaults to whatever
            ``shutil.which`` finds.
        stderr_log_path: Optional path for capturing the child's stderr.

    Attributes:
        MEASUREMENT_REQUIRES_SUBPROCESS: ``True``. Declares that
            benchmark measurements must use the subprocess mode.
    """

    name = "mcp-alchemy"
    version = "0.1.0"
    MEASUREMENT_REQUIRES_SUBPROCESS: bool = True

    def __init__(
        self,
        *,
        use_subprocess: bool = False,
        db_url: str | None = None,
        binary: str | None = None,
        stderr_log_path: str | None = None,
    ) -> None:
        self._use_subprocess = use_subprocess
        self._db_url = db_url
        self._binary = binary or shutil.which(_DEFAULT_BINARY_NAME) or _DEFAULT_BINARY_NAME
        self._stderr_log_path = stderr_log_path
        self._fallback = McpAlchemyInProcessPlugin()

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            output_format=OutputFormat.VERTICAL_TEXT,
            connection_model=ConnectionModel.SQLALCHEMY,
            declared_tool_count=_DECLARED_TOOL_COUNT,
            supports_streaming=False,
            safety_mechanism="read_only",
            schema_depth="rich",
        )

    @asynccontextmanager
    async def connect(
        self, backend: DatabaseBackend, phase: PhaseContext,
    ) -> AsyncIterator[ConnectionHandle]:
        if not self._use_subprocess:
            async with self._fallback.connect(backend, phase) as handle:
                yield handle
            return

        if not self._db_url:
            raise ValueError(
                "mcp-alchemy subprocess wrapper requires a 'db_url' constructor argument"
            )
        from pathlib import Path

        client = SubprocessMCPClient(
            self._binary,
            args=[],
            env={"DB_URL": self._db_url},
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
            response_parser=parse_mcp_alchemy_payload,
        )


__all__ = ["McpAlchemyPlugin"]
