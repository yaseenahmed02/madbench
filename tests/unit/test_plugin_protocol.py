"""Tests for the MCPServerPlugin Protocol.

The protocol is structural and runtime-checkable. A class is a plugin if it
has the right attributes/methods; no inheritance required.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp_dbs_bench.core.plugin import MCPServerPlugin
from mcp_dbs_bench.core.types import (
    ConnectionHandle,
    ConnectionModel,
    OutputFormat,
    ServerCapabilities,
    ToolCallRequest,
    ToolCallResponse,
    ToolManifest,
)


class _ConformingPlugin:
    """A minimal correctly-shaped plugin used to test isinstance() checks."""

    name = "test-plugin"
    version = "0.0.1"

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            output_format=OutputFormat.JSON_FLAT_TABLE,
            connection_model=ConnectionModel.IMPLICIT_POOL,
            declared_tool_count=1,
            supports_streaming=False,
            safety_mechanism="none",
            schema_depth="basic",
        )

    @asynccontextmanager
    async def connect(self, backend, phase) -> AsyncIterator[ConnectionHandle]:
        yield ConnectionHandle(value=None)

    async def discover(self, conn, phase) -> ToolManifest:
        return ToolManifest(tools=[{"name": "query"}], raw_bytes=b'{"tools":[{"name":"query"}]}')

    async def call_tool(self, conn, request: ToolCallRequest, phase) -> ToolCallResponse:
        return ToolCallResponse(rows=[(1,)], raw_bytes=b"[[1]]", column_names=("x",))


class _MissingNamePlugin:
    """Missing the `name` attribute — should NOT pass isinstance."""

    version = "0.0.1"

    def capabilities(self) -> ServerCapabilities:
        raise NotImplementedError

    @asynccontextmanager
    async def connect(self, backend, phase) -> AsyncIterator[ConnectionHandle]:
        yield ConnectionHandle(value=None)

    async def discover(self, conn, phase) -> ToolManifest:
        raise NotImplementedError

    async def call_tool(self, conn, request, phase) -> ToolCallResponse:
        raise NotImplementedError


def test_conforming_plugin_passes_isinstance_check():
    plugin = _ConformingPlugin()
    assert isinstance(plugin, MCPServerPlugin)


def test_plugin_missing_name_fails_isinstance_check():
    plugin = _MissingNamePlugin()
    assert not isinstance(plugin, MCPServerPlugin)


def test_protocol_is_runtime_checkable():
    # @runtime_checkable adds an __instancecheck__ that does the duck-typing.
    assert hasattr(MCPServerPlugin, "__instancecheck__")


# ---------- plugin contract for t_db_ms ----------


def test_extract_t_db_ms_raises_when_plugin_omits_attribute():
    """A plugin that runs SERIALIZE without setting ``t_db_ms``
    in attrs has violated its contract with the harness. The harness must
    raise rather than silently return 0.0 and yield a misleading
    "completeness passes" trace with PTR=None.
    """
    from mcp_dbs_bench.core.phase import Phase
    from mcp_dbs_bench.core.phase_context import PhaseRecord
    from mcp_dbs_bench.harness.runner import _extract_t_db_ms_or_raise

    record = PhaseRecord(
        phase=Phase.SERIALIZE,
        start_ns=100,
        end_ns=200,
        attrs={"row_count": 3},  # NOTE: no t_db_ms
    )
    import pytest

    with pytest.raises(KeyError, match=r"t_db_ms"):
        _extract_t_db_ms_or_raise([record])


def test_extract_t_db_ms_returns_attribute_when_present():
    """Sanity: a properly instrumented plugin returns the recorded T_db."""
    from mcp_dbs_bench.core.phase import Phase
    from mcp_dbs_bench.core.phase_context import PhaseRecord
    from mcp_dbs_bench.harness.runner import _extract_t_db_ms_or_raise

    record = PhaseRecord(
        phase=Phase.SERIALIZE,
        start_ns=100,
        end_ns=200,
        attrs={"t_db_ms": 42.5},
    )
    assert _extract_t_db_ms_or_raise([record]) == 42.5
