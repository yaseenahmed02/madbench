"""Tests for ``validate_plugin``.

The validator is exposed at the top-level ``mcp_dbs_bench`` namespace and
is the documented self-check third-party plugin authors run before
publishing. These tests cover its four checks.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator


from mcp_dbs_bench import (
    ConnectionHandle,
    ConnectionModel,
    OutputFormat,
    ServerCapabilities,
    ToolCallRequest,
    ToolCallResponse,
    ToolManifest,
    ValidationReport,
    validate_plugin,
)


class _ConformingPlugin:
    """A correctly-shaped plugin used to exercise the happy path."""

    name = "conforming"
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
        return ToolManifest()

    async def call_tool(
        self, conn, request: ToolCallRequest, phase
    ) -> ToolCallResponse:
        return ToolCallResponse(rows=[], raw_bytes=b"", column_names=())


class _MissingNamePlugin:
    """Plugin missing the ``name`` attribute — fails check 3."""

    version = "0.0.1"

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            output_format=OutputFormat.JSON_FLAT_TABLE,
            connection_model=ConnectionModel.IMPLICIT_POOL,
            declared_tool_count=1, supports_streaming=False,
            safety_mechanism="none", schema_depth="basic",
        )

    @asynccontextmanager
    async def connect(self, backend, phase):
        yield ConnectionHandle(value=None)

    async def discover(self, conn, phase):
        return ToolManifest()

    async def call_tool(self, conn, request, phase):
        return ToolCallResponse(rows=[], raw_bytes=b"", column_names=())


class _BrokenCapabilitiesPlugin:
    """``capabilities()`` returns a non-ServerCapabilities object."""

    name = "broken-caps"
    version = "0.0.1"

    def capabilities(self):
        return {"not": "the right type"}

    @asynccontextmanager
    async def connect(self, backend, phase):
        yield ConnectionHandle(value=None)

    async def discover(self, conn, phase):
        return ToolManifest()

    async def call_tool(self, conn, request, phase):
        return ToolCallResponse(rows=[], raw_bytes=b"", column_names=())


class _RaisingConstructorPlugin:
    """Construction always raises — fails check 1."""

    name = "raises"
    version = "0.0.1"

    def __init__(self) -> None:
        raise RuntimeError("intentional construction failure")

    def capabilities(self):
        raise NotImplementedError

    @asynccontextmanager
    async def connect(self, backend, phase):
        yield ConnectionHandle(value=None)

    async def discover(self, conn, phase):
        return ToolManifest()

    async def call_tool(self, conn, request, phase):
        return ToolCallResponse(rows=[], raw_bytes=b"", column_names=())


def test_validate_returns_ok_for_conforming_plugin() -> None:
    report = validate_plugin(_ConformingPlugin)
    assert isinstance(report, ValidationReport)
    assert report.ok is True
    assert report.issues == []
    assert report.plugin_name == "conforming"


def test_validate_records_missing_name() -> None:
    report = validate_plugin(_MissingNamePlugin)
    assert report.ok is False
    assert any("'name'" in issue for issue in report.issues)


def test_validate_records_bad_capabilities_return_type() -> None:
    report = validate_plugin(_BrokenCapabilitiesPlugin)
    assert report.ok is False
    assert any("ServerCapabilities" in issue for issue in report.issues)


def test_validate_records_construction_failure() -> None:
    report = validate_plugin(_RaisingConstructorPlugin)
    assert report.ok is False
    assert any("construction failed" in issue.lower() for issue in report.issues)


def test_public_surface_re_exports_sdk_types() -> None:
    """The SDK names third-party authors need are in ``mcp_dbs_bench``."""
    import mcp_dbs_bench

    for name in [
        "MCPServerPlugin", "Phase", "PhaseContext", "PhaseRecord",
        "ConnectionHandle", "ConnectionModel", "OutputFormat",
        "ServerCapabilities", "ToolCallRequest", "ToolCallResponse",
        "ToolManifest", "ValidationReport", "validate_plugin",
    ]:
        assert hasattr(mcp_dbs_bench, name), (
            f"missing public symbol {name!r} from mcp_dbs_bench"
        )
        assert name in mcp_dbs_bench.__all__, (
            f"{name!r} not in mcp_dbs_bench.__all__"
        )
