"""Tests for the DatabaseBackend Protocol."""

from __future__ import annotations

from mcp_dbs_bench.core.backends import DatabaseBackend


class _MinimalBackend:
    name = "fake"

    def dsn(self) -> str:
        return "fake://localhost"

    def execute(self, sql: str) -> tuple[list[tuple], tuple[str, ...], float]:
        return ([(1,)], ("x",), 0.5)

    def close(self) -> None:
        pass


class _MissingExecute:
    name = "fake"

    def dsn(self) -> str:
        return "fake://localhost"

    def close(self) -> None:
        pass


def test_minimal_backend_passes_protocol_check():
    assert isinstance(_MinimalBackend(), DatabaseBackend)


def test_backend_missing_execute_fails_protocol_check():
    assert not isinstance(_MissingExecute(), DatabaseBackend)
