"""DatabaseBackend Protocol — contract for DBMS adapters.

Each adapter encapsulates everything backend-specific behind a uniform
interface so the harness and the MCP-server adapters can be written without
referring to backend internals.

Native CLI baselines (psql, mysql, duckdb CLI) are exercised through the
same protocol so $T_{db}$ and $B_{native}$ are sourced consistently.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DatabaseBackend(Protocol):
    """Contract for a DBMS adapter (PostgreSQL, MySQL, DuckDB).

    Implementations are produced by the plugin registry from the
    `mcp_dbs_bench.backends` entry-point group, identically to MCP server
    plugins. Built-in adapters live in `mcp_dbs_bench.backends`.
    """

    name: str

    def dsn(self) -> str:
        """Return a DSN string the MCP server can use to connect.

        For embedded engines (DuckDB), this is a file path or `:memory:`.
        For client-server engines, it is the canonical URL form for the
        backend's driver.
        """
        ...

    def execute(
        self, sql: str, *, parameters: tuple[Any, ...] | None = None
    ) -> tuple[list[tuple[Any, ...]], tuple[str, ...], float]:
        """Execute SQL natively (CLI-baseline path).

        Returns:
            ``(rows, column_names, t_db_ms)`` — rows as a list of tuples,
            column names from the result schema, and the wall-clock
            milliseconds spent inside the driver from dispatch to
            iterator-exhausted.
        """
        ...

    def close(self) -> None:
        """Release any held resources (connection pools, file handles).

        Idempotent: safe to call multiple times.
        """
        ...
