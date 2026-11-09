"""PostgreSQL backend adapter — wraps psycopg3.

Native baseline path. The MCP servers (anthropic-pg, mcp-alchemy/PG,
dbhub/PG) connect via their own client libraries; this
adapter is used by the native CLI baseline runner and by the harness to
verify connectivity and seed TPC-H tables.
"""

from __future__ import annotations

from time import monotonic_ns
from typing import Any

from mcp_dbs_bench.core.clock import ns_to_ms
from mcp_dbs_bench.core.exceptions import BackendDriverMissingError


class PostgresBackend:
    """PostgreSQL adapter conforming to `core.backends.DatabaseBackend`.

    Args:
        dsn_url: ``postgresql://user:pass@host:port/dbname``.
    """

    name = "postgres"

    def __init__(self, dsn_url: str) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise BackendDriverMissingError(
                "postgres", "uv sync (installs psycopg)"
            ) from exc
        self._psycopg = psycopg
        self._dsn = dsn_url
        self._conn = psycopg.connect(dsn_url, autocommit=True)
        self._closed = False

    def dsn(self) -> str:
        return self._dsn

    def execute(
        self, sql: str, *, parameters: tuple[Any, ...] | None = None
    ) -> tuple[list[tuple[Any, ...]], tuple[str, ...], float]:
        """Execute SQL and return ``(rows, columns, t_db_ms)``.

        Raises:
            RuntimeError: If the backend has been closed.
        """
        if self._closed:
            raise RuntimeError("PostgreSQL backend is closed; create a new instance to query")
        start_ns = monotonic_ns()
        with self._conn.cursor() as cur:
            cur.execute(sql, parameters)
            rows = cur.fetchall() if cur.description else []
            column_names = (
                tuple(d.name for d in cur.description) if cur.description else ()
            )
        end_ns = monotonic_ns()
        return rows, column_names, ns_to_ms(end_ns - start_ns)

    def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        if self._closed:
            return
        self._conn.close()
        self._closed = True
