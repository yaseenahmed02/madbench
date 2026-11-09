"""MySQL backend adapter — wraps PyMySQL.

Native baseline path for the MySQL track-A cells.
"""

from __future__ import annotations

from time import monotonic_ns
from typing import Any
from urllib.parse import urlparse

from mcp_dbs_bench.core.clock import ns_to_ms
from mcp_dbs_bench.core.exceptions import BackendDriverMissingError


class MySQLBackend:
    """MySQL adapter conforming to `core.backends.DatabaseBackend`.

    Args:
        dsn_url: ``mysql://user:pass@host:port/dbname``. Parsed with
            ``urllib.parse``; query-string options are ignored.
    """

    name = "mysql"

    def __init__(self, dsn_url: str) -> None:
        try:
            import pymysql
        except ImportError as exc:
            raise BackendDriverMissingError(
                "mysql", "uv sync (installs pymysql)"
            ) from exc
        parsed = urlparse(dsn_url)
        if parsed.scheme != "mysql":
            raise ValueError(f"MySQL DSN scheme must be 'mysql'; got {parsed.scheme!r}")
        self._dsn = dsn_url
        self._conn = pymysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") or None,
            autocommit=True,
        )
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
            raise RuntimeError("MySQL backend is closed; create a new instance to query")
        start_ns = monotonic_ns()
        with self._conn.cursor() as cur:
            cur.execute(sql, parameters)
            rows = list(cur.fetchall()) if cur.description else []
            column_names = tuple(d[0] for d in cur.description) if cur.description else ()
        end_ns = monotonic_ns()
        return rows, column_names, ns_to_ms(end_ns - start_ns)

    def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        if self._closed:
            return
        self._conn.close()
        self._closed = True
