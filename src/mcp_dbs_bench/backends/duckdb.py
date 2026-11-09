"""DuckDB backend adapter.

DuckDB is embedded: no container, no network, no auth. This makes it the
preferred smoke-test target and the most permissive
calibration cell — if Q6/SF1 against DuckDB cannot meet Property 1's 5%
threshold, nothing will.
"""

from __future__ import annotations

from time import monotonic_ns
from typing import Any

from mcp_dbs_bench.core.clock import ns_to_ms
from mcp_dbs_bench.core.exceptions import BackendDriverMissingError


class DuckDBBackend:
    """DuckDB adapter conforming to `core.backends.DatabaseBackend`.

    Args:
        database: Path to a DuckDB file, or ``":memory:"`` for an in-process
            ephemeral instance. Tests use ``:memory:``; the harness uses a
            file under ``runs/<run_id>/duckdb/`` so cold-start variants can
            drop the file and re-create.
    """

    name = "duckdb"

    def __init__(self, database: str = ":memory:") -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise BackendDriverMissingError(
                "duckdb", "uv sync (installs duckdb)"
            ) from exc
        self._duckdb_module = duckdb
        self._database = database
        self._conn = duckdb.connect(database)
        self._closed = False

    def dsn(self) -> str:
        """Return a DuckDB DSN. Embedded form: ``duckdb://<file-or-memory>``."""
        return f"duckdb://{self._database}"

    def execute(
        self, sql: str, *, parameters: tuple[Any, ...] | None = None
    ) -> tuple[list[tuple[Any, ...]], tuple[str, ...], float]:
        """Execute SQL and return ``(rows, columns, t_db_ms)``.

        Timing covers driver dispatch through row materialization. DuckDB's
        Python API is synchronous; T_db is the wall-clock between
        ``execute()`` start and ``fetchall()`` return.

        Raises:
            RuntimeError: If the backend has been closed.
        """
        if self._closed:
            raise RuntimeError("DuckDB backend is closed; create a new instance to query")
        start_ns = monotonic_ns()
        cursor = (
            self._conn.execute(sql, parameters) if parameters else self._conn.execute(sql)
        )
        rows = cursor.fetchall()
        end_ns = monotonic_ns()
        column_names = tuple(d[0] for d in cursor.description) if cursor.description else ()
        return rows, column_names, ns_to_ms(end_ns - start_ns)

    def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        if self._closed:
            return
        self._conn.close()
        self._closed = True
