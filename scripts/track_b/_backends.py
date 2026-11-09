"""Shared backend helpers for Track B scripts.

Encapsulates the connection + execute contract that gold-capture and the
sweep runner both need: a small ``BackendClient`` that returns
``(columns, rows)`` for a SQL string and applies the right ``search_path``
/ database / DuckDB file per scale factor.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

_LOG = logging.getLogger(__name__)

SUPPORTED_BACKENDS: tuple[str, ...] = ("postgres", "mysql", "duckdb")
QUERY_TIMEOUT_SECONDS: int = 120

_PG_DEFAULT = {
    "host": "localhost", "port": 55432, "user": "bench",
    "password": "bench", "database": "tpch",
}
_MYSQL_DEFAULT = {
    "host": "localhost", "port": 53306, "user": "bench", "password": "bench",
}
_DUCKDB_FILE_TEMPLATE: str = "/tmp/madbench_duckdb_sf{sf}.duckdb"


@dataclass(frozen=True, slots=True)
class BackendClient:
    """Executes SQL against one backend + scale factor. Owns its connection.

    Use as a context manager to ensure connection cleanup.
    """

    backend: str
    scale_factor: int
    _conn: Any
    _params: dict[str, Any]

    def execute(self, sql: str) -> tuple[list[str], list[list[Any]]]:
        """Run ``sql`` and return ``(columns, rows)``.

        Rows are converted to Python-native lists so they are JSON-serialisable.
        """
        if self.backend == "postgres":
            return _execute_postgres(self._conn, sql)
        if self.backend == "mysql":
            return _execute_mysql(self._conn, sql)
        if self.backend == "duckdb":
            return _execute_duckdb(self._conn, sql)
        raise ValueError(f"unsupported backend: {self.backend!r}")

    def dsn(self) -> str:
        """Return the DSN in the shape the MCP-alchemy plugin consumes."""
        if self.backend == "postgres":
            search_path = f"sf{self.scale_factor}"
            base = self._params["dsn"]
            return f"{base}?options=-c%20search_path%3D{search_path}"
        if self.backend == "mysql":
            return self._params["dsn"]
        if self.backend == "duckdb":
            return f"duckdb:///{self._params['file']}"
        raise ValueError(f"unsupported backend: {self.backend!r}")


@contextmanager
def connect(backend: str, scale_factor: int) -> Iterator[BackendClient]:
    """Open a connection for ``(backend, scale_factor)`` and yield a client."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"backend must be one of {SUPPORTED_BACKENDS!r}; got {backend!r}"
        )
    if backend == "postgres":
        yield from _connect_postgres(scale_factor)
        return
    if backend == "mysql":
        yield from _connect_mysql(scale_factor)
        return
    if backend == "duckdb":
        yield from _connect_duckdb(scale_factor)
        return


def _connect_postgres(scale_factor: int) -> Iterator[BackendClient]:
    import psycopg
    schema = f"sf{scale_factor}"
    conn = psycopg.connect(
        f"postgresql://{_PG_DEFAULT['user']}:{_PG_DEFAULT['password']}"
        f"@{_PG_DEFAULT['host']}:{_PG_DEFAULT['port']}/{_PG_DEFAULT['database']}",
        options=f"-c search_path={schema}",
        connect_timeout=10,
    )
    conn.autocommit = True
    dsn = (
        f"postgresql+psycopg://{_PG_DEFAULT['user']}:{_PG_DEFAULT['password']}"
        f"@{_PG_DEFAULT['host']}:{_PG_DEFAULT['port']}/{_PG_DEFAULT['database']}"
    )
    client = BackendClient(
        backend="postgres", scale_factor=scale_factor,
        _conn=conn, _params={"dsn": dsn, "schema": schema},
    )
    try:
        yield client
    finally:
        conn.close()


def _connect_mysql(scale_factor: int) -> Iterator[BackendClient]:
    import pymysql
    database = f"tpch_sf{scale_factor}"
    conn = pymysql.connect(
        host=_MYSQL_DEFAULT["host"], port=_MYSQL_DEFAULT["port"],
        user=_MYSQL_DEFAULT["user"], password=_MYSQL_DEFAULT["password"],
        database=database, autocommit=True, connect_timeout=10,
        read_timeout=QUERY_TIMEOUT_SECONDS,
    )
    dsn = (
        f"mysql+pymysql://{_MYSQL_DEFAULT['user']}:{_MYSQL_DEFAULT['password']}"
        f"@{_MYSQL_DEFAULT['host']}:{_MYSQL_DEFAULT['port']}/{database}"
    )
    client = BackendClient(
        backend="mysql", scale_factor=scale_factor,
        _conn=conn, _params={"dsn": dsn, "database": database},
    )
    try:
        yield client
    finally:
        conn.close()


def _connect_duckdb(scale_factor: int) -> Iterator[BackendClient]:
    import duckdb
    file = _DUCKDB_FILE_TEMPLATE.format(sf=scale_factor)
    if not Path(file).exists():
        raise FileNotFoundError(
            f"DuckDB SF{scale_factor} file missing at {file}. "
            f"Run: python scripts/load_tpch.py --backends duckdb --sfs {scale_factor}"
        )
    conn = duckdb.connect(file, read_only=True)
    client = BackendClient(
        backend="duckdb", scale_factor=scale_factor,
        _conn=conn, _params={"file": file},
    )
    try:
        yield client
    finally:
        conn.close()


def _execute_postgres(conn: Any, sql: str) -> tuple[list[str], list[list[Any]]]:
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout TO {QUERY_TIMEOUT_SECONDS * 1000}")
        cur.execute(sql)
        cols = [str(d.name) for d in (cur.description or ())]
        rows = cur.fetchall() if cur.description else []
    return cols, [list(r) for r in rows]


def _execute_mysql(conn: Any, sql: str) -> tuple[list[str], list[list[Any]]]:
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [str(d[0]) for d in (cur.description or ())]
        rows = cur.fetchall() if cur.description else []
    return cols, [list(r) for r in rows]


def _execute_duckdb(conn: Any, sql: str) -> tuple[list[str], list[list[Any]]]:
    result = conn.execute(sql)
    cols = [str(c[0]) for c in (result.description or ())]
    rows = result.fetchall()
    return cols, [list(r) for r in rows]


__all__ = [
    "BackendClient", "QUERY_TIMEOUT_SECONDS", "SUPPORTED_BACKENDS", "connect",
]
