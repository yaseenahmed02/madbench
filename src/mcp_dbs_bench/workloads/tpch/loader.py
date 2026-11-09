"""Bulk-load TPC-H .tbl files into a backend.

The dbgen-format file is pipe-delimited with a trailing pipe on every row.
Each backend has a fast bulk loader:
    - PostgreSQL: ``COPY ... FROM STDIN``
    - MySQL: ``LOAD DATA LOCAL INFILE`` (requires local_infile=ON)
    - DuckDB: ``COPY <tbl> FROM '<path>' (DELIMITER '|')``

Callers are responsible for running ``scripts/build_tpch.sh <sf>`` first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp_dbs_bench.core.exceptions import ConfigurationError
from mcp_dbs_bench.workloads.tpch.schema import SCHEMA_DDL, TPCH_TABLES


def tpch_data_dir(scale_factor: int, *, repo_root: Path | None = None) -> Path:
    """Return the on-disk path holding TPC-H .tbl files for ``scale_factor``.

    Resolution order:
        1. ``$TPCH_DATA_DIR`` env var (if set; treated as a base dir,
           per-SF subdirs ``sf<N>`` are appended).
        2. ``<repo_root>/data/tpch/sf<N>`` (default for in-repo data).

    The env var path is preferred for HPC/CI setups where the raw .tbl
    files (10+ GB at SF10) live outside the repo to keep NFS home quotas
    small.
    """
    override = os.environ.get("TPCH_DATA_DIR")
    if override:
        return Path(override) / f"sf{scale_factor}"
    root = repo_root or Path(__file__).resolve().parents[4]
    return root / "data" / "tpch" / f"sf{scale_factor}"


def assert_tpch_data_present(scale_factor: int, *, repo_root: Path | None = None) -> Path:
    """Verify SF data has been generated. Returns the data directory on success."""
    data_dir = tpch_data_dir(scale_factor, repo_root=repo_root)
    if not data_dir.is_dir():
        raise ConfigurationError(
            f"TPC-H SF{scale_factor} data directory not found at {data_dir}. "
            f"Run: scripts/build_tpch.sh {scale_factor}"
        )
    for table in TPCH_TABLES:
        tbl_path = data_dir / f"{table}.tbl"
        if not tbl_path.is_file():
            raise ConfigurationError(
                f"Missing TPC-H file {tbl_path}. Re-run scripts/build_tpch.sh {scale_factor}."
            )
    return data_dir


def reset_schema(backend: Any, backend_name: str) -> None:
    """Drop and recreate the eight TPC-H tables on ``backend``.

    Args:
        backend: A DatabaseBackend instance.
        backend_name: ``postgres`` | ``mysql`` | ``duckdb``.
    """
    if backend_name not in SCHEMA_DDL:
        raise ConfigurationError(f"Unknown backend {backend_name!r} for TPC-H schema reset")
    _drop_all(backend, backend_name)
    for stmt in SCHEMA_DDL[backend_name]:
        backend.execute(stmt)


def _drop_all(backend: Any, backend_name: str) -> None:
    if backend_name == "mysql":
        backend.execute("SET FOREIGN_KEY_CHECKS=0")
        for tbl in reversed(TPCH_TABLES):
            backend.execute(f"DROP TABLE IF EXISTS {tbl}")
        backend.execute("SET FOREIGN_KEY_CHECKS=1")
        return
    for tbl in reversed(TPCH_TABLES):
        suffix = " CASCADE" if backend_name == "postgres" else ""
        backend.execute(f"DROP TABLE IF EXISTS {tbl}{suffix}")


def load_table_duckdb(backend: Any, table: str, tbl_path: Path) -> int:
    """Load one .tbl file into DuckDB via COPY FROM."""
    sql = (
        f"COPY {table} FROM '{tbl_path}' "
        "(DELIMITER '|', HEADER FALSE, NULL '', AUTO_DETECT FALSE)"
    )
    rows, _, _ = backend.execute(sql)
    if rows and rows[0]:
        return int(rows[0][0])
    count_rows, _, _ = backend.execute(f"SELECT COUNT(*) FROM {table}")
    return int(count_rows[0][0])


def load_table_postgres(conn: Any, table: str, tbl_path: Path) -> int:
    """Stream one .tbl file into Postgres via COPY ... FROM STDIN.

    Args:
        conn: Underlying psycopg.Connection (not the adapter wrapper).
        table: Table name.
        tbl_path: Path to the .tbl file.

    Returns:
        Number of rows loaded.
    """
    copy_sql = (
        f"COPY {table} FROM STDIN WITH (FORMAT csv, DELIMITER '|', NULL '')"
    )
    with conn.cursor() as cur, tbl_path.open("rb") as fh:
        with cur.copy(copy_sql) as copy:
            for raw_line in fh:
                # dbgen lines end with `|\n` — strip the trailing pipe.
                line = raw_line.rstrip(b"\n").rstrip(b"|") + b"\n"
                copy.write(line)
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        result = cur.fetchone()
    return int(result[0]) if result else 0
