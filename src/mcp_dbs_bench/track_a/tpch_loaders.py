"""Per-scale-factor TPC-H loaders.

Postgres: ``COPY`` into a per-SF schema (``sf1``, ``sf3``, ``sf10``).
MySQL:    ``LOAD DATA LOCAL INFILE`` into per-SF databases (``tpch_sf1``...).
DuckDB:   one file per SF (``/tmp/madbench_duckdb_sf<n>.duckdb``).

Each loader is idempotent: it drops the target schema/database/file before
recreating, so repeated runs always start from clean ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp_dbs_bench.workloads.tpch.loader import (
    load_table_postgres, tpch_data_dir,
)
from mcp_dbs_bench.workloads.tpch.schema import SCHEMA_DDL, TPCH_TABLES

_LOG = logging.getLogger(__name__)

EXPECTED_LINEITEM_COUNTS: dict[int, int] = {
    1: 6_001_215,
    3: 17_996_609,
    10: 59_986_052,
}


def load_postgres_scale_factor(
    *, host: str, port: int, user: str, password: str, database: str,
    scale_factor: int, repo_root: Path,
) -> dict[str, int]:
    """Drop+create schema ``sf<n>`` and load 8 TPC-H tables via COPY.

    Returns a mapping table -> row count.
    """
    import psycopg
    data_dir = tpch_data_dir(scale_factor, repo_root=repo_root)
    schema = f"sf{scale_factor}"
    dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    counts: dict[str, int] = {}
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(f"SET search_path TO {schema}")
        for ddl in SCHEMA_DDL["postgres"]:
            with conn.cursor() as cur:
                cur.execute(f"SET search_path TO {schema}")
                cur.execute(ddl)
        for table in TPCH_TABLES:
            tbl = data_dir / f"{table}.tbl"
            _LOG.info("postgres SF%d: loading %s from %s", scale_factor, table, tbl)
            with conn.cursor() as cur:
                cur.execute(f"SET search_path TO {schema}")
            counts[table] = load_table_postgres(conn, table, tbl)
    return counts


def load_mysql_scale_factor(
    *, host: str, port: int, user: str, password: str,
    scale_factor: int, repo_root: Path,
) -> dict[str, int]:
    """Drop+create database ``tpch_sf<n>`` and bulk-load via LOAD DATA LOCAL INFILE.

    Returns a mapping table -> row count.
    """
    import pymysql
    database = f"tpch_sf{scale_factor}"
    data_dir = tpch_data_dir(scale_factor, repo_root=repo_root)
    server_conn = pymysql.connect(
        host=host, port=port, user=user, password=password,
        local_infile=True, autocommit=True,
    )
    try:
        with server_conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {database}")
            cur.execute(f"CREATE DATABASE {database}")
    finally:
        server_conn.close()
    counts: dict[str, int] = {}
    db_conn = pymysql.connect(
        host=host, port=port, user=user, password=password, database=database,
        local_infile=True, autocommit=True,
    )
    try:
        with db_conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for ddl in SCHEMA_DDL["mysql"]:
                cur.execute(ddl)
            for table in TPCH_TABLES:
                tbl_path = data_dir / f"{table}.tbl"
                _LOG.info(
                    "mysql SF%d: loading %s from %s", scale_factor, table, tbl_path,
                )
                # dbgen lines end with `|\n`. We use LINES TERMINATED BY '\n'
                # and append an unused dummy column via "@dummy" so the trailing
                # pipe is absorbed without polluting the real columns.
                column_names = _MYSQL_LOAD_COLUMNS[table]
                cur.execute(
                    f"LOAD DATA LOCAL INFILE %s INTO TABLE {table} "
                    f"FIELDS TERMINATED BY '|' LINES TERMINATED BY '\\n' "
                    f"({column_names}, @dummy)",
                    (str(tbl_path),),
                )
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                row = cur.fetchone()
                counts[table] = int(row[0]) if row else 0
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
    finally:
        db_conn.close()
    return counts


_MYSQL_LOAD_COLUMNS: dict[str, str] = {
    "region":   "r_regionkey, r_name, r_comment",
    "nation":   "n_nationkey, n_name, n_regionkey, n_comment",
    "part":     ("p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, "
                 "p_container, p_retailprice, p_comment"),
    "supplier": ("s_suppkey, s_name, s_address, s_nationkey, s_phone, "
                 "s_acctbal, s_comment"),
    "partsupp": "ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment",
    "customer": ("c_custkey, c_name, c_address, c_nationkey, c_phone, "
                 "c_acctbal, c_mktsegment, c_comment"),
    "orders":   ("o_orderkey, o_custkey, o_orderstatus, o_totalprice, "
                 "o_orderdate, o_orderpriority, o_clerk, o_shippriority, "
                 "o_comment"),
    "lineitem": (
        "l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, "
        "l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, "
        "l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, "
        "l_shipmode, l_comment"
    ),
}


def load_duckdb_scale_factor(
    *, db_path: Path, scale_factor: int, repo_root: Path,
) -> dict[str, int]:
    """Recreate the DuckDB file and load all 8 TPC-H tables."""
    import duckdb
    if db_path.exists():
        db_path.unlink()
    data_dir = tpch_data_dir(scale_factor, repo_root=repo_root)
    counts: dict[str, int] = {}
    conn = duckdb.connect(str(db_path))
    try:
        for ddl in SCHEMA_DDL["duckdb"]:
            conn.execute(ddl)
        for table in TPCH_TABLES:
            tbl_path = data_dir / f"{table}.tbl"
            _LOG.info(
                "duckdb SF%d: loading %s from %s", scale_factor, table, tbl_path,
            )
            sql = (
                f"COPY {table} FROM '{tbl_path}' "
                "(DELIMITER '|', HEADER FALSE, NULL '', AUTO_DETECT FALSE)"
            )
            conn.execute(sql)
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = int(row[0]) if row else 0
    finally:
        conn.close()
    return counts


def verify_lineitem_count(counts: dict[str, int], scale_factor: int) -> None:
    """Raise if loaded lineitem count does not match the spec.

    The TPC-H SF1/3/10 line-item cardinalities are deterministic in dbgen.
    """
    actual = counts.get("lineitem", -1)
    expected = EXPECTED_LINEITEM_COUNTS[scale_factor]
    if actual != expected:
        raise RuntimeError(
            f"SF{scale_factor} lineitem count mismatch: "
            f"expected {expected}, got {actual}"
        )


__all__ = [
    "EXPECTED_LINEITEM_COUNTS",
    "load_duckdb_scale_factor",
    "load_mysql_scale_factor",
    "load_postgres_scale_factor",
    "verify_lineitem_count",
]
