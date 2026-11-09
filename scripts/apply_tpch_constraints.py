#!/usr/bin/env python
"""Apply TPC-H PK + FK constraints + standard FK/date indexes.

Why this script exists
----------------------
The TPC-H kit (gregrahn/tpch-kit) ships ``dss.ddl`` (table definitions only)
and ``dss.ri`` (PK + FK constraints). Our original load pipeline only applied
``dss.ddl``, leaving the databases with NO primary keys, NO foreign keys, and
NO indexes. This script applies the equivalent of ``dss.ri`` (adapted from
DB2 syntax to PG/MySQL) plus the canonical FK index set used in published
TPC-H methodology (Leis 2015, Dreseler 2020, Boncz 2013).

What it does
------------
For each (backend, scale_factor) pair:
  1. PRIMARY KEY constraints on all 8 tables
  2. FOREIGN KEY constraints (referencing PKs)
  3. Indexes on FK columns not covered by PKs (PG/MySQL do not auto-index FKs)
  4. Date-locality indexes on l_shipdate and o_orderdate (common in published
     TPC-H runs; matches the tpch-citus reference DDL)

Layouts handled
---------------
- PostgreSQL mcp-alchemy layout: db=``tpch``, schemas=``sf1, sf3, sf10``
- PostgreSQL DBHub layout: dbs=``tpch_sf1, tpch_sf3, tpch_sf10``, schema=``public``
- MySQL: dbs=``tpch_sf1, tpch_sf3, tpch_sf10``

Idempotent: catches "already exists" errors and continues.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Iterable

import psycopg
import pymysql

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
_LOG = logging.getLogger("apply_tpch_constraints")


TPCH_TABLES = (
    "region", "nation", "part", "supplier",
    "partsupp", "customer", "orders", "lineitem",
)


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """One PK / FK / INDEX statement, parameterized on schema name."""

    label: str
    sql_template: str

    def render(self, schema: str) -> str:
        return self.sql_template.format(s=schema)


PG_CONSTRAINTS: tuple[ConstraintSpec, ...] = (
    # Primary keys (auto-creates unique btree index for each)
    ConstraintSpec("pk_region",   "ALTER TABLE {s}.region   ADD PRIMARY KEY (r_regionkey)"),
    ConstraintSpec("pk_nation",   "ALTER TABLE {s}.nation   ADD PRIMARY KEY (n_nationkey)"),
    ConstraintSpec("pk_part",     "ALTER TABLE {s}.part     ADD PRIMARY KEY (p_partkey)"),
    ConstraintSpec("pk_supplier", "ALTER TABLE {s}.supplier ADD PRIMARY KEY (s_suppkey)"),
    ConstraintSpec("pk_partsupp", "ALTER TABLE {s}.partsupp ADD PRIMARY KEY (ps_partkey, ps_suppkey)"),
    ConstraintSpec("pk_customer", "ALTER TABLE {s}.customer ADD PRIMARY KEY (c_custkey)"),
    ConstraintSpec("pk_orders",   "ALTER TABLE {s}.orders   ADD PRIMARY KEY (o_orderkey)"),
    ConstraintSpec("pk_lineitem", "ALTER TABLE {s}.lineitem ADD PRIMARY KEY (l_orderkey, l_linenumber)"),

    # Foreign keys (PG does NOT auto-index FK columns)
    ConstraintSpec("fk_nation_region",      "ALTER TABLE {s}.nation   ADD FOREIGN KEY (n_regionkey)         REFERENCES {s}.region"),
    ConstraintSpec("fk_supplier_nation",    "ALTER TABLE {s}.supplier ADD FOREIGN KEY (s_nationkey)         REFERENCES {s}.nation"),
    ConstraintSpec("fk_partsupp_part",      "ALTER TABLE {s}.partsupp ADD FOREIGN KEY (ps_partkey)          REFERENCES {s}.part"),
    ConstraintSpec("fk_partsupp_supplier",  "ALTER TABLE {s}.partsupp ADD FOREIGN KEY (ps_suppkey)          REFERENCES {s}.supplier"),
    ConstraintSpec("fk_customer_nation",    "ALTER TABLE {s}.customer ADD FOREIGN KEY (c_nationkey)         REFERENCES {s}.nation"),
    ConstraintSpec("fk_orders_customer",    "ALTER TABLE {s}.orders   ADD FOREIGN KEY (o_custkey)           REFERENCES {s}.customer"),
    ConstraintSpec("fk_lineitem_orders",    "ALTER TABLE {s}.lineitem ADD FOREIGN KEY (l_orderkey)          REFERENCES {s}.orders"),
    ConstraintSpec("fk_lineitem_partsupp",  "ALTER TABLE {s}.lineitem ADD FOREIGN KEY (l_partkey, l_suppkey) REFERENCES {s}.partsupp"),

    # FK indexes (canonical set per Leis 2015, tpch-citus reference DDL)
    ConstraintSpec("idx_nation_regionkey",    "CREATE INDEX ON {s}.nation   (n_regionkey)"),
    ConstraintSpec("idx_supplier_nationkey",  "CREATE INDEX ON {s}.supplier (s_nationkey)"),
    ConstraintSpec("idx_partsupp_suppkey",    "CREATE INDEX ON {s}.partsupp (ps_suppkey)"),
    ConstraintSpec("idx_customer_nationkey",  "CREATE INDEX ON {s}.customer (c_nationkey)"),
    ConstraintSpec("idx_orders_custkey",      "CREATE INDEX ON {s}.orders   (o_custkey)"),
    ConstraintSpec("idx_lineitem_part_supp",  "CREATE INDEX ON {s}.lineitem (l_partkey, l_suppkey)"),

    # Date-locality indexes (used by Q1, Q3, Q6, Q21; common in published runs)
    ConstraintSpec("idx_lineitem_shipdate",  "CREATE INDEX ON {s}.lineitem (l_shipdate)"),
    ConstraintSpec("idx_orders_orderdate",   "CREATE INDEX ON {s}.orders   (o_orderdate)"),
)


MYSQL_CONSTRAINTS: tuple[ConstraintSpec, ...] = (
    # MySQL: each schema is a separate database, so '{s}' is unused; we just
    # CONNECT to the right db and run unqualified SQL.
    ConstraintSpec("pk_region",   "ALTER TABLE region   ADD PRIMARY KEY (r_regionkey)"),
    ConstraintSpec("pk_nation",   "ALTER TABLE nation   ADD PRIMARY KEY (n_nationkey)"),
    ConstraintSpec("pk_part",     "ALTER TABLE part     ADD PRIMARY KEY (p_partkey)"),
    ConstraintSpec("pk_supplier", "ALTER TABLE supplier ADD PRIMARY KEY (s_suppkey)"),
    ConstraintSpec("pk_partsupp", "ALTER TABLE partsupp ADD PRIMARY KEY (ps_partkey, ps_suppkey)"),
    ConstraintSpec("pk_customer", "ALTER TABLE customer ADD PRIMARY KEY (c_custkey)"),
    ConstraintSpec("pk_orders",   "ALTER TABLE orders   ADD PRIMARY KEY (o_orderkey)"),
    ConstraintSpec("pk_lineitem", "ALTER TABLE lineitem ADD PRIMARY KEY (l_orderkey, l_linenumber)"),

    # MySQL requires explicit referenced columns when there are no UNIQUE on parent.
    # PK covers it; we can rely on shorthand (references parent without col list).
    ConstraintSpec("fk_nation_region",      "ALTER TABLE nation   ADD FOREIGN KEY (n_regionkey)          REFERENCES region   (r_regionkey)"),
    ConstraintSpec("fk_supplier_nation",    "ALTER TABLE supplier ADD FOREIGN KEY (s_nationkey)          REFERENCES nation   (n_nationkey)"),
    ConstraintSpec("fk_partsupp_part",      "ALTER TABLE partsupp ADD FOREIGN KEY (ps_partkey)           REFERENCES part     (p_partkey)"),
    ConstraintSpec("fk_partsupp_supplier",  "ALTER TABLE partsupp ADD FOREIGN KEY (ps_suppkey)           REFERENCES supplier (s_suppkey)"),
    ConstraintSpec("fk_customer_nation",    "ALTER TABLE customer ADD FOREIGN KEY (c_nationkey)          REFERENCES nation   (n_nationkey)"),
    ConstraintSpec("fk_orders_customer",    "ALTER TABLE orders   ADD FOREIGN KEY (o_custkey)            REFERENCES customer (c_custkey)"),
    ConstraintSpec("fk_lineitem_orders",    "ALTER TABLE lineitem ADD FOREIGN KEY (l_orderkey)           REFERENCES orders   (o_orderkey)"),
    ConstraintSpec("fk_lineitem_partsupp",  "ALTER TABLE lineitem ADD FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey)"),

    # MySQL auto-creates an index on FK columns, so we only need date indexes.
    ConstraintSpec("idx_lineitem_shipdate",  "CREATE INDEX idx_lineitem_shipdate  ON lineitem (l_shipdate)"),
    ConstraintSpec("idx_orders_orderdate",   "CREATE INDEX idx_orders_orderdate   ON orders   (o_orderdate)"),
)


def _is_already_exists(exc: Exception) -> bool:
    """Heuristic: True if the error means the constraint/index already exists."""
    msg = str(exc).lower()
    return any(s in msg for s in (
        "already exists",
        "duplicate key name",
        "multiple primary key",
        "duplicate column name",
        "duplicate foreign key",
    ))


def apply_pg(*, host: str, port: int, user: str, password: str,
             database: str, schemas: Iterable[str]) -> None:
    """Apply constraints to PostgreSQL across the given schemas."""
    dsn = f"host={host} port={port} user={user} password={password} dbname={database}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        for schema in schemas:
            _LOG.info("PG db=%s schema=%s: applying %d constraints",
                      database, schema, len(PG_CONSTRAINTS))
            _apply_to_pg_schema(conn, schema)


def _apply_to_pg_schema(conn: "psycopg.Connection", schema: str) -> None:
    applied, skipped = 0, 0
    for spec in PG_CONSTRAINTS:
        sql = spec.render(schema)
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            applied += 1
            _LOG.info("  [OK]   %s.%s", schema, spec.label)
        except psycopg.errors.DuplicateTable:
            skipped += 1
            _LOG.info("  [SKIP] %s.%s (already exists)", schema, spec.label)
        except psycopg.errors.DuplicateObject:
            skipped += 1
            _LOG.info("  [SKIP] %s.%s (already exists)", schema, spec.label)
        except psycopg.errors.InvalidTableDefinition:
            skipped += 1
            _LOG.info("  [SKIP] %s.%s (multiple PK)", schema, spec.label)
        except Exception as exc:  # noqa: BLE001
            if _is_already_exists(exc):
                skipped += 1
                _LOG.info("  [SKIP] %s.%s (already exists)", schema, spec.label)
            else:
                _LOG.error("  [FAIL] %s.%s: %s", schema, spec.label, exc)
                raise
    _LOG.info("PG schema=%s done: %d applied, %d skipped", schema, applied, skipped)


def apply_mysql(*, host: str, port: int, user: str, password: str,
                databases: Iterable[str]) -> None:
    """Apply constraints to MySQL across the given databases."""
    for db in databases:
        _LOG.info("MySQL db=%s: applying %d constraints", db, len(MYSQL_CONSTRAINTS))
        conn = pymysql.connect(host=host, port=port, user=user,
                               password=password, database=db, autocommit=True)
        try:
            _apply_to_mysql_db(conn, db)
        finally:
            conn.close()


def _apply_to_mysql_db(conn: "pymysql.Connection", db: str) -> None:
    applied, skipped = 0, 0
    for spec in MYSQL_CONSTRAINTS:
        try:
            with conn.cursor() as cur:
                cur.execute(spec.sql_template)
            applied += 1
            _LOG.info("  [OK]   %s/%s", db, spec.label)
        except Exception as exc:  # noqa: BLE001
            if _is_already_exists(exc):
                skipped += 1
                _LOG.info("  [SKIP] %s/%s (already exists)", db, spec.label)
            else:
                _LOG.error("  [FAIL] %s/%s: %s", db, spec.label, exc)
                raise
    _LOG.info("MySQL db=%s done: %d applied, %d skipped", db, applied, skipped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="postgres,mysql",
                        help="Comma-separated: postgres,mysql")
    parser.add_argument("--sfs", default="1,3,10",
                        help="Comma-separated scale factors")
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=55432)
    parser.add_argument("--pg-user", default="bench")
    parser.add_argument("--pg-password", default="bench")
    parser.add_argument("--pg-database", default="tpch",
                        help="PG database name (mcp-alchemy layout: per-SF schemas inside)")
    parser.add_argument("--pg-dbhub-databases", default="tpch_sf1,tpch_sf3,tpch_sf10",
                        help="Comma-separated DBHub per-SF databases; pass empty to skip.")
    parser.add_argument("--mysql-host", default="localhost")
    parser.add_argument("--mysql-port", type=int, default=53306)
    parser.add_argument("--mysql-user", default="bench")
    parser.add_argument("--mysql-password", default="bench")
    args = parser.parse_args()

    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    sfs = tuple(int(s.strip()) for s in args.sfs.split(",") if s.strip())

    if "postgres" in backends:
        # mcp-alchemy layout: one `tpch` db, per-SF schemas
        schemas = tuple(f"sf{sf}" for sf in sfs)
        apply_pg(host=args.pg_host, port=args.pg_port,
                 user=args.pg_user, password=args.pg_password,
                 database=args.pg_database, schemas=schemas)

        # DBHub layout: per-SF databases, schema=public
        if args.pg_dbhub_databases:
            for db in (d.strip() for d in args.pg_dbhub_databases.split(",") if d.strip()):
                apply_pg(host=args.pg_host, port=args.pg_port,
                         user=args.pg_user, password=args.pg_password,
                         database=db, schemas=("public",))

    if "mysql" in backends:
        databases = tuple(f"tpch_sf{sf}" for sf in sfs)
        apply_mysql(host=args.mysql_host, port=args.mysql_port,
                    user=args.mysql_user, password=args.mysql_password,
                    databases=databases)

    _LOG.info("all backends complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
