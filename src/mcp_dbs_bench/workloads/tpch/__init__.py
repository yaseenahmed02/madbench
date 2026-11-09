"""TPC-H workload: schema loader + dialect-validated queries.

Public API:
    - `SCHEMA_DDL`           : per-backend CREATE TABLE statements.
    - `QUERIES`              : Q1, Q3, Q6, Q9, Q21 across pg/mysql/duckdb.
    - `load_query(query_id, dialect)` : read SQL text from disk.
    - `expected_cardinality(query_id, scale_factor)` : expected row count.

The five queries cover both result-size regimes:
    - Q1, Q6  : constant cardinality (engine scaling baseline)
    - Q3, Q9, Q21 : growing cardinality (serialization stress)
"""

from __future__ import annotations

from mcp_dbs_bench.workloads.tpch.queries_index import (
    QUERIES,
    QUERY_IDS,
    expected_cardinality,
    load_query,
)
from mcp_dbs_bench.workloads.tpch.schema import SCHEMA_DDL

__all__ = [
    "QUERIES",
    "QUERY_IDS",
    "SCHEMA_DDL",
    "expected_cardinality",
    "load_query",
]
