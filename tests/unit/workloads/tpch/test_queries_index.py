"""Tests for the TPC-H query index."""

from __future__ import annotations

import pytest

from mcp_dbs_bench.workloads.tpch import expected_cardinality, load_query
from mcp_dbs_bench.workloads.tpch.queries_index import DIALECTS, QUERY_IDS


def test_query_ids_match_v01_scope():
    # The benchmark query set: Q1, Q3, Q6, Q9, Q21.
    assert set(QUERY_IDS) == {"Q1", "Q3", "Q6", "Q9", "Q21"}


def test_dialects_match_v01_scope():
    assert set(DIALECTS) == {"postgres", "mysql", "duckdb"}


@pytest.mark.parametrize("query_id", QUERY_IDS)
@pytest.mark.parametrize("dialect", DIALECTS)
def test_every_query_dialect_combination_has_a_file(query_id, dialect):
    sql = load_query(query_id, dialect)
    assert sql.strip().upper().startswith("SELECT") or sql.strip().startswith("--")


@pytest.mark.parametrize("query_id", QUERY_IDS)
@pytest.mark.parametrize("dialect", DIALECTS)
def test_every_query_mentions_an_expected_tpch_table(query_id, dialect):
    sql = load_query(query_id, dialect).lower()
    # Every TPC-H query touches at least one of the 8 standard tables.
    tpch_tables = ["lineitem", "orders", "customer", "supplier", "part", "partsupp", "nation", "region"]
    assert any(t in sql for t in tpch_tables), f"{query_id}/{dialect} mentions no TPC-H table"


def test_load_query_rejects_unknown_query_id():
    with pytest.raises(ValueError, match="query_id"):
        load_query("Q99", "duckdb")


def test_load_query_rejects_unknown_dialect():
    with pytest.raises(ValueError, match="dialect"):
        load_query("Q1", "oracle")


def test_expected_cardinality_for_q1_is_4_at_any_sf():
    # Q1 always returns 4 rows regardless of scale (constant cardinality).
    assert expected_cardinality("Q1", 1) == 4
    assert expected_cardinality("Q1", 10) == 4


def test_expected_cardinality_for_q21_grows_with_sf():
    # Q21 ~100 rows at SF1, ~411 at SF10.
    assert expected_cardinality("Q21", 1) == 100
    assert expected_cardinality("Q21", 10) == 411


def test_expected_cardinality_unknown_raises():
    with pytest.raises(KeyError):
        expected_cardinality("Q99", 1)
