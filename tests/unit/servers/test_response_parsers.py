"""Tests for the text-payload parsers used by subprocess wrappers."""

from __future__ import annotations

from mcp_dbs_bench.servers._response_parsers import (
    parse_anthropic_pg_payload,
    parse_dbhub_payload,
    parse_mcp_alchemy_payload,
)


def test_dbhub_parses_single_row():
    text = (
        '{"success": true, "data": {"rows": [{"one": 1}],'
        '"count": 1, "source_id": "default"}}'
    )
    rows, cols = parse_dbhub_payload(text)
    assert rows == [(1,)]
    assert cols == ("one",)


def test_dbhub_parses_empty_result():
    rows, cols = parse_dbhub_payload('{"success": true, "data": {"rows": [], "count": 0}}')
    assert rows == []
    assert cols == ()


def test_anthropic_pg_parses_jsonarray():
    rows, cols = parse_anthropic_pg_payload('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]')
    assert rows == [(1, "x"), (2, "y")]
    assert cols == ("a", "b")





def test_mcp_alchemy_parses_vertical_text():
    text = "1. row\none: 1\n\nResult: 1 rows"
    rows, cols = parse_mcp_alchemy_payload(text)
    assert rows == [(1,)]
    assert cols == ("one",)


def test_mcp_alchemy_parses_multiple_rows():
    text = "1. row\na: 1\nb: x\n\n2. row\na: 2\nb: y\n\nResult: 2 rows"
    rows, cols = parse_mcp_alchemy_payload(text)
    assert rows == [(1, "x"), (2, "y")]
    assert cols == ("a", "b")


def test_mcp_alchemy_coerces_floats_and_nulls():
    text = "1. row\nx: 1.5\ny: None\n\nResult: 1 rows"
    rows, cols = parse_mcp_alchemy_payload(text)
    assert rows == [(1.5, None)]
    assert cols == ("x", "y")
