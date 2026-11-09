"""Tests for the shared MCP server adapter utilities."""

from __future__ import annotations

import json

from mcp_dbs_bench.servers._base import (
    encode_jsonrpc_tool_call_response,
    encode_jsonrpc_tools_list_response,
)


def test_tools_list_envelope_is_valid_jsonrpc():
    raw = encode_jsonrpc_tools_list_response([{"name": "query"}], request_id=42)
    payload = json.loads(raw)
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == 42
    assert payload["result"]["tools"] == [{"name": "query"}]


def test_tool_call_response_json_flat_table():
    raw = encode_jsonrpc_tool_call_response(
        columns=["x", "y"], rows=[(1, "a"), (2, "b")], output_format="json_flat_table"
    )
    payload = json.loads(raw)
    assert payload["result"]["columns"] == ["x", "y"]
    assert payload["result"]["rows"] == [[1, "a"], [2, "b"]]


def test_tool_call_response_json_per_row():
    raw = encode_jsonrpc_tool_call_response(
        columns=["x", "y"], rows=[(1, "a"), (2, "b")], output_format="json_per_row"
    )
    payload = json.loads(raw)
    assert payload["result"] == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]


def test_tool_call_response_csv():
    raw = encode_jsonrpc_tool_call_response(
        columns=["x", "y"], rows=[(1, "a"), (2, "b")], output_format="csv"
    )
    payload = json.loads(raw)
    assert "x,y" in payload["result"]
    assert "1,a" in payload["result"]


def test_csv_escapes_commas_and_quotes():
    raw = encode_jsonrpc_tool_call_response(
        columns=["x"], rows=[("a,b",), ('c"d',)], output_format="csv"
    )
    payload = json.loads(raw)
    assert '"a,b"' in payload["result"]
    assert '"c""d"' in payload["result"]


def test_response_handles_decimal_and_date():
    import datetime
    from decimal import Decimal

    raw = encode_jsonrpc_tool_call_response(
        columns=["amt", "d"],
        rows=[(Decimal("1.5"), datetime.date(2026, 5, 25))],
        output_format="json_flat_table",
    )
    payload = json.loads(raw)
    assert payload["result"]["rows"][0][0] == 1.5
    assert payload["result"]["rows"][0][1] == "2026-05-25"


def test_response_handles_none_in_json_per_row():
    raw = encode_jsonrpc_tool_call_response(
        columns=["x"], rows=[(None,)], output_format="json_per_row"
    )
    payload = json.loads(raw)
    assert payload["result"][0]["x"] is None
