"""Tests for the plugin-SDK data types.

These dataclasses are the contract between plugins, the harness, and the
metric layer. Frozen + slots ensures plugins cannot mutate values mid-run.
"""

from __future__ import annotations

import pytest

from mcp_dbs_bench.core.types import (
    ConnectionHandle,
    ConnectionModel,
    OutputFormat,
    ServerCapabilities,
    ToolCallRequest,
    ToolCallResponse,
    ToolManifest,
)


def test_output_format_has_canonical_values():
    # These values appear in span attributes; renaming breaks downstream parquet.
    assert OutputFormat.JSON_FLAT_TABLE.value == "json_flat_table"
    assert OutputFormat.JSON_PER_ROW.value == "json_per_row"
    assert OutputFormat.CSV.value == "csv"
    assert OutputFormat.VERTICAL_TEXT.value == "vertical_text"


def test_connection_model_includes_sqlalchemy():
    assert ConnectionModel.SQLALCHEMY.value == "sqlalchemy"
    assert ConnectionModel.GATEWAY.value == "gateway"
    assert ConnectionModel.IMPLICIT_POOL.value == "implicit_pool"


def test_server_capabilities_is_frozen():
    caps = ServerCapabilities(
        output_format=OutputFormat.JSON_FLAT_TABLE,
        connection_model=ConnectionModel.IMPLICIT_POOL,
        declared_tool_count=1,
        supports_streaming=False,
        safety_mechanism="read_only_tx",
        schema_depth="basic",
    )
    with pytest.raises((AttributeError, TypeError)):
        caps.declared_tool_count = 999  # type: ignore[misc]


def test_server_capabilities_rejects_negative_tool_count():
    with pytest.raises(ValueError, match="declared_tool_count"):
        ServerCapabilities(
            output_format=OutputFormat.CSV,
            connection_model=ConnectionModel.GATEWAY,
            declared_tool_count=-1,
            supports_streaming=False,
            safety_mechanism="none",
            schema_depth="basic",
        )


def test_tool_call_request_byte_size_returns_int():
    req = ToolCallRequest(name="query", arguments={"sql": "SELECT 1"})
    assert isinstance(req.byte_size(), int)
    assert req.byte_size() > 0


def test_tool_call_request_byte_size_is_canonical_json_encoded():
    # Same arguments => same byte size; order-independent for dict keys.
    req_a = ToolCallRequest(name="q", arguments={"a": 1, "b": 2})
    req_b = ToolCallRequest(name="q", arguments={"b": 2, "a": 1})
    assert req_a.byte_size() == req_b.byte_size()


def test_tool_call_response_holds_rows_and_raw_bytes():
    resp = ToolCallResponse(
        rows=[("a", 1), ("b", 2)],
        raw_bytes=b'[{"x":"a","y":1},{"x":"b","y":2}]',
        column_names=("x", "y"),
    )
    assert resp.row_count == 2
    assert resp.byte_size == len(resp.raw_bytes)


def test_tool_call_response_row_count_zero_for_empty():
    resp = ToolCallResponse(rows=[], raw_bytes=b"[]", column_names=())
    assert resp.row_count == 0


def test_tool_manifest_lists_tools():
    manifest = ToolManifest(
        tools=[
            {"name": "query", "description": "Run SQL"},
            {"name": "describe", "description": "Show schema"},
        ],
        raw_bytes=b'{"tools":[...]}',
    )
    assert manifest.tool_count == 2
    assert manifest.tool_names == ("query", "describe")


def test_tool_manifest_from_jsonrpc_extracts_tools_field():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": [{"name": "query"}, {"name": "list_tables"}]},
    }
    manifest = ToolManifest.from_jsonrpc(payload)
    assert manifest.tool_count == 2
    assert "query" in manifest.tool_names


def test_connection_handle_wraps_arbitrary_value():
    conn = ConnectionHandle(value=object())
    assert conn.value is not None


# ---------- edge branches ----------


def test_server_capabilities_rejects_empty_safety_mechanism():
    """``safety_mechanism`` is a contract field; the empty string must
    raise so a misconfigured plugin fails fast at construction."""
    with pytest.raises(ValueError, match="safety_mechanism"):
        ServerCapabilities(
            output_format=OutputFormat.JSON_FLAT_TABLE,
            connection_model=ConnectionModel.IMPLICIT_POOL,
            declared_tool_count=1, supports_streaming=False,
            safety_mechanism="", schema_depth="basic",
        )


def test_server_capabilities_rejects_unknown_schema_depth():
    """``schema_depth`` is enum-like; only the four documented values."""
    with pytest.raises(ValueError, match="schema_depth"):
        ServerCapabilities(
            output_format=OutputFormat.JSON_FLAT_TABLE,
            connection_model=ConnectionModel.IMPLICIT_POOL,
            declared_tool_count=1, supports_streaming=False,
            safety_mechanism="none", schema_depth="bogus",
        )


def test_tool_call_request_byte_size_with_no_arguments():
    """``byte_size`` must work on a request with empty arguments and
    return a positive byte count for the wrapper envelope."""
    request = ToolCallRequest(name="ping", arguments={})
    size = request.byte_size()
    assert size > 0


def test_tool_manifest_from_jsonrpc_rejects_missing_tools_field():
    """A JSON-RPC payload without ``result.tools`` must raise loudly."""
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"servers": []}}
    with pytest.raises(ValueError, match="result.tools"):
        ToolManifest.from_jsonrpc(payload)


def test_tool_manifest_from_jsonrpc_rejects_non_list_tools():
    """A ``result.tools`` that isn't a list must raise (covers the type
    check at types.py:149)."""
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"tools": {"name": "x"}}}
    with pytest.raises(ValueError, match="must be a list"):
        ToolManifest.from_jsonrpc(payload)
