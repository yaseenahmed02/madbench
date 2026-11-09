"""Tests for the OTel tracer + JSONL exporter."""

from __future__ import annotations

import json
from pathlib import Path


from mcp_dbs_bench.instrumentation.tracer import OTelTracer


def test_jsonl_exporter_writes_one_span_per_line(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    tracer = OTelTracer.with_jsonl_exporter(out)
    with tracer.start_as_current_span("mcp.discovery") as span:
        span.set_attribute("response.bytes", 100)
        span.set_attribute("tool.count", 4)
    tracer.shutdown()
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["name"] == "mcp.discovery"
    assert record["attributes"]["response.bytes"] == 100
    assert record["attributes"]["tool.count"] == 4
    assert record["start_ns"] > 0
    assert record["end_ns"] >= record["start_ns"]


def test_tracer_records_parent_child_relationship(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    tracer = OTelTracer.with_jsonl_exporter(out)
    with tracer.start_as_current_span("mcp.tool_call") as parent:
        parent.set_attribute("tool.name", "query")
        with tracer.start_as_current_span("db.execute") as child:
            child.set_attribute("db.statement", "SELECT 1")
    tracer.shutdown()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    # OTel writes children first (end-order), then parents.
    by_name = {r["name"]: r for r in records}
    assert by_name["db.execute"]["parent_span_id"] == by_name["mcp.tool_call"]["span_id"]


def test_tracer_records_attribute_with_dot_keys(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    tracer = OTelTracer.with_jsonl_exporter(out)
    with tracer.start_as_current_span("mcp.interaction") as span:
        span.set_attribute("server.name", "anthropic-pg")
        span.set_attribute("dbms.backend", "duckdb")
    tracer.shutdown()
    record = json.loads(out.read_text().splitlines()[0])
    assert record["attributes"]["server.name"] == "anthropic-pg"


def test_tracer_shutdown_is_idempotent(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    tracer = OTelTracer.with_jsonl_exporter(out)
    tracer.shutdown()
    tracer.shutdown()


def test_tracer_records_schema_version(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    tracer = OTelTracer.with_jsonl_exporter(out)
    with tracer.start_as_current_span("x"):
        pass
    tracer.shutdown()
    record = json.loads(out.read_text().splitlines()[0])
    assert "schema_version" in record
