"""Tests for the OTel-backed PhaseContext.

This binding wraps PhaseContextImpl: records phase intervals AND emits OTel
spans with the canonical names from Phase.span_name.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.instrumentation.phase_context_otel import OTelPhaseContext
from mcp_dbs_bench.instrumentation.tracer import OTelTracer


def test_otel_phase_context_emits_span_with_canonical_name(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    tracer = OTelTracer.with_jsonl_exporter(out)
    ctx = OTelPhaseContext(tracer=tracer)
    with ctx.span(Phase.DISCOVERY, response_bytes=100):
        pass
    tracer.shutdown()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    names = [r["name"] for r in records]
    assert Phase.DISCOVERY.span_name in names


def test_otel_phase_context_attaches_attributes(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    tracer = OTelTracer.with_jsonl_exporter(out)
    ctx = OTelPhaseContext(tracer=tracer)
    with ctx.span(Phase.HANDOVER, request_bytes=50):
        pass
    tracer.shutdown()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    handover = next(r for r in records if r["name"] == "mcp.tool_call")
    assert handover["attributes"]["request_bytes"] == 50


def test_otel_phase_context_records_phase_records_for_decomposition(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    tracer = OTelTracer.with_jsonl_exporter(out)
    ctx = OTelPhaseContext(tracer=tracer)
    with ctx.span(Phase.DISCOVERY):
        pass
    with ctx.span(Phase.HANDOVER):
        pass
    tracer.shutdown()
    assert len(ctx.records) == 2
    assert ctx.records[0].phase is Phase.DISCOVERY
    assert ctx.records[1].phase is Phase.HANDOVER
