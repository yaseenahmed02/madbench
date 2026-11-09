"""Tests for the SpanRecord dataclass shared by metric computations.

SpanRecord is the validated, in-memory shape every metric operates on. It
mirrors the OTel span shape but is decoupled from the OTel SDK so metric
computation is testable without touching the tracer.
"""

from __future__ import annotations

import pytest

from mcp_dbs_bench.metrics.span_record import SpanRecord


class TestSpanRecord:
    def test_span_record_duration_ns_returns_difference(self) -> None:
        span = SpanRecord(name="mcp.parse", start_ns=1_000_000, end_ns=3_000_000, attributes={})
        assert span.duration_ns == 2_000_000

    def test_span_record_duration_ms_returns_milliseconds(self) -> None:
        span = SpanRecord(name="mcp.parse", start_ns=1_000_000, end_ns=3_000_000, attributes={})
        assert span.duration_ms == 2.0

    def test_span_record_rejects_negative_duration(self) -> None:
        # end_ns < start_ns is a constructed-bad-data case; fail fast.
        with pytest.raises(ValueError, match="end_ns"):
            SpanRecord(name="mcp.parse", start_ns=10_000_000, end_ns=1_000_000, attributes={})

    def test_span_record_zero_duration_allowed(self) -> None:
        # Zero-duration spans are legal (e.g., a phase with no work).
        span = SpanRecord(name="mcp.parse", start_ns=1_000, end_ns=1_000, attributes={})
        assert span.duration_ns == 0

    def test_span_record_attribute_access(self) -> None:
        span = SpanRecord(
            name="mcp.tool_call",
            start_ns=1_000_000,
            end_ns=2_000_000,
            attributes={"response.tokens": 42, "response.bytes": 1024},
        )
        assert span.attributes["response.tokens"] == 42
        assert span.attributes["response.bytes"] == 1024
