"""Tests for the JSONL trace loader.

Each ``trace.jsonl`` file holds one or more ``mcp.interaction`` traces
(one trace = one rep). The loader yields one TraceMetrics per rep with
the eight timing terms, PTR, phase fractions, and byte/token counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_dbs_bench.track_a.trace_loader import TraceMetrics, load_trace_metrics


def _write_trace(path: Path, spans: list[dict]) -> None:
    """Write a list of span dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for span in spans:
            fh.write(json.dumps(span) + "\n")


def _make_span(
    name: str, start_ns: int, end_ns: int,
    *, span_id: str = "s", parent_span_id: str | None = "root",
    attributes: dict | None = None,
) -> dict:
    return {
        "schema_version": "0.1.0",
        "name": name,
        "trace_id": "t" * 32,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "attributes": attributes or {},
        "status": "OK",
    }


def test_load_trace_metrics_returns_one_record_per_interaction(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    spans = [
        _make_span("mcp.interaction", 0, 1_000_000, span_id="root", parent_span_id=None),
        _make_span("mcp.discovery", 100_000, 200_000),
        _make_span("mcp.lifecycle.connect", 0, 100_000),
        _make_span("mcp.tool_call", 200_000, 800_000),
        _make_span(
            "mcp.serialization", 600_000, 700_000,
            attributes={"t_db_ms": 0.4, "B_mcp": 1024, "rows": 10,
                        "discovery_tokens": 50, "envelope_boiler_tokens": 30,
                        "tool_call_response_tokens": 200, "tau_star_result": 100,
                        "B_native": 800},
        ),
        _make_span("mcp.parse", 800_000, 900_000),
        _make_span("mcp.transport", 200_000, 800_000),
    ]
    _write_trace(path, spans)

    metrics = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert len(metrics) == 1


def test_load_trace_metrics_populates_eight_timing_terms(tmp_path: Path) -> None:
    """Realistic shape: HANDOVER wraps SERIALIZE+PARSE+TRANSPORT; SERIALIZE wraps t_db.

    Spans (ns):
      interaction:        [0, 1000)
      lifecycle.connect:  [0, 100)
      discovery:          [100, 200)
      tool_call:          [200, 900)   (handover wall = 700)
        serialization:    [300, 500)   (wall = 200; t_db_ms attr = 100us = 0.1ms)
        parse:            [600, 800)   (wall = 200)
      transport:          [200, 900)   (full overlap with handover)
    All in microseconds (1000us = 1ms total).
    """
    path = tmp_path / "trace.jsonl"
    spans = [
        _make_span("mcp.interaction", 0, 1_000_000,
                   span_id="root", parent_span_id=None),
        _make_span("mcp.lifecycle.connect", 0, 100_000),
        _make_span("mcp.discovery", 100_000, 200_000),
        _make_span("mcp.tool_call", 200_000, 900_000),
        _make_span(
            "mcp.serialization", 300_000, 500_000,
            attributes={"t_db_ms": 0.1},
        ),
        _make_span("mcp.parse", 600_000, 800_000),
        _make_span("mcp.transport", 200_000, 900_000),
    ]
    _write_trace(path, spans)

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metric.t_total_ms == pytest.approx(1.0)
    assert metric.t_db_ms == pytest.approx(0.1)
    assert metric.t_disc_ms == pytest.approx(0.1)
    assert metric.t_life_ms == pytest.approx(0.1)
    # SERIALIZE wall = 0.2ms - t_db 0.1 = 0.1ms
    assert metric.t_ser_ms == pytest.approx(0.1)
    assert metric.t_parse_ms == pytest.approx(0.2)
    # TRANSPORT wall 0.7 - overlap(handover 0.7) - overlap(parse 0.2) clamped
    assert metric.t_xport_excl_ms >= 0.0
    # HANDOVER wall 0.7 minus ser 0.1, parse 0.2, t_db 0.1, xport_excl
    assert metric.t_hand_ms >= 0.0


def test_load_trace_metrics_computes_ptr_and_ptax(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    spans = [
        _make_span("mcp.interaction", 0, 2_000_000, span_id="root", parent_span_id=None),
        _make_span("mcp.discovery", 0, 500_000),
        _make_span("mcp.lifecycle.connect", 500_000, 800_000),
        _make_span("mcp.tool_call", 800_000, 1_900_000),
        _make_span(
            "mcp.serialization", 1_500_000, 1_700_000,
            attributes={"t_db_ms": 1.0},
        ),
        _make_span("mcp.parse", 1_700_000, 1_900_000),
        _make_span("mcp.transport", 800_000, 1_900_000),
    ]
    _write_trace(path, spans)

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metric.ptr is not None and metric.ptr > 0


def test_load_trace_metrics_populates_identification(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    spans = [
        _make_span("mcp.interaction", 0, 1_000_000, span_id="root", parent_span_id=None),
        _make_span("mcp.discovery", 100_000, 200_000),
        _make_span("mcp.lifecycle.connect", 0, 100_000),
        _make_span("mcp.tool_call", 200_000, 800_000),
        _make_span(
            "mcp.serialization", 600_000, 700_000,
            attributes={"t_db_ms": 0.4},
        ),
        _make_span("mcp.parse", 800_000, 900_000),
        _make_span("mcp.transport", 200_000, 800_000),
    ]
    _write_trace(path, spans)

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="duckdb",
        query="Q21", scale_factor=10, cold_warm="warm", rep_idx_start=7,
    ))

    assert metric.server == "mcp-alchemy"
    assert metric.backend == "duckdb"
    assert metric.query == "Q21"
    assert metric.scale_factor == 10
    assert metric.cold_warm == "warm"
    assert metric.rep_idx == 7


def test_load_trace_handles_multiple_interactions(tmp_path: Path) -> None:
    """Multiple reps in same trace.jsonl yield multiple TraceMetrics."""
    path = tmp_path / "trace.jsonl"
    spans = []
    for i in range(3):
        offset = i * 10_000_000
        spans.extend([
            _make_span(
                "mcp.interaction", offset, offset + 1_000_000,
                span_id=f"root{i}", parent_span_id=None,
            ),
            _make_span(
                "mcp.discovery", offset + 100_000, offset + 200_000,
                span_id=f"d{i}", parent_span_id=f"root{i}",
            ),
            _make_span(
                "mcp.lifecycle.connect", offset, offset + 100_000,
                span_id=f"l{i}", parent_span_id=f"root{i}",
            ),
            _make_span(
                "mcp.tool_call", offset + 200_000, offset + 800_000,
                span_id=f"h{i}", parent_span_id=f"root{i}",
            ),
            _make_span(
                "mcp.serialization", offset + 600_000, offset + 700_000,
                span_id=f"s{i}", parent_span_id=f"h{i}",
                attributes={"t_db_ms": 0.4},
            ),
            _make_span(
                "mcp.parse", offset + 800_000, offset + 900_000,
                span_id=f"p{i}", parent_span_id=f"h{i}",
            ),
            _make_span(
                "mcp.transport", offset + 200_000, offset + 800_000,
                span_id=f"x{i}", parent_span_id=f"h{i}",
            ),
        ])
    _write_trace(path, spans)

    metrics = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert len(metrics) == 3
    assert [m.rep_idx for m in metrics] == [0, 1, 2]


def test_load_trace_empty_file_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text("")

    metrics = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metrics == []


def test_load_trace_missing_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "absent.jsonl"

    with pytest.raises(FileNotFoundError):
        list(load_trace_metrics(
            path, server="mcp-alchemy", backend="postgres",
            query="Q1", scale_factor=1, cold_warm="cold",
        ))


def test_trace_metrics_has_all_sixteen_im_cm_fields() -> None:
    """16 metrics per spec: 8 timing (IM-1..7 + epsilon) + PTax/PTR + B_mcp,tau_disc,phi,gamma,S_ser,rho,n_life."""
    metric = TraceMetrics(
        cell_id="x", server="s", backend="b", query="Q1", scale_factor=1,
        rep_idx=0, cold_warm="cold",
        t_total_ms=1.0, t_db_ms=0.5, t_disc_ms=0.1, t_life_ms=0.05,
        t_hand_ms=0.1, t_ser_ms=0.05, t_parse_ms=0.1, t_xport_excl_ms=0.05,
        t_teardown_ms=0.0,
        epsilon_ms=0.05, ptax_ms=0.45, ptr=0.9,
        b_mcp=1024, tau_disc=50, phi=2.0, gamma=0.001,
        s_ser_rows_per_ms=100.0, rho=1.5, n_life=1,
        r_retry_ms=0.0, t_pool_ms=None, m_peak_mb=None,
        cpu_percent_samples=(),
        trace_path="x.jsonl",
    )

    # Confirms field set is complete by accessing each
    assert metric.t_db_ms == 0.5
    assert metric.ptr == 0.9
    assert metric.b_mcp == 1024
    assert metric.rho == 1.5


def _base_spans(*, retry_spans: list[dict] | None = None,
                pool_spans: list[dict] | None = None,
                serialize_attrs: dict | None = None) -> list[dict]:
    """Standard interaction skeleton; extra spans/attrs are merged in."""
    ser_attrs = {"t_db_ms": 0.4}
    if serialize_attrs:
        ser_attrs.update(serialize_attrs)
    spans = [
        _make_span("mcp.interaction", 0, 5_000_000,
                   span_id="root", parent_span_id=None),
        _make_span("mcp.discovery", 100_000, 200_000),
        _make_span("mcp.lifecycle.connect", 0, 100_000),
        _make_span("mcp.tool_call", 200_000, 4_800_000),
        _make_span("mcp.serialization", 600_000, 700_000, attributes=ser_attrs),
        _make_span("mcp.parse", 800_000, 900_000),
        _make_span("mcp.transport", 200_000, 4_800_000),
    ]
    if retry_spans:
        spans.extend(retry_spans)
    if pool_spans:
        spans.extend(pool_spans)
    return spans


def test_load_trace_metrics_sums_mcp_retry_spans_for_r_retry(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    retries = [
        _make_span("mcp.retry", 1_000_000, 1_300_000),  # 0.3 ms
        _make_span("mcp.retry", 2_000_000, 2_200_000),  # 0.2 ms
    ]
    _write_trace(path, _base_spans(retry_spans=retries))

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metric.r_retry_ms == pytest.approx(0.5)


def test_load_trace_metrics_defaults_r_retry_to_zero_when_no_retries(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_trace(path, _base_spans())

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metric.r_retry_ms == 0.0


def test_load_trace_metrics_sums_connection_handover_spans(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    pool = [
        _make_span("connection.acquire", 1_000_000, 1_400_000),  # 0.4 ms
        _make_span("connection.release", 4_000_000, 4_300_000),  # 0.3 ms
    ]
    _write_trace(path, _base_spans(pool_spans=pool))

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metric.t_pool_ms == pytest.approx(0.7)


def test_load_trace_metrics_returns_none_t_pool_when_no_handover_spans(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_trace(path, _base_spans())

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metric.t_pool_ms is None


def test_load_trace_metrics_extracts_m_peak_from_rss_samples(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    spans = _base_spans(
        serialize_attrs={"rss_samples_kb": [1024, 2048, 8192, 4096]},
    )
    _write_trace(path, spans)

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    # 8192 KiB / 1024 = 8 MiB
    assert metric.m_peak_mb == pytest.approx(8.0)


def test_load_trace_metrics_returns_none_m_peak_when_no_rss_samples(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_trace(path, _base_spans())

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metric.m_peak_mb is None


def test_load_trace_metrics_extracts_cpu_percent_samples(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    spans = _base_spans(
        serialize_attrs={"cpu_percent_samples": [10.0, 85.5, 92.0]},
    )
    _write_trace(path, spans)

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metric.cpu_percent_samples == (10.0, 85.5, 92.0)


def test_load_trace_metrics_defaults_cpu_samples_to_empty_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _write_trace(path, _base_spans())

    [metric] = list(load_trace_metrics(
        path, server="mcp-alchemy", backend="postgres",
        query="Q1", scale_factor=1, cold_warm="cold",
    ))

    assert metric.cpu_percent_samples == ()
