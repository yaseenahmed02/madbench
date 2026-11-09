"""Tests for the CSV writers (per_trace + per_cell_summary)."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from mcp_dbs_bench.track_a.csv_writer import (
    PER_CELL_SUMMARY_COLUMNS,
    PER_TRACE_COLUMNS,
    write_per_cell_summary,
    write_per_trace_metrics,
    write_phase_breakdown,
    write_ptr_by_cell,
    write_token_metrics,
)
from mcp_dbs_bench.track_a.trace_loader import TraceMetrics


def _make_metric(
    rep_idx: int = 0, cold_warm: str = "cold", t_db_ms: float = 0.5,
    ptr: float | None = 0.9, scale_factor: int = 1,
    r_retry_ms: float = 0.0, t_pool_ms: float | None = None,
    m_peak_mb: float | None = None,
    cpu_percent_samples: tuple[float, ...] = (),
    t_total_ms: float = 1.0,
) -> TraceMetrics:
    return TraceMetrics(
        cell_id=f"mcp-alchemy_postgres_Q1_sf{scale_factor}_" + cold_warm,
        server="mcp-alchemy", backend="postgres", query="Q1",
        scale_factor=scale_factor, rep_idx=rep_idx, cold_warm=cold_warm,
        t_total_ms=t_total_ms, t_db_ms=t_db_ms, t_disc_ms=0.05, t_life_ms=0.05,
        t_hand_ms=0.1, t_ser_ms=0.05, t_parse_ms=0.05, t_xport_excl_ms=0.05,
        t_teardown_ms=0.0,
        epsilon_ms=0.15, ptax_ms=0.35, ptr=ptr,
        b_mcp=1024, tau_disc=50, phi=2.0, gamma=0.001,
        s_ser_rows_per_ms=100.0, rho=1.5, n_life=1,
        r_retry_ms=r_retry_ms, t_pool_ms=t_pool_ms, m_peak_mb=m_peak_mb,
        cpu_percent_samples=cpu_percent_samples,
        trace_path="trace.jsonl",
    )


def test_per_trace_metrics_writes_header_and_one_row_per_metric(tmp_path: Path) -> None:
    path = tmp_path / "per_trace_metrics.csv"
    metrics = [_make_metric(rep_idx=i) for i in range(3)]

    write_per_trace_metrics(path, metrics)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert all(set(row.keys()) == set(PER_TRACE_COLUMNS) for row in rows)


def test_per_trace_metrics_serializes_none_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "per_trace_metrics.csv"
    metric = _make_metric(ptr=None)

    write_per_trace_metrics(path, [metric])

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["PTR"] == ""


def test_per_trace_metrics_uses_proper_csv_escaping(tmp_path: Path) -> None:
    """Trace paths with commas must not break the CSV."""
    path = tmp_path / "per_trace_metrics.csv"
    metric = _make_metric()
    metric_with_comma = replace(metric, trace_path="weird,name,with,commas.jsonl")

    write_per_trace_metrics(path, [metric_with_comma])

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["trace_path"] == "weird,name,with,commas.jsonl"


def test_per_cell_summary_writes_one_row_per_cell(tmp_path: Path) -> None:
    path = tmp_path / "per_cell_summary.csv"
    metrics_cell_a = [_make_metric(rep_idx=i, cold_warm="cold") for i in range(30)]
    metrics_cell_b = [_make_metric(rep_idx=i, cold_warm="warm") for i in range(30)]

    write_per_cell_summary(
        path, [metrics_cell_a, metrics_cell_b], n_bootstrap=200,
    )

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert all(set(PER_CELL_SUMMARY_COLUMNS).issubset(set(row.keys())) for row in rows)


def test_per_cell_summary_includes_bootstrap_ci_for_ptr(tmp_path: Path) -> None:
    path = tmp_path / "per_cell_summary.csv"
    metrics = [_make_metric(rep_idx=i, t_db_ms=0.5, ptr=0.9 + 0.01 * i) for i in range(30)]

    write_per_cell_summary(path, [metrics], n_bootstrap=200)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert "PTR_ci_lo" in rows[0]
    assert "PTR_ci_hi" in rows[0]
    assert float(rows[0]["PTR_ci_lo"]) <= float(rows[0]["PTR_ci_hi"])


def test_per_cell_summary_handles_all_none_ptr_gracefully(tmp_path: Path) -> None:
    """A cell where every rep has t_db=0 yields PTR=None; CIs should be empty."""
    path = tmp_path / "per_cell_summary.csv"
    metrics = [_make_metric(rep_idx=i, ptr=None) for i in range(5)]

    write_per_cell_summary(path, [metrics], n_bootstrap=200)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["PTR_p50"] == ""


def test_ptr_by_cell_pivots_cold_and_warm(tmp_path: Path) -> None:
    path = tmp_path / "ptr_by_cell.csv"
    cold = [_make_metric(rep_idx=i, cold_warm="cold", ptr=0.5) for i in range(10)]
    warm = [_make_metric(rep_idx=i, cold_warm="warm", ptr=0.3) for i in range(10)]

    write_ptr_by_cell(path, [cold, warm])

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["server"] == "mcp-alchemy"
    assert row["backend"] == "postgres"
    assert "PTR_cold" in row
    assert "PTR_warm" in row
    assert "delta_CW" in row


def test_phase_breakdown_writes_per_cell_phase_durations(tmp_path: Path) -> None:
    path = tmp_path / "phase_breakdown.csv"
    metrics = [_make_metric(rep_idx=i) for i in range(5)]

    write_phase_breakdown(path, [metrics])

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert "T_db" in rows[0]
    assert "T_disc" in rows[0]
    assert "T_life" in rows[0]
    assert "T_hand" in rows[0]
    assert "T_ser" in rows[0]
    assert "T_parse" in rows[0]
    assert "T_xport" in rows[0]


def test_token_metrics_writes_per_cell_token_summary(tmp_path: Path) -> None:
    path = tmp_path / "token_metrics.csv"
    metrics = [_make_metric(rep_idx=i) for i in range(5)]

    write_token_metrics(path, [metrics])

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert "tau_disc" in rows[0]
    assert "B_mcp" in rows[0]
    assert "phi" in rows[0]


def test_per_trace_metrics_emits_r_retry_t_pool_m_peak_columns(tmp_path: Path) -> None:
    path = tmp_path / "per_trace_metrics.csv"
    metric = _make_metric(r_retry_ms=0.5, t_pool_ms=0.3, m_peak_mb=12.5)

    write_per_trace_metrics(path, [metric])

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    row = rows[0]
    assert row["R_retry_ms"] == "0.5"
    assert row["T_pool_ms"] == "0.3"
    assert row["M_peak_mb"] == "12.5"


def test_per_trace_metrics_serializes_missing_t_pool_and_m_peak_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "per_trace_metrics.csv"
    metric = _make_metric(t_pool_ms=None, m_peak_mb=None)

    write_per_trace_metrics(path, [metric])

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["T_pool_ms"] == ""
    assert rows[0]["M_peak_mb"] == ""


def test_per_cell_summary_emits_m_peak_p50_p95(tmp_path: Path) -> None:
    path = tmp_path / "per_cell_summary.csv"
    metrics = [
        _make_metric(rep_idx=i, m_peak_mb=10.0 + i) for i in range(5)
    ]

    write_per_cell_summary(path, [metrics], n_bootstrap=50)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert float(rows[0]["M_peak_mb_p50"]) == 12.0
    assert float(rows[0]["M_peak_mb_p95"]) >= 13.0


def test_per_cell_summary_delta_cw_is_cold_minus_warm_t_total_median(tmp_path: Path) -> None:
    path = tmp_path / "per_cell_summary.csv"
    cold = [
        _make_metric(rep_idx=i, cold_warm="cold", t_total_ms=2.0) for i in range(5)
    ]
    warm = [
        _make_metric(rep_idx=i, cold_warm="warm", t_total_ms=1.0) for i in range(5)
    ]

    write_per_cell_summary(path, [cold, warm], n_bootstrap=50)

    with path.open() as fh:
        rows = {row["cold_warm"]: row for row in csv.DictReader(fh)}
    assert float(rows["cold"]["delta_CW_ms"]) == pytest.approx(1.0)
    assert rows["warm"]["delta_CW_ms"] == ""


def test_per_cell_summary_sigma_star_cpu_picks_smallest_saturating_sf(tmp_path: Path) -> None:
    path = tmp_path / "per_cell_summary.csv"
    cells = [
        [_make_metric(rep_idx=i, scale_factor=1, cpu_percent_samples=(50.0,))
         for i in range(3)],
        [_make_metric(rep_idx=i, scale_factor=3, cpu_percent_samples=(85.0,))
         for i in range(3)],
        [_make_metric(rep_idx=i, scale_factor=10, cpu_percent_samples=(95.0,))
         for i in range(3)],
    ]

    write_per_cell_summary(path, cells, n_bootstrap=50)

    with path.open() as fh:
        rows = {int(row["scale_factor"]): row for row in csv.DictReader(fh)}
    # All three cells share the same sigma_star value derived from the
    # per-SF CPU% map (10 is the first SF >= 90% threshold).
    assert rows[1]["sigma_star_cpu_sf"] == "10"
    assert rows[3]["sigma_star_cpu_sf"] == "10"
    assert rows[10]["sigma_star_cpu_sf"] == "10"


def test_per_cell_summary_sigma_star_cpu_empty_when_no_cpu_samples(tmp_path: Path) -> None:
    path = tmp_path / "per_cell_summary.csv"
    metrics = [_make_metric(rep_idx=i) for i in range(5)]

    write_per_cell_summary(path, [metrics], n_bootstrap=50)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["sigma_star_cpu_sf"] == ""


def test_per_cell_summary_backpressure_classified_when_m_peak_present(tmp_path: Path) -> None:
    path = tmp_path / "per_cell_summary.csv"
    metrics = [_make_metric(rep_idx=i, m_peak_mb=10.0 + i) for i in range(5)]

    write_per_cell_summary(path, [metrics], n_bootstrap=50, memory_limit_mb=1024.0)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    # Below the 95% memory-pressure fraction, default classification is BUFFER_OOM.
    assert rows[0]["backpressure_pattern"] == "buffer_oom"


def test_per_cell_summary_backpressure_empty_when_m_peak_absent(tmp_path: Path) -> None:
    path = tmp_path / "per_cell_summary.csv"
    metrics = [_make_metric(rep_idx=i, m_peak_mb=None) for i in range(5)]

    write_per_cell_summary(path, [metrics], n_bootstrap=50)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["backpressure_pattern"] == ""
