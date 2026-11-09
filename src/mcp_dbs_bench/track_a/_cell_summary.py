"""Per-cell summary row builders for ``per_cell_summary.csv``.

Split out of :mod:`csv_writer` to keep that module under the 400-line
length budget. The exposed API is the single :func:`summarize_cell`
entry point plus the two cross-cell index builders required by IM-6
(δ_CW) and CM-4 (σ*_CPU).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mcp_dbs_bench.analysis.bootstrap import BOOTSTRAP_SEED, bootstrap_statistic
from mcp_dbs_bench.metrics.coupling.back_pressure import (
    BackpressurePattern,
    cm6_backpressure_classification,
)
from mcp_dbs_bench.metrics.coupling.cpu_saturation import (
    cm4_cpu_saturation_threshold,
)
from mcp_dbs_bench.track_a.trace_loader import TraceMetrics

_CPU_SATURATION_THRESHOLD_PCT = 90.0


def _format(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def summarize_cell(
    metrics: Sequence[TraceMetrics], *, n_bootstrap: int,
    warm_total_median: dict[tuple, float],
    sf_to_cpu_pct: dict[tuple, dict[int, float]],
    memory_limit_mb: float,
    columns: Sequence[str],
) -> dict[str, str]:
    """Build one row for ``per_cell_summary.csv``."""
    if not metrics:
        return {col: "" for col in columns}
    head = metrics[0]
    row: dict[str, str] = {
        "cell_id": head.cell_id,
        "server": head.server,
        "backend": head.backend,
        "query": head.query,
        "scale_factor": str(head.scale_factor),
        "cold_warm": head.cold_warm,
        "n_reps": str(len(metrics)),
    }
    _write_timing_percentiles(row, metrics)
    _write_ptr_summary(row, metrics, n_bootstrap=n_bootstrap)
    _write_ratio_summaries(row, metrics)
    _write_peak_memory_summary(row, metrics)
    _write_delta_cw(row, metrics, warm_total_median=warm_total_median)
    _write_cpu_saturation(row, metrics, sf_to_cpu_pct=sf_to_cpu_pct)
    _write_backpressure(row, metrics, memory_limit_mb=memory_limit_mb)
    return row


def build_warm_total_index(
    cells: Sequence[Sequence[TraceMetrics]],
) -> dict[tuple, float]:
    index: dict[tuple, float] = {}
    for cell in cells:
        if not cell:
            continue
        head = cell[0]
        if head.cold_warm != "warm":
            continue
        key = (head.server, head.backend, head.query, head.scale_factor)
        index[key] = float(np.median([m.t_total_ms for m in cell]))
    return index


def build_sf_cpu_index(
    cells: Sequence[Sequence[TraceMetrics]],
) -> dict[tuple, dict[int, float]]:
    """One SF->CPU% map per (server, backend, query, cold_warm) group.

    CPU% per SF is the median of the per-rep peak ``cpu_percent_samples``.
    Cells without sampled CPU% contribute nothing.
    """
    grouped: dict[tuple, dict[int, float]] = {}
    for cell in cells:
        if not cell:
            continue
        head = cell[0]
        per_rep_peaks = [
            max(m.cpu_percent_samples) for m in cell if m.cpu_percent_samples
        ]
        if not per_rep_peaks:
            continue
        key = (head.server, head.backend, head.query, head.cold_warm)
        grouped.setdefault(key, {})[head.scale_factor] = float(np.median(per_rep_peaks))
    return grouped


def _write_timing_percentiles(
    row: dict[str, str], metrics: Sequence[TraceMetrics],
) -> None:
    t_db = np.array([m.t_db_ms for m in metrics], dtype=float)
    t_total = np.array([m.t_total_ms for m in metrics], dtype=float)
    ptax = np.array([m.ptax_ms for m in metrics], dtype=float)
    row["T_db_p50"] = _format(float(np.percentile(t_db, 50)))
    row["T_db_p95"] = _format(float(np.percentile(t_db, 95)))
    row["T_total_p50"] = _format(float(np.percentile(t_total, 50)))
    row["T_total_p95"] = _format(float(np.percentile(t_total, 95)))
    row["PTax_p50"] = _format(float(np.percentile(ptax, 50)))
    row["PTax_p95"] = _format(float(np.percentile(ptax, 95)))


def _write_ptr_summary(
    row: dict[str, str], metrics: Sequence[TraceMetrics], *, n_bootstrap: int,
) -> None:
    ptr_samples = np.array(
        [m.ptr for m in metrics if m.ptr is not None], dtype=float,
    )
    if ptr_samples.size == 0:
        row["PTR_p50"] = ""
        row["PTR_p95"] = ""
        row["PTR_ci_lo"] = ""
        row["PTR_ci_hi"] = ""
        return
    row["PTR_p50"] = _format(float(np.percentile(ptr_samples, 50)))
    row["PTR_p95"] = _format(float(np.percentile(ptr_samples, 95)))
    ci = bootstrap_statistic(
        ptr_samples, np.median, n_resamples=n_bootstrap, seed=BOOTSTRAP_SEED,
    )
    row["PTR_ci_lo"] = _format(ci.ci_low_95)
    row["PTR_ci_hi"] = _format(ci.ci_high_95)


def _write_ratio_summaries(
    row: dict[str, str], metrics: Sequence[TraceMetrics],
) -> None:
    rho_samples = np.array(
        [m.rho for m in metrics if m.rho is not None], dtype=float,
    )
    phi_samples = np.array(
        [m.phi for m in metrics if m.phi is not None], dtype=float,
    )
    row["rho_p50"] = (
        _format(float(np.percentile(rho_samples, 50))) if rho_samples.size > 0 else ""
    )
    row["phi_p50"] = (
        _format(float(np.percentile(phi_samples, 50))) if phi_samples.size > 0 else ""
    )


def _write_peak_memory_summary(
    row: dict[str, str], metrics: Sequence[TraceMetrics],
) -> None:
    samples = np.array(
        [m.m_peak_mb for m in metrics if m.m_peak_mb is not None], dtype=float,
    )
    if samples.size == 0:
        row["M_peak_mb_p50"] = ""
        row["M_peak_mb_p95"] = ""
        return
    row["M_peak_mb_p50"] = _format(float(np.percentile(samples, 50)))
    row["M_peak_mb_p95"] = _format(float(np.percentile(samples, 95)))


def _write_delta_cw(
    row: dict[str, str], metrics: Sequence[TraceMetrics],
    *, warm_total_median: dict[tuple, float],
) -> None:
    """IM-6 delta_CW reported on the cold-cell row only.

    Definition is T_total(cold) - T_total(warm), matching the time-delta
    variant reported in per-cell output (the PTR variant lives
    in ``ptr_by_cell.csv``).
    """
    head = metrics[0]
    if head.cold_warm != "cold":
        row["delta_CW_ms"] = ""
        return
    warm_med = warm_total_median.get(
        (head.server, head.backend, head.query, head.scale_factor),
    )
    if warm_med is None:
        row["delta_CW_ms"] = ""
        return
    cold_med = float(np.median([m.t_total_ms for m in metrics]))
    row["delta_CW_ms"] = _format(cold_med - warm_med)


def _write_cpu_saturation(
    row: dict[str, str], metrics: Sequence[TraceMetrics],
    *, sf_to_cpu_pct: dict[tuple, dict[int, float]],
) -> None:
    head = metrics[0]
    sf_map = sf_to_cpu_pct.get(
        (head.server, head.backend, head.query, head.cold_warm),
    )
    if not sf_map:
        row["sigma_star_cpu_sf"] = ""
        return
    sigma = cm4_cpu_saturation_threshold(
        sf_map, threshold_pct=_CPU_SATURATION_THRESHOLD_PCT,
    )
    row["sigma_star_cpu_sf"] = "" if sigma is None else str(sigma)


def _write_backpressure(
    row: dict[str, str], metrics: Sequence[TraceMetrics],
    *, memory_limit_mb: float,
) -> None:
    peak_samples = [m.m_peak_mb for m in metrics if m.m_peak_mb is not None]
    if not peak_samples:
        row["backpressure_pattern"] = ""
        return
    pattern = cm6_backpressure_classification(
        stream_mode=None,
        requested_rows=0,
        returned_rows=0,
        interaction_status="OK",
        peak_memory_mb=float(np.max(peak_samples)),
        memory_limit_mb=memory_limit_mb,
    )
    row["backpressure_pattern"] = (
        pattern.value if isinstance(pattern, BackpressurePattern) else str(pattern)
    )


__all__ = ["build_sf_cpu_index", "build_warm_total_index", "summarize_cell"]
