"""CSV writers for Track A per-trace + per-cell-summary artifacts.

All writers use the standard ``csv`` module for proper field escaping —
do not switch to f-string formatting.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from mcp_dbs_bench.analysis.bootstrap import N_BOOTSTRAP_RESAMPLES
from mcp_dbs_bench.track_a._cell_summary import (
    build_sf_cpu_index,
    build_warm_total_index,
    summarize_cell,
)
from mcp_dbs_bench.track_a.trace_loader import TraceMetrics

PER_TRACE_COLUMNS: tuple[str, ...] = (
    "cell_id", "server", "backend", "query", "scale_factor",
    "rep_idx", "cold_warm",
    "T_disc_ms", "T_life_ms", "T_hand_ms", "T_db_ms",
    "T_ser_ms", "T_parse_ms", "T_xport_ms", "T_teardown_ms", "T_total_ms",
    "epsilon_ms", "PTax_ms", "PTR",
    "B_mcp", "tau_disc", "phi", "gamma",
    "S_ser", "rho", "n_life",
    "R_retry_ms", "T_pool_ms", "M_peak_mb",
    "trace_path",
)

PER_CELL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "cell_id", "server", "backend", "query", "scale_factor", "cold_warm",
    "n_reps",
    "T_db_p50", "T_db_p95", "T_total_p50", "T_total_p95",
    "PTax_p50", "PTax_p95",
    "PTR_p50", "PTR_p95", "PTR_ci_lo", "PTR_ci_hi",
    "rho_p50", "phi_p50",
    "M_peak_mb_p50", "M_peak_mb_p95",
    "delta_CW_ms",
    "sigma_star_cpu_sf",
    "backpressure_pattern",
)

_DEFAULT_MEMORY_LIMIT_MB = 16 * 1024.0

PTR_BY_CELL_COLUMNS: tuple[str, ...] = (
    "server", "backend", "query", "scale_factor",
    "PTR_cold", "PTR_warm", "delta_CW",
)

PHASE_BREAKDOWN_COLUMNS: tuple[str, ...] = (
    "cell_id", "server", "backend", "query", "scale_factor", "cold_warm",
    "T_disc", "T_life", "T_hand", "T_db", "T_ser", "T_parse", "T_xport",
)

TOKEN_METRICS_COLUMNS: tuple[str, ...] = (
    "cell_id", "server", "backend", "query", "scale_factor", "cold_warm",
    "tau_disc", "B_mcp", "phi",
)


def _format(value: float | int | None) -> str:
    """Empty string for None; full-precision repr for floats."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def write_per_trace_metrics(path: Path, metrics: Iterable[TraceMetrics]) -> None:
    """Write one row per TraceMetrics. Creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PER_TRACE_COLUMNS))
        writer.writeheader()
        for m in metrics:
            writer.writerow({
                "cell_id": m.cell_id, "server": m.server, "backend": m.backend,
                "query": m.query, "scale_factor": m.scale_factor,
                "rep_idx": m.rep_idx, "cold_warm": m.cold_warm,
                "T_disc_ms": _format(m.t_disc_ms),
                "T_life_ms": _format(m.t_life_ms),
                "T_hand_ms": _format(m.t_hand_ms),
                "T_db_ms": _format(m.t_db_ms),
                "T_ser_ms": _format(m.t_ser_ms),
                "T_parse_ms": _format(m.t_parse_ms),
                "T_xport_ms": _format(m.t_xport_excl_ms),
                "T_teardown_ms": _format(m.t_teardown_ms),
                "T_total_ms": _format(m.t_total_ms),
                "epsilon_ms": _format(m.epsilon_ms),
                "PTax_ms": _format(m.ptax_ms),
                "PTR": _format(m.ptr),
                "B_mcp": _format(m.b_mcp),
                "tau_disc": _format(m.tau_disc),
                "phi": _format(m.phi),
                "gamma": _format(m.gamma),
                "S_ser": _format(m.s_ser_rows_per_ms),
                "rho": _format(m.rho),
                "n_life": _format(m.n_life),
                "R_retry_ms": _format(m.r_retry_ms),
                "T_pool_ms": _format(m.t_pool_ms),
                "M_peak_mb": _format(m.m_peak_mb),
                "trace_path": m.trace_path,
            })


def write_per_cell_summary(
    path: Path,
    cells: Sequence[Sequence[TraceMetrics]],
    *,
    n_bootstrap: int = N_BOOTSTRAP_RESAMPLES,
    memory_limit_mb: float = _DEFAULT_MEMORY_LIMIT_MB,
) -> None:
    """Write one row per cell with P5/P50/P95 + bootstrap CI for PTR.

    Each element of ``cells`` is a list of TraceMetrics for one cell.
    δ_CW (IM-6) and σ*_CPU (CM-4) are cross-cell derived metrics — they
    need warm-cell + per-SF context, so the writer scans all cells first
    to build the lookup maps.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    warm_total_median = build_warm_total_index(cells)
    sf_to_cpu_pct = build_sf_cpu_index(cells)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PER_CELL_SUMMARY_COLUMNS))
        writer.writeheader()
        for cell_metrics in cells:
            row = summarize_cell(
                cell_metrics,
                n_bootstrap=n_bootstrap,
                warm_total_median=warm_total_median,
                sf_to_cpu_pct=sf_to_cpu_pct,
                memory_limit_mb=memory_limit_mb,
                columns=PER_CELL_SUMMARY_COLUMNS,
            )
            writer.writerow(row)


def write_ptr_by_cell(
    path: Path, cells: Sequence[Sequence[TraceMetrics]],
) -> None:
    """Pivot cold/warm cells into one row per (server, backend, query, sf)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot: dict[tuple, dict] = {}
    for cell_metrics in cells:
        if not cell_metrics:
            continue
        head = cell_metrics[0]
        key = (head.server, head.backend, head.query, head.scale_factor)
        pivot.setdefault(key, {
            "server": head.server, "backend": head.backend,
            "query": head.query, "scale_factor": head.scale_factor,
            "PTR_cold": None, "PTR_warm": None,
        })
        ptrs = [m.ptr for m in cell_metrics if m.ptr is not None]
        if not ptrs:
            continue
        median_ptr = float(np.median(ptrs))
        if head.cold_warm == "cold":
            pivot[key]["PTR_cold"] = median_ptr
        else:
            pivot[key]["PTR_warm"] = median_ptr
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PTR_BY_CELL_COLUMNS))
        writer.writeheader()
        for entry in pivot.values():
            cold, warm = entry["PTR_cold"], entry["PTR_warm"]
            delta = (cold - warm) if (cold is not None and warm is not None) else None
            writer.writerow({
                "server": entry["server"], "backend": entry["backend"],
                "query": entry["query"], "scale_factor": entry["scale_factor"],
                "PTR_cold": _format(cold), "PTR_warm": _format(warm),
                "delta_CW": _format(delta),
            })


def write_phase_breakdown(
    path: Path, cells: Sequence[Sequence[TraceMetrics]],
) -> None:
    """Median phase durations per cell, suitable for stacked-area plots."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PHASE_BREAKDOWN_COLUMNS))
        writer.writeheader()
        for cell_metrics in cells:
            if not cell_metrics:
                continue
            head = cell_metrics[0]
            row = {
                "cell_id": head.cell_id, "server": head.server,
                "backend": head.backend, "query": head.query,
                "scale_factor": head.scale_factor, "cold_warm": head.cold_warm,
                "T_disc": _format(float(np.median([m.t_disc_ms for m in cell_metrics]))),
                "T_life": _format(float(np.median([m.t_life_ms for m in cell_metrics]))),
                "T_hand": _format(float(np.median([m.t_hand_ms for m in cell_metrics]))),
                "T_db": _format(float(np.median([m.t_db_ms for m in cell_metrics]))),
                "T_ser": _format(float(np.median([m.t_ser_ms for m in cell_metrics]))),
                "T_parse": _format(float(np.median([m.t_parse_ms for m in cell_metrics]))),
                "T_xport": _format(float(np.median([m.t_xport_excl_ms for m in cell_metrics]))),
            }
            writer.writerow(row)


def write_token_metrics(
    path: Path, cells: Sequence[Sequence[TraceMetrics]],
) -> None:
    """Median tau_disc / B_mcp / phi per cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TOKEN_METRICS_COLUMNS))
        writer.writeheader()
        for cell_metrics in cells:
            if not cell_metrics:
                continue
            head = cell_metrics[0]
            phi_vals = [m.phi for m in cell_metrics if m.phi is not None]
            row = {
                "cell_id": head.cell_id, "server": head.server,
                "backend": head.backend, "query": head.query,
                "scale_factor": head.scale_factor, "cold_warm": head.cold_warm,
                "tau_disc": _format(float(np.median([m.tau_disc for m in cell_metrics]))),
                "B_mcp": _format(float(np.median([m.b_mcp for m in cell_metrics]))),
                "phi": _format(float(np.median(phi_vals))) if phi_vals else "",
            }
            writer.writerow(row)


__all__ = [
    "PER_CELL_SUMMARY_COLUMNS", "PER_TRACE_COLUMNS",
    "PHASE_BREAKDOWN_COLUMNS", "PTR_BY_CELL_COLUMNS", "TOKEN_METRICS_COLUMNS",
    "write_per_cell_summary", "write_per_trace_metrics",
    "write_phase_breakdown", "write_ptr_by_cell", "write_token_metrics",
]
