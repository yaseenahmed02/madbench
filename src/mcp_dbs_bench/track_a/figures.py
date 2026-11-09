"""Per-run figure renderers for Track A.

All four figures consume the CSV artifacts that the framework writes under
``results/track_a/<server>/csv/``. Outputs go to ``figures/png``
(300 dpi) and ``figures/pdf`` (vector). Single shared style.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

_LOG = logging.getLogger(__name__)

_FIG_DPI: int = 300


def _import_matplotlib():
    """Import matplotlib lazily so unit tests that don't render don't pay the cost."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _save(plt, fig, *, name: str, png_dir: Path, pdf_dir: Path) -> tuple[Path, Path]:
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{name}.png"
    pdf_path = pdf_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=_FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def render_ptr_by_cell(
    *, per_cell_csv: Path, run_id: str,
    png_dir: Path, pdf_dir: Path,
) -> tuple[Path, Path]:
    """Stacked bar: PTR by (backend, query) at SF=1 only."""
    _, plt = _import_matplotlib()
    rows = [
        r for r in _read_csv(per_cell_csv)
        if r["scale_factor"] == "1" and r["cold_warm"] == "warm"
    ]
    rows.sort(key=lambda r: (r["backend"], r["query"]))
    labels = [f"{r['backend'][:2]}/{r['query']}" for r in rows]
    ptrs = [_safe_float(r["PTR_p50"]) or 0.0 for r in rows]
    ci_lo = [_safe_float(r["PTR_ci_lo"]) or 0.0 for r in rows]
    ci_hi = [_safe_float(r["PTR_ci_hi"]) or 0.0 for r in rows]
    err_lo = [max(0.0, p - lo) for p, lo in zip(ptrs, ci_lo)]
    err_hi = [max(0.0, hi - p) for p, hi in zip(ptrs, ci_hi)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, ptrs, yerr=[err_lo, err_hi], capsize=3, color="#4477AA")
    ax.set_ylabel("PTR (warm, SF=1)")
    ax.set_xlabel("backend / query")
    ax.set_title(f"{run_id}: Protocol Tax Ratio by cell")
    ax.tick_params(axis="x", rotation=45)
    return _save(plt, fig, name=f"{run_id}_ptr_by_cell",
                 png_dir=png_dir, pdf_dir=pdf_dir)


def render_phase_dominance(
    *, phase_breakdown_csv: Path, run_id: str,
    png_dir: Path, pdf_dir: Path,
) -> tuple[Path, Path]:
    """Stacked area showing T_disc / T_life / T_hand / T_db / T_ser / T_parse / T_xport fraction by SF.

    Aggregates across backends + queries by median per SF.
    """
    _, plt = _import_matplotlib()
    rows = [r for r in _read_csv(phase_breakdown_csv) if r["cold_warm"] == "warm"]
    by_sf: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    components = ("T_disc", "T_life", "T_hand", "T_db", "T_ser", "T_parse", "T_xport")
    for r in rows:
        sf = int(r["scale_factor"])
        for comp in components:
            value = _safe_float(r[comp])
            if value is not None:
                by_sf[sf][comp].append(value)
    sfs_sorted = sorted(by_sf)
    medians = {
        comp: [
            (sum(by_sf[sf][comp]) / max(1, len(by_sf[sf][comp])))
            for sf in sfs_sorted
        ]
        for comp in components
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(
        sfs_sorted, *[medians[c] for c in components],
        labels=list(components), alpha=0.85,
    )
    ax.set_xlabel("Scale Factor")
    ax.set_ylabel("median time per phase (ms)")
    ax.set_title(f"{run_id}: Phase dominance by SF (warm)")
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    ax.set_xticks(sfs_sorted)
    return _save(plt, fig, name=f"{run_id}_phase_dominance",
                 png_dir=png_dir, pdf_dir=pdf_dir)


def render_cold_warm_delta(
    *, ptr_by_cell_csv: Path, run_id: str,
    png_dir: Path, pdf_dir: Path,
) -> tuple[Path, Path]:
    """Bar chart of delta_CW per (backend, query) — cold minus warm PTR."""
    _, plt = _import_matplotlib()
    rows = [r for r in _read_csv(ptr_by_cell_csv) if r["scale_factor"] == "1"]
    rows.sort(key=lambda r: (r["backend"], r["query"]))
    labels = [f"{r['backend'][:2]}/{r['query']}" for r in rows]
    deltas = [_safe_float(r["delta_CW"]) or 0.0 for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#CC3311" if d > 0 else "#228833" for d in deltas]
    ax.bar(labels, deltas, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_ylabel("delta_CW = PTR_cold - PTR_warm")
    ax.set_xlabel("backend / query")
    ax.set_title(f"{run_id}: Cold/Warm PTR delta (SF=1)")
    ax.tick_params(axis="x", rotation=45)
    return _save(plt, fig, name=f"{run_id}_cold_warm_delta",
                 png_dir=png_dir, pdf_dir=pdf_dir)


def render_power_law_per_query(
    *, per_cell_csv: Path, run_id: str,
    png_dir: Path, pdf_dir: Path,
) -> tuple[Path, Path]:
    """Log-log scatter of PTR vs SF per query with power-law fit lines."""
    import math
    _, plt = _import_matplotlib()
    rows = [r for r in _read_csv(per_cell_csv) if r["cold_warm"] == "warm"]
    by_query: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        ptr = _safe_float(r["PTR_p50"])
        if ptr is None:
            continue
        by_query[r["query"]][int(r["scale_factor"])].append(ptr)
    fig, ax = plt.subplots(figsize=(8, 6))
    for query, sf_to_ptrs in sorted(by_query.items()):
        xs = sorted(sf_to_ptrs.keys())
        ys = [sum(sf_to_ptrs[x]) / len(sf_to_ptrs[x]) for x in xs]
        ax.plot(xs, ys, "o-", label=query, markersize=8)
        if len(xs) >= 2 and all(y > 0 for y in ys):
            log_xs = [math.log(x) for x in xs]
            log_ys = [math.log(y) for y in ys]
            n = len(log_xs)
            x_mean = sum(log_xs) / n
            y_mean = sum(log_ys) / n
            num = sum((lx - x_mean) * (ly - y_mean) for lx, ly in zip(log_xs, log_ys))
            den = sum((lx - x_mean) ** 2 for lx in log_xs)
            slope = num / den if den != 0 else 0.0
            ax.text(
                xs[-1], ys[-1], f" k~={slope:.2f}",
                va="center", fontsize=8,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Scale Factor")
    ax.set_ylabel("PTR (warm median)")
    ax.set_title(f"{run_id}: Power-law: PTR vs SF (per query)")
    if by_query:
        ax.legend(loc="best", fontsize=9)
    return _save(plt, fig, name=f"{run_id}_power_law_per_query",
                 png_dir=png_dir, pdf_dir=pdf_dir)


def render_all(
    *, run_id: str, csv_dir: Path, png_dir: Path, pdf_dir: Path,
) -> dict[str, tuple[Path, Path]]:
    """Render all four phase figures from CSVs in ``csv_dir``."""
    out: dict[str, tuple[Path, Path]] = {}
    out["ptr_by_cell"] = render_ptr_by_cell(
        per_cell_csv=csv_dir / "per_cell_summary.csv",
        run_id=run_id, png_dir=png_dir, pdf_dir=pdf_dir,
    )
    out["phase_dominance"] = render_phase_dominance(
        phase_breakdown_csv=csv_dir / "phase_breakdown.csv",
        run_id=run_id, png_dir=png_dir, pdf_dir=pdf_dir,
    )
    out["cold_warm_delta"] = render_cold_warm_delta(
        ptr_by_cell_csv=csv_dir / "ptr_by_cell.csv",
        run_id=run_id, png_dir=png_dir, pdf_dir=pdf_dir,
    )
    out["power_law"] = render_power_law_per_query(
        per_cell_csv=csv_dir / "per_cell_summary.csv",
        run_id=run_id, png_dir=png_dir, pdf_dir=pdf_dir,
    )
    return out


__all__ = [
    "render_all", "render_cold_warm_delta", "render_phase_dominance",
    "render_power_law_per_query", "render_ptr_by_cell",
]
