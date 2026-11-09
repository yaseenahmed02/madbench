#!/usr/bin/env python
"""Run Track A through mcp-alchemy on PostgreSQL, MySQL and DuckDB.

Default matrix: 3 backends x 5 queries x 3 SFs x cold/warm x 30 reps
= 90 cells, 2,700 traces. No LLM is involved.

Outputs go under ``<output-root>/mcp-alchemy/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from mcp_dbs_bench.track_a.csv_writer import (
    write_per_cell_summary, write_per_trace_metrics,
    write_phase_breakdown, write_ptr_by_cell, write_token_metrics,
)
from mcp_dbs_bench.track_a.figures import render_all
from mcp_dbs_bench.track_a.phase_runner import PhaseConfig, run_phase
from mcp_dbs_bench.track_a.trace_loader import load_trace_metrics

_REPO_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
_LOG = logging.getLogger("track_a")

SERVER = "mcp-alchemy"
RUN_ID = SERVER
BACKENDS = ("postgres", "mysql", "duckdb")
QUERIES = ("Q1", "Q3", "Q6", "Q9", "Q21")
SCALE_FACTORS = (1, 3, 10)

# Per-SF namespaces.
SF_TO_NAMESPACE: dict[str, dict[int, str]] = {
    "postgres": {1: "sf1", 3: "sf3", 10: "sf10"},
    "mysql": {1: "tpch_sf1", 3: "tpch_sf3", 10: "tpch_sf10"},
    "duckdb": {
        1: "/tmp/madbench_duckdb_sf1.duckdb",
        3: "/tmp/madbench_duckdb_sf3.duckdb",
        10: "/tmp/madbench_duckdb_sf10.duckdb",
    },
}

# Base DSNs (namespace is applied per SF inside phase_runner).
BACKEND_DSNS = {
    "postgres": "postgresql://bench:bench@localhost:55432/tpch",
    "mysql": "mysql://bench:bench@localhost:53306/tpch",
    "duckdb": "/tmp/madbench_duckdb_sf1.duckdb",  # overridden via namespace map
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--cold-warm", default="cold,warm",
        help="Comma-separated subset of 'cold,warm'",
    )
    parser.add_argument(
        "--backends", default=",".join(BACKENDS),
        help="Comma-separated backends",
    )
    parser.add_argument(
        "--queries", default=",".join(QUERIES),
        help="Comma-separated TPC-H queries",
    )
    parser.add_argument(
        "--sfs", default="1,3,10",
        help="Comma-separated scale factors",
    )
    parser.add_argument(
        "--use-subprocess", action=argparse.BooleanOptionalAction, default=True,
        help="Run the upstream server as a subprocess (the measured mode)",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=_REPO_ROOT / "results" / "track_a",
    )
    parser.add_argument(
        "--skip-execution", action="store_true",
        help="Skip running cells; just aggregate existing traces into CSVs+figures",
    )
    parser.add_argument(
        "--skip-if-complete", action="store_true",
        help="Skip cells whose trace.jsonl already has at least one interaction",
    )
    args = parser.parse_args()

    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    queries = tuple(q.strip() for q in args.queries.split(",") if q.strip())
    sfs = tuple(int(x.strip()) for x in args.sfs.split(",") if x.strip())
    cold_warm = tuple(x.strip() for x in args.cold_warm.split(",") if x.strip())

    # DuckDB in-process fallback: the upstream mcp-alchemy binary depends on
    # duckdb-engine, which is currently incompatible with SQLAlchemy 2.x +
    # DuckDB 1.x (missing `set_isolation_level` on ConnectionWrapper). The
    # in-process plugin uses the same wire-format parsers and produces the
    # same trace shape; the only divergence is sub-microsecond T_xport.
    backend_overrides = {"duckdb": False} if args.use_subprocess else None

    config = PhaseConfig(
        run_id=RUN_ID, server=SERVER,
        backend_dsns=BACKEND_DSNS,
        backends=backends, queries=queries, scale_factors=sfs,
        reps=args.reps, warmup=args.warmup,
        cold_warm=cold_warm,
        output_root=args.output_root,
        use_subprocess=args.use_subprocess,
        backend_subprocess_overrides=backend_overrides,
        track="A", llm_budget_usd=10.0,
        sf_to_namespace=SF_TO_NAMESPACE,
        skip_if_complete=args.skip_if_complete,
    )

    started_at = time.time()
    if not args.skip_execution:
        _LOG.info(
            "starting run %s: %d cells",
            RUN_ID,
            len(cold_warm) * len(backends) * len(queries) * len(sfs),
        )
        report = run_phase(config)
        _LOG.info(
            "run %s done: %d cells executed, %d traces collected",
            RUN_ID, len(report.cells), report.total_traces_collected,
        )
    else:
        _LOG.info("skipping execution; aggregating existing traces only")

    aggregate_artifacts(config=config, cold_warm=cold_warm,
                        backends=backends, queries=queries, sfs=sfs)
    write_phase_summary(
        config=config, cold_warm=cold_warm, backends=backends,
        queries=queries, sfs=sfs,
        wall_clock_seconds=time.time() - started_at,
    )
    return 0


def aggregate_artifacts(
    *, config: PhaseConfig, cold_warm: tuple[str, ...],
    backends: tuple[str, ...], queries: tuple[str, ...], sfs: tuple[int, ...],
) -> None:
    """Load every cell's trace.jsonl under raw_traces/ and write CSV + figures.

    We scan the filesystem rather than iterate the args so partial reruns
    (e.g. just PG SF10) still produce CSVs that contain every cell that
    has been collected to date.
    """
    phase_root_path = config.output_root / SERVER
    raw_traces_dir = phase_root_path / "raw_traces"
    csv_dir = phase_root_path / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    cells_by_id = []
    if not raw_traces_dir.is_dir():
        _LOG.warning("raw_traces dir missing at %s", raw_traces_dir)
        return
    for cell_dir in sorted(raw_traces_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        tpath = cell_dir / "trace.jsonl"
        if not tpath.is_file() or tpath.stat().st_size == 0:
            _LOG.warning("skipping empty cell %s", cell_dir.name)
            continue
        spec_parts = _parse_cell_dir_name(cell_dir.name)
        if spec_parts is None:
            _LOG.warning("could not parse cell name %s", cell_dir.name)
            continue
        backend, query, sf, cw = spec_parts
        cell_metrics = list(load_trace_metrics(
            tpath, server=SERVER, backend=backend,
            query=query, scale_factor=sf, cold_warm=cw,
        ))
        if cell_metrics:
            cells_by_id.append(cell_metrics)
            all_metrics.extend(cell_metrics)
    _LOG.info(
        "aggregating: %d cells, %d total metrics",
        len(cells_by_id), len(all_metrics),
    )
    write_per_trace_metrics(csv_dir / "per_trace_metrics.csv", all_metrics)
    write_per_cell_summary(csv_dir / "per_cell_summary.csv", cells_by_id)
    write_ptr_by_cell(csv_dir / "ptr_by_cell.csv", cells_by_id)
    write_phase_breakdown(csv_dir / "phase_breakdown.csv", cells_by_id)
    write_token_metrics(csv_dir / "token_metrics.csv", cells_by_id)
    write_csv_readme(csv_dir / "README.md")
    fig_root = phase_root_path / "figures"
    out = render_all(
        run_id=RUN_ID, csv_dir=csv_dir,
        png_dir=fig_root / "png", pdf_dir=fig_root / "pdf",
    )
    _LOG.info("rendered %d figures", len(out))
    write_figure_source(fig_root / "source.py")


def _parse_cell_dir_name(name: str) -> tuple[str, str, int, str] | None:
    """Parse ``<server>_<backend>_<query>_sf<sf>_<cw>`` -> (backend, query, sf, cw).

    The server prefix is stripped because all cells share one server per run.
    """
    parts = name.split("_")
    if len(parts) < 5:
        return None
    cw = parts[-1]
    if cw not in ("cold", "warm"):
        return None
    sf_token = parts[-2]
    if not sf_token.startswith("sf"):
        return None
    try:
        sf = int(sf_token[2:])
    except ValueError:
        return None
    query = parts[-3]
    backend = parts[-4]
    return backend, query, sf, cw


def write_csv_readme(path: Path) -> None:
    path.write_text(
        "# Track A CSV dictionary\n\n"
        "## per_trace_metrics.csv\n"
        "One row per measured rep. Columns:\n"
        "- cell_id, server, backend, query, scale_factor, rep_idx, cold_warm: identification.\n"
        "- T_disc_ms / T_life_ms / T_hand_ms / T_db_ms / T_ser_ms / T_parse_ms / T_xport_ms: phase durations (ms).\n"
        "- T_total_ms: wall-clock interaction time.\n"
        "- T_teardown_ms: interval from the last parse span to the end of the interaction.\n"
        "- epsilon_ms: residual (T_total - T_db - sum of the named phases).\n"
        "- PTax_ms: protocol tax = T_total - T_db (all phases, teardown and residual).\n"
        "- PTR: PTax_ms / T_db_ms (empty when T_db <= 0).\n"
        "- B_mcp: bytes received over MCP wire.\n"
        "- tau_disc: discovery token count.\n"
        "- phi: token inflation factor (empty if not measured).\n"
        "- gamma: context decay (empty if not measured).\n"
        "- S_ser: serialization throughput (rows/ms).\n"
        "- rho: payload expansion ratio (empty if no B_native baseline).\n"
        "- n_life: lifecycle handshake count.\n"
        "- trace_path: source trace.jsonl path.\n\n"
        "## per_cell_summary.csv\n"
        "One row per cell. P50/P95 for T_db, T_total, PTax, PTR, rho_p50, phi_p50.\n"
        "PTR_ci_lo / PTR_ci_hi: 95% bootstrap CI of median PTR (10,000 resamples).\n\n"
        "## ptr_by_cell.csv\n"
        "Per (server, backend, query, sf) with cold and warm pivoted side by side.\n\n"
        "## phase_breakdown.csv\n"
        "Per-cell median phase durations for stacked-area plots.\n\n"
        "## token_metrics.csv\n"
        "Per-cell median tau_disc, B_mcp, phi.\n"
    )


def write_figure_source(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python\n"
        "\"\"\"Regenerate the Track A run figures from ../csv/.\"\"\"\n\n"
        "from pathlib import Path\n"
        "from mcp_dbs_bench.track_a.figures import render_all\n\n"
        "ROOT = Path(__file__).resolve().parent\n"
        "render_all(\n"
        "    run_id='mcp-alchemy',\n"
        "    csv_dir=ROOT.parent / 'csv',\n"
        "    png_dir=ROOT / 'png',\n"
        "    pdf_dir=ROOT / 'pdf',\n"
        ")\n"
    )


def write_phase_summary(
    *, config: PhaseConfig, cold_warm: tuple[str, ...],
    backends: tuple[str, ...], queries: tuple[str, ...], sfs: tuple[int, ...],
    wall_clock_seconds: float,
) -> None:
    """Synthesize summary.md from per_cell_summary + ptr_by_cell."""
    import csv as csv_mod
    phase_root_path = config.output_root / SERVER
    csv_dir = phase_root_path / "csv"
    per_cell = csv_dir / "per_cell_summary.csv"
    if not per_cell.is_file():
        return
    rows = list(csv_mod.DictReader(per_cell.open()))
    warm_rows = [r for r in rows if r["cold_warm"] == "warm"]
    ptrs = [
        (float(r["PTR_p50"]), r["backend"], r["query"], r["scale_factor"])
        for r in warm_rows if r["PTR_p50"] != ""
    ]
    most_overheaded = max(ptrs, default=(None, None, None, None))
    least_overheaded = min(ptrs, default=(None, None, None, None))
    ptr_low = min((p[0] for p in ptrs), default=None)
    ptr_high = max((p[0] for p in ptrs), default=None)
    n_cells = len(rows)
    n_traces = sum(int(r["n_reps"]) for r in rows if r["n_reps"])
    # Property 1: epsilon as fraction of t_total should be < 5%.
    # We do not yet have the per-trace epsilon% in per_cell_summary; report
    # cells from per_trace_metrics where epsilon/T_total > 0.05.
    per_trace_path = csv_dir / "per_trace_metrics.csv"
    eps_violations = 0
    eps_total_rows = 0
    if per_trace_path.is_file():
        for r in csv_mod.DictReader(per_trace_path.open()):
            try:
                eps = float(r["epsilon_ms"])
                t_total = float(r["T_total_ms"])
            except (KeyError, ValueError):
                continue
            if t_total <= 0:
                continue
            eps_total_rows += 1
            if abs(eps) / t_total > 0.05:
                eps_violations += 1
    eps_pass_rate = (
        1.0 - (eps_violations / eps_total_rows) if eps_total_rows > 0 else None
    )
    anomalies = [r for r in rows if r["n_reps"] == "0" or r["n_reps"] == ""]
    cells_summary_table = "\n".join(
        f"- {r['cell_id']}: n_reps={r['n_reps']}, "
        f"PTR_p50={r['PTR_p50'] or 'n/a'}, "
        f"T_db_p50={r['T_db_p50'] or 'n/a'}ms"
        for r in rows[:10]
    )
    summary = f"""# Track A summary: mcp-alchemy x {{PG, MySQL, DuckDB}}

## Configuration
- Track: A (no LLM)
- Server: {SERVER}
- Backends: {", ".join(backends)}
- Queries: {", ".join(queries)}
- Scale factors: {", ".join(str(s) for s in sfs)}
- Cold/warm: {", ".join(cold_warm)}
- Reps per cell: {config.reps}
- Warmup per cell: {config.warmup}
- Subprocess mode: {config.use_subprocess}
- Per-backend subprocess overrides: {config.backend_subprocess_overrides or {{}}}

## Results
- Cells: {n_cells}
- Traces collected: {n_traces}
- Wall-clock (this invocation): {wall_clock_seconds:.1f} s ({wall_clock_seconds/60:.1f} min)
- LLM cost: $0.00 (Track A)
- PTR_warm range: {ptr_low} .. {ptr_high}
- Property 1 (|epsilon|/T_total < 5%) pass rate: {eps_pass_rate}
  (violations: {eps_violations} / {eps_total_rows})

## Most-overheaded cell (warm PTR_p50)
{most_overheaded}

## Least-overheaded cell (warm PTR_p50)
{least_overheaded}

## First 10 cells (sample)
{cells_summary_table}

## Anomalies
"""
    if anomalies:
        for a in anomalies:
            summary += f"- {a['cell_id']}: 0 reps recorded\n"
    else:
        summary += "(none)\n"
    summary += """
## Takeaway
On mcp-alchemy, the costliest (backend, query) combination on warm reps is
"""
    summary += f"{most_overheaded[1]}/{most_overheaded[2]} at SF{most_overheaded[3]} (PTR={most_overheaded[0]}).\n"
    (phase_root_path / "summary.md").write_text(summary)


if __name__ == "__main__":
    sys.exit(main())
