#!/usr/bin/env python
"""Run Track A through anthropic-pg (@modelcontextprotocol/server-postgres) on PostgreSQL.

Default matrix: 1 backend x 5 queries x 3 SFs x cold/warm x 30 reps
= 30 cells, 900 traces. No LLM is involved.

Outputs go under ``<output-root>/anthropic-pg/``.
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

SERVER = "anthropic-pg"
RUN_ID = SERVER
BACKENDS = ("postgres",)
QUERIES = ("Q1", "Q3", "Q6", "Q9", "Q21")
SCALE_FACTORS = (1, 3, 10)

SF_TO_NAMESPACE: dict[str, dict[int, str]] = {
    "postgres": {1: "sf1", 3: "sf3", 10: "sf10"},
}

BACKEND_DSNS = {
    "postgres": "postgresql://bench:bench@localhost:55432/tpch",
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

    config = PhaseConfig(
        run_id=RUN_ID, server=SERVER,
        backend_dsns=BACKEND_DSNS,
        backends=backends, queries=queries, scale_factors=sfs,
        reps=args.reps, warmup=args.warmup,
        cold_warm=cold_warm,
        output_root=args.output_root,
        use_subprocess=args.use_subprocess,
        backend_subprocess_overrides=None,
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
    """Load every cell's trace.jsonl under raw_traces/ and write CSV + figures."""
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


def _parse_cell_dir_name(name: str) -> tuple[str, str, int, str] | None:
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
        "Server: anthropic-pg (@modelcontextprotocol/server-postgres)\n\n"
        "Columns match the mcp-alchemy run; see the README data dictionary.\n"
    )


def write_phase_summary(
    *, config: PhaseConfig, cold_warm: tuple[str, ...],
    backends: tuple[str, ...], queries: tuple[str, ...], sfs: tuple[int, ...],
    wall_clock_seconds: float,
) -> None:
    phase_root_path = config.output_root / SERVER
    summary_path = phase_root_path / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    expected_cells = len(backends) * len(queries) * len(sfs) * len(cold_warm)
    raw_dir = phase_root_path / "raw_traces"
    actual_cells = (
        sum(1 for d in raw_dir.iterdir() if (d / "trace.jsonl").is_file())
        if raw_dir.is_dir() else 0
    )
    summary_path.write_text(
        f"# Track A summary\n\n"
        f"- Server: anthropic-pg (@modelcontextprotocol/server-postgres)\n"
        f"- Backends: {', '.join(backends)}\n"
        f"- Expected cells: {expected_cells}\n"
        f"- Cells finalized: {actual_cells}\n"
        f"- Wall-clock: {wall_clock_seconds:.0f}s\n"
        f"- Reps per cell: {config.reps}\n"
        f"- Warmups: {config.warmup}\n\n"
        f"See csv/per_cell_summary.csv for cell-level statistics.\n"
        f"See figures/ for PTR-by-cell, phase-dominance, cold/warm, power-law charts.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
