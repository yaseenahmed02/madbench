#!/usr/bin/env python
"""Track B sweep runner.

Executes the full (query × backend × rep) grid for one MCP server. Uses
OpenRouter (DeepSeek v4 Flash by default) via the existing agent loop,
verifies correctness against captured gold results, and enforces a hard
USD budget cap using OpenRouter's authoritative usage counters.

Usage
-----
    python scripts/track_b/run_track_b_sweep.py \\
        --server mcp-alchemy \\
        --backends postgres,mysql,duckdb \\
        --query-file workloads/track_b/questions.yaml \\
        --reps 3 --sf 1 --paired-cache \\
        --budget-usd 8.0 \\
        --output-root results/track_b/deepseek-v4-flash/mcp-alchemy/sf1

Environment
-----------
    OPENROUTER_API_KEY   required — bearer token
    OPENROUTER_LLM_MODEL optional — defaults to deepseek/deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from mcp_dbs_bench.agent.openrouter_llm import OpenRouterLLMClient  # noqa: E402

from track_b import _paths  # noqa: E402
from track_b._attempt import run_one_attempt  # noqa: E402
from track_b._backends import SUPPORTED_BACKENDS, connect  # noqa: E402
from track_b._cost import BudgetExceeded, OpenRouterCostTracker  # noqa: E402
from track_b._csv_io import PerTraceWriter, write_per_cell_summary  # noqa: E402
from track_b._mcp_launch import SUPPORTED_SERVERS, build_launch_spec_for  # noqa: E402
from track_b._metrics import TrackBMetrics  # noqa: E402
from track_b._paired_cache import restart_postgres_wait_healthy  # noqa: E402
from track_b._workload import WorkloadError, load_workload  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
_LOG = logging.getLogger("track_b.sweep")


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)
    api_key = _resolve_api_key()
    queries = _load_queries(args.query_file, args.query_ids)
    backends = _parse_backends(args.backends)
    output_root = Path(args.output_root)
    cost_tracker = OpenRouterCostTracker(
        api_key=api_key, budget_usd=args.budget_usd,
    )
    cost_tracker.initialise_baseline()
    _write_manifest(
        output_root=output_root, server=args.server, args=args,
        queries=queries, backends=backends, baseline=cost_tracker.session_baseline_usd,
    )
    all_rows = asyncio.run(_run_sweep(
        args=args, api_key=api_key, queries=queries,
        backends=backends, output_root=output_root,
        cost_tracker=cost_tracker,
    ))
    _emit_summary(output_root=output_root, server=args.server, rows=all_rows)
    _emit_budget(output_root=output_root, server=args.server, tracker=cost_tracker)
    _LOG.info("sweep complete: rows=%d", len(all_rows))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server", default="mcp-alchemy", choices=list(SUPPORTED_SERVERS),
        help="Which MCP server upstream to spawn per attempt.",
    )
    parser.add_argument("--backends", default="postgres,mysql,duckdb")
    parser.add_argument(
        "--query-file", nargs="+", default=[str(_paths.query_file())],
        help="one or more workload YAML files",
    )
    parser.add_argument("--query-ids", default="",
                        help="Optional CSV of query ids to filter (e.g. tb_c2_01,tb_c2_06)")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--sf", type=int, default=1)
    parser.add_argument("--paired-cache", action="store_true",
                        help="Restart Postgres before each PG attempt")
    parser.add_argument("--budget-usd", type=float, default=8.0)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--model", default=os.environ.get(
        "OPENROUTER_LLM_MODEL") or "deepseek/deepseek-v4-flash")
    parser.add_argument("--output-root", default=str(_paths.default_output_root()))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.getLogger().setLevel(level.upper())


def _resolve_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY must be set in the environment")
    return api_key


def _parse_backends(spec: str) -> list[str]:
    items = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [x for x in items if x not in SUPPORTED_BACKENDS]
    if unknown:
        raise SystemExit(
            f"unknown backend(s): {unknown!r}; expected any of {SUPPORTED_BACKENDS!r}"
        )
    return items


def _load_queries(paths: list[str], filter_csv: str) -> list[dict]:
    """Load one or more workload YAMLs and optionally filter to query ids."""
    try:
        all_queries = load_workload([Path(p) for p in paths])
    except WorkloadError as exc:
        raise SystemExit(str(exc)) from exc
    if not filter_csv:
        return all_queries
    wanted = {x.strip() for x in filter_csv.split(",") if x.strip()}
    filtered = [q for q in all_queries if q["id"] in wanted]
    if not filtered:
        raise SystemExit(f"filter {wanted!r} matched no queries in {paths}")
    return filtered


async def _run_sweep(
    *,
    args: argparse.Namespace,
    api_key: str,
    queries: list[dict],
    backends: list[str],
    output_root: Path,
    cost_tracker: OpenRouterCostTracker,
) -> list[TrackBMetrics]:
    """Iterate (backend, query, rep), collect rows, halt on budget breach."""
    llm_client = OpenRouterLLMClient(api_key=api_key, model=args.model)
    per_trace_path = _paths.per_trace_csv(output_root, server=args.server)
    events_path = _paths.events_log(output_root, server=args.server)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(queries) * len(backends) * args.reps
    all_rows: list[TrackBMetrics] = []
    with PerTraceWriter(per_trace_path) as writer, events_path.open("a") as events:
        _log_event(events, "sweep_start", total=total)
        completed = 0
        for backend in backends:
            for query in queries:
                cell_rows = await _run_cell(
                    args=args, query=query, backend=backend,
                    llm_client=llm_client, output_root=output_root,
                    cost_tracker=cost_tracker, events_fh=events,
                )
                for row in cell_rows:
                    writer.write(row)
                    all_rows.append(row)
                    completed += 1
                    if completed % 10 == 0:
                        _log_progress(
                            completed=completed, total=total,
                            rows=all_rows, cost_tracker=cost_tracker,
                            events=events,
                        )
                if cost_tracker.samples and _budget_check_or_stop(
                    tracker=cost_tracker, events=events,
                ):
                    return all_rows
    return all_rows


async def _run_cell(
    *,
    args: argparse.Namespace,
    query: dict,
    backend: str,
    llm_client: OpenRouterLLMClient,
    output_root: Path,
    cost_tracker: OpenRouterCostTracker,
    events_fh,
) -> list[TrackBMetrics]:
    """Run all reps for one (query, backend). Returns the metrics rows."""
    cell_id = f"{args.server}_{backend}_{query['id']}_sf{args.sf}"
    trace_dir = _paths.cell_trace_dir(
        output_root, server=args.server, cell_id=cell_id,
    )
    launch = build_launch_spec_for(
        server=args.server, backend=backend, scale_factor=args.sf,
    )
    rows: list[TrackBMetrics] = []
    for rep_idx in range(args.reps):
        if args.paired_cache and backend == "postgres":
            try:
                restart_postgres_wait_healthy()
            except Exception as exc:
                _LOG.warning("paired-cache restart failed: %s", exc)
        row = await _run_one_rep(
            args=args, query=query, backend=backend, rep_idx=rep_idx,
            launch=launch, llm_client=llm_client,
            trace_dir=trace_dir, cost_tracker=cost_tracker,
        )
        _log_event(events_fh, "rep_complete", cell_id=cell_id, rep_idx=rep_idx,
               outcome=row.outcome, wall_ms=row.wall_clock_ms,
               tokens_in=row.total_input_tokens,
               tokens_out=row.total_output_tokens)
        rows.append(row)
    return rows


async def _run_one_rep(
    *,
    args: argparse.Namespace,
    query: dict,
    backend: str,
    rep_idx: int,
    launch,
    llm_client: OpenRouterLLMClient,
    trace_dir: Path,
    cost_tracker: OpenRouterCostTracker,
) -> TrackBMetrics:
    """Run one rep + attribute cost to it."""
    sql_client_cm = _sql_executor_for(backend=backend, sf=args.sf)
    with sql_client_cm as sql_client:
        row = await run_one_attempt(
            query=query, backend=backend, scale_factor=args.sf,
            rep_idx=rep_idx, server=args.server,
            launch=launch, llm_client=llm_client,
            max_tool_calls=args.max_tool_calls,
            max_retries=args.max_retries,
            trace_dir=trace_dir, sql_executor=sql_client,
        )
    _attribute_cost(cost_tracker=cost_tracker, row=row)
    return row


def _sql_executor_for(*, backend: str, sf: int):
    """Context manager yielding a BackendClient for verifier SQL execution.

    Best-effort: on connect failure we yield None so verification proceeds
    without SQL execution (agent SQL answers -> unparseable).

    Special case for DuckDB: the mcp-alchemy subprocess opens the DuckDB
    file with a per-process file lock (DuckDB is single-writer). Holding a
    verifier connection open concurrently causes ``McpError: Connection
    closed`` in the child. We skip the verifier SQL executor for DuckDB
    entirely; agents returning SQL-shape answers on DuckDB are marked
    ``unparseable``. Scalar / list answers are unaffected — those are the
    common shape for our C2/C5 workload (COUNT / SUM / AVG / row lists).
    """
    from contextlib import nullcontext
    if backend == "duckdb":
        _LOG.info(
            "DuckDB: skipping verifier SQL executor to avoid file-lock "
            "conflict with mcp-alchemy child; SQL-shape agent answers on "
            "DuckDB will verify as unparseable"
        )
        return nullcontext(None)
    try:
        return connect(backend, sf)
    except Exception as exc:
        _LOG.warning("sql executor connect failed: %s", exc)
        return nullcontext(None)


def _attribute_cost(
    *, cost_tracker: OpenRouterCostTracker, row: TrackBMetrics,
) -> None:
    """Record cost for one attempt using the baseline+delta method.

    We do not know per-generation ids without extra API calls, so we
    attribute cost by polling ``/key`` and diffing against the samples
    already accumulated in this session.
    """
    sample = cost_tracker.record_generation(
        cell_id=row.cell_id, rep_idx=row.rep_idx, generation_id="",
        input_tokens=row.total_input_tokens,
        output_tokens=row.total_output_tokens,
    )
    row.cost_usd = sample.cost_usd


def _budget_check_or_stop(
    *, tracker: OpenRouterCostTracker, events,
) -> bool:
    """Poll OpenRouter and return True (stop sweep) if we are over budget."""
    try:
        delta = tracker.check_or_raise()
    except BudgetExceeded as exc:
        _LOG.error("BUDGET EXCEEDED: %s", exc)
        _log_event(events, "budget_exceeded", message=str(exc))
        return True
    _log_event(events, "budget_ok", spent_usd=delta)
    return False


def _log_progress(
    *,
    completed: int,
    total: int,
    rows: list[TrackBMetrics],
    cost_tracker: OpenRouterCostTracker,
    events,
) -> None:
    n_correct = sum(1 for r in rows if r.outcome == "correct")
    n_incorrect = sum(1 for r in rows if r.outcome == "incorrect")
    n_error = len(rows) - n_correct - n_incorrect
    _LOG.info(
        "progress %d/%d correct=%d incorrect=%d error=%d local_cost=$%.4f",
        completed, total, n_correct, n_incorrect, n_error,
        cost_tracker.cumulative_local_cost(),
    )
    _log_event(
        events, "progress", completed=completed, total=total,
        n_correct=n_correct, n_incorrect=n_incorrect, n_error=n_error,
        local_cost_usd=cost_tracker.cumulative_local_cost(),
    )


def _write_manifest(
    *, output_root: Path, server: str, args: argparse.Namespace,
    queries: list[dict], backends: list[str], baseline: float,
) -> None:
    path = _paths.manifest_json(output_root, server=server)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "track": "B",
        "server": server,
        "model": args.model,
        "openrouter_baseline_usd": baseline,
        "budget_usd": args.budget_usd,
        "backends": backends,
        "queries": [q["id"] for q in queries],
        "reps": args.reps,
        "sf": args.sf,
        "paired_cache": args.paired_cache,
        "max_tool_calls": args.max_tool_calls,
        "max_retries": args.max_retries,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(manifest, indent=2))


def _emit_summary(
    *, output_root: Path, server: str, rows: list[TrackBMetrics],
) -> None:
    path = _paths.per_cell_csv(output_root, server=server)
    write_per_cell_summary(path, rows)
    _LOG.info("wrote per-cell summary -> %s", path)


def _emit_budget(
    *, output_root: Path, server: str, tracker: OpenRouterCostTracker,
) -> None:
    path = _paths.budget_json(output_root, server=server)
    payload = {
        "openrouter_baseline_usd": tracker.session_baseline_usd,
        "budget_usd": tracker.budget_usd,
        "local_cost_estimate_usd": tracker.cumulative_local_cost(),
        "n_samples": len(tracker.samples),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))


def _log_event(fh, event: str, **fields) -> None:
    """Emit one JSON line to run_events.log."""
    line = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    fh.write(json.dumps(line, default=str) + "\n")
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except (OSError, AttributeError):
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _LOG.warning("interrupted by user")
        sys.exit(130)
