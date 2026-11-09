"""Phase runner: iterates cells, drives TrackARunner, organizes outputs.

Track A (no LLM) is the default; ``llm_budget_usd`` sets the cap recorded
in ``budget.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from mcp_dbs_bench.core.registry import load_backend
from mcp_dbs_bench.harness.runner import TrackARunner
from mcp_dbs_bench.instrumentation.tracer import OTelTracer
from mcp_dbs_bench.track_a.budget_tracker import BudgetTracker, DEFAULT_CAP_USD
from mcp_dbs_bench.track_a.organize import CellSpec, cell_id, phase_root, trace_dir
from mcp_dbs_bench.workloads.tpch import load_query

_LOG = logging.getLogger(__name__)

_DEFAULT_TOOL_NAMES = {
    "anthropic-pg": "query",
    "dbhub": "execute_sql",
    "mcp-alchemy": "execute_query",
    "direct": "execute_sql",
}


@dataclass(frozen=True, slots=True)
class CellOutcome:
    """Result of running one (server, backend, query, sf, cold/warm) cell."""

    cell_id: str
    success: bool
    trace_path: Path
    reps_completed: int
    error_message: str | None


@dataclass
class PhaseConfig:
    """All inputs the phase runner needs.

    Attributes:
        run_id: Run identifier, e.g. the server name.
        server: Server plugin name (one run = one server).
        backend_dsns: Mapping ``backend -> DSN`` (only required for cold reps
            with backends that need a connection string).
        backends: Backends to iterate.
        queries: TPC-H query ids.
        scale_factors: TPC-H scale factors.
        reps: Measured iterations per cell.
        warmup: Discarded warmup iterations per cell (cold reps still count).
        cold_warm: Variants to run; default both cold and warm.
        output_root: Root directory; one folder per server is created underneath.
        use_subprocess: True to run the upstream server as a subprocess (the measured mode).
        llm_budget_usd: Cap for Track B; ignored for Track A.
        track: ``"A"`` or ``"B"``.
        seed: Random seed (recorded in manifest; no shuffling unless extended).
        sf_to_namespace: Optional per-SF DSN namespace map. For Postgres we
            use schemas (``sf1``/``sf3``/``sf10``); for MySQL separate
            databases (``tpch_sf1`` etc); for DuckDB separate files.
    """

    run_id: str
    server: str
    backend_dsns: dict[str, str] = field(default_factory=dict)
    backends: Sequence[str] = ()
    queries: Sequence[str] = ()
    scale_factors: Sequence[int] = ()
    reps: int = 30
    warmup: int = 5
    cold_warm: tuple[str, ...] = ("cold", "warm")
    output_root: Path = Path("results/track_a")
    use_subprocess: bool = True
    llm_budget_usd: float = DEFAULT_CAP_USD
    track: str = "A"
    seed: int = 20260525
    sf_to_namespace: dict[str, dict[int, str]] | None = None
    # Optional per-backend override of use_subprocess.
    # Use case: when an upstream MCP server has a broken dialect adapter
    # for one backend (e.g. duckdb-engine + SQLAlchemy 2.x incompatibility),
    # fall back to the in-process plugin for that backend only and document
    # the divergence in the phase manifest.
    backend_subprocess_overrides: dict[str, bool] | None = None
    # Skip cells whose trace.jsonl already has at least one mcp.interaction.
    # Enables resumable runs across machine restarts.
    skip_if_complete: bool = False
    # How to namespace PostgreSQL scale factors:
    #   "schema"   -> sf1/sf3/sf10 schemas in one db, set via search_path
    #                 (works for libpq-forwarding servers: mcp-alchemy, anthropic-pg)
    #   "database" -> tpch_sf1/tpch_sf3/tpch_sf10 separate databases
    #                 (for servers like DBHub whose DSN parser drops the libpq
    #                 'options' param, so search_path can't be set via DSN)
    pg_namespace_mode: str = "schema"


@dataclass(frozen=True, slots=True)
class PhaseReport:
    """Aggregate report returned by :func:`run_phase`."""

    run_id: str
    server: str
    cells: list[CellOutcome]
    started_at: str
    ended_at: str
    total_traces_collected: int
    budget_state: dict


def run_phase(config: PhaseConfig) -> PhaseReport:
    """Iterate every cell, write trace files, write phase artifacts."""
    return asyncio.run(_run_phase_async(config))


async def _run_phase_async(config: PhaseConfig) -> PhaseReport:
    root = phase_root(config.output_root, server=config.server)
    root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    budget = BudgetTracker(
        run_name=config.run_id, track=config.track,
        cap_usd=config.llm_budget_usd,
    )
    outcomes: list[CellOutcome] = []
    total_traces = 0
    for cw in config.cold_warm:
        for backend in config.backends:
            for query in config.queries:
                for sf in config.scale_factors:
                    spec = CellSpec(
                        server=config.server, backend=backend, query=query,
                        scale_factor=sf, cold_warm=cw,
                    )
                    outcome = await _run_one_cell(spec=spec, config=config)
                    outcomes.append(outcome)
                    total_traces += outcome.reps_completed
    ended_at = datetime.now(timezone.utc).isoformat()
    _write_phase_manifest(
        root=root, config=config, outcomes=outcomes,
        started_at=started_at, ended_at=ended_at,
    )
    _write_budget_json(root=root, budget=budget)
    return PhaseReport(
        run_id=config.run_id, server=config.server, cells=outcomes,
        started_at=started_at, ended_at=ended_at,
        total_traces_collected=total_traces,
        budget_state=budget.to_dict(),
    )


async def _run_one_cell(*, spec: CellSpec, config: PhaseConfig) -> CellOutcome:
    cid = cell_id(spec)
    cell_dir = trace_dir(config.output_root, spec=spec)
    cell_dir.mkdir(parents=True, exist_ok=True)
    trace_path = cell_dir / "trace.jsonl"
    dsn = _resolve_dsn(config=config, backend=spec.backend, scale_factor=spec.scale_factor)
    if config.skip_if_complete and _has_interactions(trace_path):
        existing = _count_interactions(trace_path)
        _LOG.info("cell %s: skipping (already has %d interactions)", cid, existing)
        return CellOutcome(
            cell_id=cid, success=True, trace_path=trace_path,
            reps_completed=existing, error_message=None,
        )
    try:
        reps_done = await _drive_cell(
            spec=spec, config=config, trace_path=trace_path, dsn=dsn,
        )
        _write_cell_manifest(cell_dir=cell_dir, spec=spec, reps=reps_done, dsn=dsn)
        return CellOutcome(
            cell_id=cid, success=True, trace_path=trace_path,
            reps_completed=reps_done, error_message=None,
        )
    except Exception as exc:
        _LOG.exception("cell %s failed", cid)
        _write_cell_manifest(
            cell_dir=cell_dir, spec=spec, reps=0, dsn=dsn,
            error=str(exc),
        )
        return CellOutcome(
            cell_id=cid, success=False, trace_path=trace_path,
            reps_completed=0, error_message=str(exc),
        )


def _resolve_dsn(
    *, config: PhaseConfig, backend: str, scale_factor: int,
) -> str | None:
    """Return the DSN for (backend, sf), applying optional namespace map."""
    base = config.backend_dsns.get(backend)
    if base is None:
        return None
    if config.sf_to_namespace and backend in config.sf_to_namespace:
        ns = config.sf_to_namespace[backend].get(scale_factor)
        if ns is None:
            return base
        return _apply_namespace(
            base=base, backend=backend, ns=ns,
            pg_namespace_mode=config.pg_namespace_mode,
        )
    return base


def _apply_namespace(
    *, base: str, backend: str, ns: str, pg_namespace_mode: str = "schema",
) -> str:
    """Tweak a DSN for the scale-factor namespace.

    Postgres (schema mode):   append ``?options=-csearch_path%3D<schema>``.
    Postgres (database mode):  swap the trailing /dbname to ``<ns>`` (for
                               servers whose DSN parser drops libpq options).
    MySQL:    swap database name to ``<ns>``.
    DuckDB:   replace path with ``<ns>``.
    """
    if backend == "postgres" and pg_namespace_mode == "database":
        # Swap the database name; tables live in public, no search_path needed.
        if "?" in base:
            head, query = base.split("?", 1)
            query = f"?{query}"
        else:
            head, query = base, ""
        if "/" in head.rsplit("@", 1)[-1]:
            prefix, _ = head.rsplit("/", 1)
            return f"{prefix}/{ns}{query}"
        return f"{head}/{ns}{query}"
    if backend == "postgres":
        sep = "&" if "?" in base else "?"
        # Disable parallel workers — Docker's /dev/shm is 64M, too small for
        # PG parallel-worker DSM segments. ALTER ROLE didn't stick; embed in DSN.
        opts = (
            f"-csearch_path%3D{ns}"
            "%20-cmax_parallel_workers_per_gather%3D0"
            "%20-cmax_parallel_workers%3D0"
        )
        return f"{base}{sep}options={opts}"
    if backend == "mysql":
        # Replace the trailing /dbname segment.
        if "/" in base.rsplit("@", 1)[-1]:
            head, _ = base.rsplit("/", 1)
            return f"{head}/{ns}"
        return f"{base}/{ns}"
    if backend == "duckdb":
        return ns
    return base


async def _drive_cell(
    *, spec: CellSpec, config: PhaseConfig, trace_path: Path, dsn: str | None,
) -> int:
    """Run reps for one cell.

    Cold semantics: one fresh plugin instance + tracer is created PER rep.
    Warm semantics: one plugin + tracer covers all reps in the cell.

    The tracer is shared across reps because OTel JSONL exports are append-only,
    so one trace.jsonl per cell holds N ``mcp.interaction`` roots whether the
    underlying subprocess is recycled or not.
    """
    if spec.cold_warm == "cold":
        return await _drive_cell_cold(
            spec=spec, config=config, trace_path=trace_path, dsn=dsn,
        )
    return await _drive_cell_warm(
        spec=spec, config=config, trace_path=trace_path, dsn=dsn,
    )


async def _drive_cell_warm(
    *, spec: CellSpec, config: PhaseConfig, trace_path: Path, dsn: str | None,
) -> int:
    use_subprocess = _effective_use_subprocess(config=config, backend=spec.backend)
    plugin = _load_plugin(config=config, dsn=dsn, use_subprocess=use_subprocess)
    backend_obj = load_backend(spec.backend, dsn) if dsn else load_backend(spec.backend)
    warmup_path = trace_path.parent / "trace_warmup.jsonl"
    sql = load_query(spec.query, spec.backend)
    tool_name = _DEFAULT_TOOL_NAMES.get(config.server, "query")
    reps_done = 0
    if config.warmup > 0:
        warmup_tracer = OTelTracer.with_jsonl_exporter(warmup_path)
        warmup_runner = TrackARunner(
            plugin=plugin, backend=backend_obj, tracer=warmup_tracer,
            backend_name=spec.backend,
        )
        try:
            for i in range(config.warmup):
                await warmup_runner.run_interaction(
                    query_id=spec.query, scale_factor=spec.scale_factor, sql=sql,
                    iteration=i, cold_start=False, tool_name=tool_name,
                )
        finally:
            warmup_tracer.shutdown()
    tracer = OTelTracer.with_jsonl_exporter(trace_path)
    runner = TrackARunner(
        plugin=plugin, backend=backend_obj, tracer=tracer,
        backend_name=spec.backend,
    )
    try:
        for i in range(config.reps):
            await runner.run_interaction(
                query_id=spec.query, scale_factor=spec.scale_factor, sql=sql,
                iteration=i, cold_start=False, tool_name=tool_name,
            )
            reps_done += 1
    finally:
        tracer.shutdown()
        if hasattr(backend_obj, "close"):
            backend_obj.close()
    return reps_done


async def _drive_cell_cold(
    *, spec: CellSpec, config: PhaseConfig, trace_path: Path, dsn: str | None,
) -> int:
    """Cold cell: respawn plugin + backend connection per rep.

    Warmup interactions write to ``trace_warmup.jsonl``; measured reps
    write to ``trace.jsonl``. Each rep produces one ``mcp.interaction`` root.
    """
    sql = load_query(spec.query, spec.backend)
    tool_name = _DEFAULT_TOOL_NAMES.get(config.server, "query")
    warmup_path = trace_path.parent / "trace_warmup.jsonl"
    if config.warmup > 0:
        warmup_tracer = OTelTracer.with_jsonl_exporter(warmup_path)
        try:
            for i in range(config.warmup):
                await _one_cold_rep(
                    spec=spec, config=config, dsn=dsn, tracer=warmup_tracer,
                    iteration=i, sql=sql, tool_name=tool_name,
                )
        finally:
            warmup_tracer.shutdown()
    tracer = OTelTracer.with_jsonl_exporter(trace_path)
    reps_done = 0
    try:
        for i in range(config.reps):
            await _one_cold_rep(
                spec=spec, config=config, dsn=dsn, tracer=tracer,
                iteration=i, sql=sql, tool_name=tool_name,
            )
            reps_done += 1
    finally:
        tracer.shutdown()
    return reps_done


async def _one_cold_rep(
    *, spec: CellSpec, config: PhaseConfig, dsn: str | None,
    tracer: OTelTracer, iteration: int, sql: str, tool_name: str,
) -> None:
    use_subprocess = _effective_use_subprocess(config=config, backend=spec.backend)
    plugin = _load_plugin(config=config, dsn=dsn, use_subprocess=use_subprocess)
    backend_obj = load_backend(spec.backend, dsn) if dsn else load_backend(spec.backend)
    runner = TrackARunner(
        plugin=plugin, backend=backend_obj, tracer=tracer,
        backend_name=spec.backend,
    )
    try:
        await runner.run_interaction(
            query_id=spec.query, scale_factor=spec.scale_factor, sql=sql,
            iteration=iteration, cold_start=True, tool_name=tool_name,
        )
    finally:
        if hasattr(backend_obj, "close"):
            backend_obj.close()


def _effective_use_subprocess(*, config: PhaseConfig, backend: str) -> bool:
    """Resolve per-backend subprocess setting, falling back to the global default."""
    overrides = config.backend_subprocess_overrides or {}
    return overrides.get(backend, config.use_subprocess)


def _load_plugin(*, config: PhaseConfig, dsn: str | None, use_subprocess: bool):
    """Instantiate the named server plugin.

    DSN handoff differs by upstream:
      - mcp-alchemy: ``db_url`` (SQLAlchemy URL form).
      - anthropic-pg, dbhub: ``dsn`` (native PG/MySQL DSN form).
    """
    from mcp_dbs_bench.core.registry import _resolve, _server_overrides
    cls = _resolve(
        config.server, _server_overrides, "mcp_dbs_bench.servers", "server",
    )
    if config.server == "mcp-alchemy":
        return cls(
            use_subprocess=use_subprocess,
            db_url=_to_sqlalchemy_url(dsn) if dsn else None,
        )
    if config.server in ("anthropic-pg", "dbhub"):
        return cls(use_subprocess=use_subprocess, dsn=dsn)
    try:
        return cls(use_subprocess=use_subprocess)
    except TypeError:
        return cls()


def _to_sqlalchemy_url(dsn: str) -> str:
    """Translate a harness DSN to a SQLAlchemy URL for mcp-alchemy."""
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    if dsn.startswith("mysql://"):
        return dsn.replace("mysql://", "mysql+pymysql://", 1)
    if dsn.endswith(".duckdb") or dsn.startswith("/"):
        return f"duckdb:///{dsn}"
    return dsn


def _write_cell_manifest(
    *, cell_dir: Path, spec: CellSpec, reps: int, dsn: str | None,
    error: str | None = None,
) -> None:
    manifest = {
        "cell_id": cell_id(spec),
        "server": spec.server,
        "backend": spec.backend,
        "query": spec.query,
        "scale_factor": spec.scale_factor,
        "cold_warm": spec.cold_warm,
        "reps": reps,
        "dsn": dsn,
        "error": error,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_phase_manifest(
    *, root: Path, config: PhaseConfig, outcomes: list[CellOutcome],
    started_at: str, ended_at: str,
) -> None:
    """Write the phase manifest. If one exists, merge runs and cells."""
    new_run = {
        "started_at": started_at,
        "ended_at": ended_at,
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": {
            "backends": list(config.backends),
            "queries": list(config.queries),
            "scale_factors": list(config.scale_factors),
            "reps": config.reps,
            "warmup": config.warmup,
            "cold_warm": list(config.cold_warm),
            "use_subprocess": config.use_subprocess,
            "backend_subprocess_overrides":
                config.backend_subprocess_overrides or {},
            "seed": config.seed,
            "skip_if_complete": config.skip_if_complete,
        },
        "cells": [
            {
                "cell_id": o.cell_id, "success": o.success,
                "reps_completed": o.reps_completed,
                "error_message": o.error_message,
                "trace_path": str(o.trace_path),
            }
            for o in outcomes
        ],
    }
    path = root / "manifest.json"
    existing_runs: list = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            existing_runs = list(existing.get("runs", []))
        except (json.JSONDecodeError, OSError):
            _LOG.warning("could not read existing run manifest at %s", path)
    existing_runs.append(new_run)
    manifest = {
        "run_id": config.run_id,
        "server": config.server,
        "track": config.track,
        "last_started_at": started_at,
        "last_ended_at": ended_at,
        "runs": existing_runs,
    }
    path.write_text(json.dumps(manifest, indent=2))


def _write_budget_json(*, root: Path, budget: BudgetTracker) -> None:
    (root / "budget.json").write_text(json.dumps(budget.to_dict(), indent=2))


def _has_interactions(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open() as fh:
        for line in fh:
            if '"mcp.interaction"' in line:
                return True
    return False


def _count_interactions(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open() as fh:
        for line in fh:
            if '"mcp.interaction"' in line:
                count += 1
    return count


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[3],
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"


__all__ = ["CellOutcome", "PhaseConfig", "PhaseReport", "run_phase"]
