"""Smoke test for the phase runner using in-process DuckDB.

The phase runner iterates cells, runs N reps per cell, writes
``trace.jsonl`` + ``manifest.json`` per cell, and aggregates all
artifacts under the canonical phase root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_dbs_bench.track_a.phase_runner import (
    CellOutcome,
    PhaseConfig,
    run_phase,
)


@pytest.fixture
def duckdb_with_tpch_q6(tmp_path: Path) -> str:
    """Create an in-memory DuckDB file with a minimal lineitem fragment for Q6.

    Q6 needs columns l_extendedprice, l_discount, l_quantity, l_shipdate
    on a single ``lineitem`` table. We seed 100 rows so the test is fast.
    """
    import duckdb
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE lineitem (
            l_orderkey INTEGER, l_partkey INTEGER, l_suppkey INTEGER,
            l_linenumber INTEGER,
            l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2),
            l_discount DECIMAL(15,2), l_tax DECIMAL(15,2),
            l_returnflag CHAR(1), l_linestatus CHAR(1),
            l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE,
            l_shipinstruct CHAR(25), l_shipmode CHAR(10), l_comment VARCHAR(44)
        )
    """)
    rows = [
        (i, i, i, 1, 10.0, 1000.0, 0.05, 0.08, "N", "O",
         "1994-06-15", "1994-07-01", "1994-07-15", "DELIVER", "AIR", "comment")
        for i in range(1, 101)
    ]
    conn.executemany(
        "INSERT INTO lineitem VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.close()
    return str(db_path)


def test_run_phase_executes_one_rep_per_cold_cell(
    tmp_path: Path, duckdb_with_tpch_q6: str,
) -> None:
    config = PhaseConfig(
        run_id="test", server="mcp-alchemy",
        backend_dsns={"duckdb": duckdb_with_tpch_q6},
        backends=["duckdb"], queries=["Q6"], scale_factors=[1],
        reps=1, warmup=0, cold_warm=("cold",),
        output_root=tmp_path, use_subprocess=False,
    )

    report = run_phase(config)

    assert report.total_traces_collected == 1
    assert len([c for c in report.cells if c.success]) == 1


def test_run_phase_writes_trace_jsonl_per_cell(
    tmp_path: Path, duckdb_with_tpch_q6: str,
) -> None:
    config = PhaseConfig(
        run_id="test", server="mcp-alchemy",
        backend_dsns={"duckdb": duckdb_with_tpch_q6},
        backends=["duckdb"], queries=["Q6"], scale_factors=[1],
        reps=2, warmup=0, cold_warm=("cold",),
        output_root=tmp_path, use_subprocess=False,
    )

    run_phase(config)

    expected_trace = (
        tmp_path / "mcp-alchemy" / "raw_traces"
        / "mcp-alchemy_duckdb_Q6_sf1_cold" / "trace.jsonl"
    )
    assert expected_trace.is_file()
    assert expected_trace.stat().st_size > 0


def test_run_phase_writes_cell_manifest(
    tmp_path: Path, duckdb_with_tpch_q6: str,
) -> None:
    config = PhaseConfig(
        run_id="test", server="mcp-alchemy",
        backend_dsns={"duckdb": duckdb_with_tpch_q6},
        backends=["duckdb"], queries=["Q6"], scale_factors=[1],
        reps=1, warmup=0, cold_warm=("cold",),
        output_root=tmp_path, use_subprocess=False,
    )

    run_phase(config)

    manifest_path = (
        tmp_path / "mcp-alchemy" / "raw_traces"
        / "mcp-alchemy_duckdb_Q6_sf1_cold" / "manifest.json"
    )
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["server"] == "mcp-alchemy"
    assert manifest["backend"] == "duckdb"
    assert manifest["query"] == "Q6"
    assert manifest["scale_factor"] == 1
    assert manifest["cold_warm"] == "cold"
    assert manifest["reps"] == 1


def test_run_phase_writes_phase_manifest_with_git_sha(
    tmp_path: Path, duckdb_with_tpch_q6: str,
) -> None:
    config = PhaseConfig(
        run_id="test", server="mcp-alchemy",
        backend_dsns={"duckdb": duckdb_with_tpch_q6},
        backends=["duckdb"], queries=["Q6"], scale_factors=[1],
        reps=1, warmup=0, cold_warm=("cold",),
        output_root=tmp_path, use_subprocess=False,
    )

    run_phase(config)

    phase_manifest = tmp_path / "mcp-alchemy" / "manifest.json"
    assert phase_manifest.is_file()
    data = json.loads(phase_manifest.read_text())
    assert "runs" in data
    assert len(data["runs"]) == 1
    run = data["runs"][0]
    assert "git_sha" in run
    assert "started_at" in run
    assert "ended_at" in run
    assert "python_version" in run
    assert "cells" in run
    assert len(run["cells"]) == 1


def test_run_phase_writes_budget_json(
    tmp_path: Path, duckdb_with_tpch_q6: str,
) -> None:
    config = PhaseConfig(
        run_id="test", server="mcp-alchemy",
        backend_dsns={"duckdb": duckdb_with_tpch_q6},
        backends=["duckdb"], queries=["Q6"], scale_factors=[1],
        reps=1, warmup=0, cold_warm=("cold",),
        output_root=tmp_path, use_subprocess=False,
    )

    run_phase(config)

    budget_path = tmp_path / "mcp-alchemy" / "budget.json"
    assert budget_path.is_file()
    budget = json.loads(budget_path.read_text())
    assert budget["track"] == "A"
    assert budget["llm_cost_usd"] == 0.0


def test_run_phase_cell_failure_is_recorded_and_does_not_abort(
    tmp_path: Path,
) -> None:
    """Bad DSN must surface as a CellOutcome failure, not raise."""
    config = PhaseConfig(
        run_id="test", server="mcp-alchemy",
        backend_dsns={"duckdb": "/nonexistent/path/that/cannot/open"},
        backends=["duckdb"], queries=["Q6"], scale_factors=[1],
        reps=1, warmup=0, cold_warm=("cold",),
        output_root=tmp_path, use_subprocess=False,
    )

    report = run_phase(config)

    failed = [c for c in report.cells if not c.success]
    assert len(failed) == 1
    assert failed[0].error_message is not None


def test_cell_outcome_carries_paths_and_status() -> None:
    outcome = CellOutcome(
        cell_id="x", success=True, trace_path=Path("trace.jsonl"),
        reps_completed=30, error_message=None,
    )
    assert outcome.success is True
    assert outcome.reps_completed == 30
