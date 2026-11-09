"""Tests for canonical paths and run-id formats."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_dbs_bench.track_a.organize import (
    CellSpec,
    cell_id,
    phase_root,
    trace_dir,
)


def test_cell_id_canonical_format() -> None:
    spec = CellSpec(
        server="mcp-alchemy", backend="postgres", query="Q1",
        scale_factor=1, cold_warm="cold",
    )

    assert cell_id(spec) == "mcp-alchemy_postgres_Q1_sf1_cold"


def test_cell_id_warm_variant() -> None:
    spec = CellSpec(
        server="mcp-alchemy", backend="duckdb", query="Q21",
        scale_factor=10, cold_warm="warm",
    )

    assert cell_id(spec) == "mcp-alchemy_duckdb_Q21_sf10_warm"


def test_phase_root_canonical_structure(tmp_path: Path) -> None:
    root = phase_root(tmp_path, server="mcp-alchemy")

    assert root == tmp_path / "mcp-alchemy"


def test_trace_dir_under_raw_traces(tmp_path: Path) -> None:
    spec = CellSpec(
        server="mcp-alchemy", backend="mysql", query="Q3",
        scale_factor=3, cold_warm="cold",
    )

    path = trace_dir(tmp_path, spec=spec)

    expected = (
        tmp_path / "mcp-alchemy" / "raw_traces"
        / "mcp-alchemy_mysql_Q3_sf3_cold"
    )
    assert path == expected


def test_cell_spec_rejects_invalid_cold_warm() -> None:
    with pytest.raises(ValueError):
        CellSpec(
            server="mcp-alchemy", backend="postgres", query="Q1",
            scale_factor=1, cold_warm="hot",
        )


def test_cell_spec_rejects_non_positive_sf() -> None:
    with pytest.raises(ValueError):
        CellSpec(
            server="mcp-alchemy", backend="postgres", query="Q1",
            scale_factor=0, cold_warm="cold",
        )
