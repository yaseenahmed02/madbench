"""Smoke tests for the figure renderer."""

from __future__ import annotations

from pathlib import Path

from mcp_dbs_bench.track_a.csv_writer import (
    write_per_cell_summary, write_phase_breakdown, write_ptr_by_cell,
)
from mcp_dbs_bench.track_a.figures import render_all
from mcp_dbs_bench.track_a.trace_loader import TraceMetrics


def _make_metric(
    backend: str, query: str, sf: int, cw: str, rep: int,
) -> TraceMetrics:
    return TraceMetrics(
        cell_id=f"s_{backend}_{query}_sf{sf}_{cw}",
        server="s", backend=backend, query=query,
        scale_factor=sf, rep_idx=rep, cold_warm=cw,
        t_total_ms=1.0 + 0.01 * rep, t_db_ms=0.5,
        t_disc_ms=0.05, t_life_ms=0.05, t_hand_ms=0.1,
        t_ser_ms=0.05, t_parse_ms=0.05, t_xport_excl_ms=0.05,
        t_teardown_ms=0.0,
        epsilon_ms=0.15, ptax_ms=0.35, ptr=0.7,
        b_mcp=1024, tau_disc=50, phi=2.0, gamma=0.001,
        s_ser_rows_per_ms=100.0, rho=1.5, n_life=1,
        r_retry_ms=0.0, t_pool_ms=None, m_peak_mb=None,
        cpu_percent_samples=(),
        trace_path="t.jsonl",
    )


def test_render_all_writes_eight_files(tmp_path: Path) -> None:
    cells = []
    for backend in ("postgres", "mysql", "duckdb"):
        for query in ("Q1", "Q3", "Q6", "Q9", "Q21"):
            for sf in (1, 3, 10):
                for cw in ("cold", "warm"):
                    cells.append([
                        _make_metric(backend, query, sf, cw, i)
                        for i in range(5)
                    ])
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    write_per_cell_summary(csv_dir / "per_cell_summary.csv", cells, n_bootstrap=50)
    write_phase_breakdown(csv_dir / "phase_breakdown.csv", cells)
    write_ptr_by_cell(csv_dir / "ptr_by_cell.csv", cells)

    out = render_all(
        run_id="mcp-alchemy", csv_dir=csv_dir,
        png_dir=tmp_path / "png", pdf_dir=tmp_path / "pdf",
    )

    assert len(out) == 4
    for png, pdf in out.values():
        assert png.is_file()
        assert pdf.is_file()
        assert png.stat().st_size > 0
        assert pdf.stat().st_size > 0
