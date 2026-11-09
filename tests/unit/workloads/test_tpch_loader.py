"""Smoke tests for the TPC-H bulk loader.

The tests exercise the non-network code paths with synthetic .tbl data against an in-memory
DuckDB instance plus mocks for the directory-discovery helpers — enough
to catch silent type coercion or schema-shape errors on each row.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mcp_dbs_bench.backends.duckdb import DuckDBBackend
from mcp_dbs_bench.core.exceptions import ConfigurationError
from mcp_dbs_bench.workloads.tpch.loader import (
    assert_tpch_data_present,
    load_table_duckdb,
    reset_schema,
    tpch_data_dir,
)


def test_tpch_data_dir_resolves_under_repo_root() -> None:
    """Default repo-root resolution places SF data under ``data/tpch/sfN``."""
    root = Path("/tmp/repo")
    path = tpch_data_dir(1, repo_root=root)
    assert path == root / "data" / "tpch" / "sf1"


def test_tpch_data_dir_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``$TPCH_DATA_DIR`` env var must override the repo-root default so
    HPC/CI setups can keep raw .tbl files outside the repo (e.g., on /tmp
    when the home dir lives on a quota-limited NFS mount)."""
    monkeypatch.setenv("TPCH_DATA_DIR", "/scratch/tpch")
    path = tpch_data_dir(3, repo_root=Path("/should/be/ignored"))
    assert path == Path("/scratch/tpch") / "sf3"


def test_tpch_data_dir_falls_back_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``$TPCH_DATA_DIR`` is unset the repo-root default applies."""
    monkeypatch.delenv("TPCH_DATA_DIR", raising=False)
    root = Path("/tmp/repo")
    path = tpch_data_dir(10, repo_root=root)
    assert path == root / "data" / "tpch" / "sf10"


def test_assert_tpch_data_present_raises_when_directory_missing(
    tmp_path: Path,
) -> None:
    """A missing SF directory must raise ``ConfigurationError`` with a
    repro hint to ``scripts/build_tpch.sh``."""
    with pytest.raises(ConfigurationError, match="not found"):
        assert_tpch_data_present(1, repo_root=tmp_path)


def test_assert_tpch_data_present_raises_when_table_file_missing(
    tmp_path: Path,
) -> None:
    """Even with the SF dir present, missing .tbl files must raise."""
    data_dir = tmp_path / "data" / "tpch" / "sf1"
    data_dir.mkdir(parents=True)
    with pytest.raises(ConfigurationError, match="Missing"):
        assert_tpch_data_present(1, repo_root=tmp_path)


def test_reset_schema_rejects_unknown_backend() -> None:
    """The schema map is keyed by the three supported backends."""
    backend = DuckDBBackend(":memory:")
    try:
        with pytest.raises(ConfigurationError, match="Unknown backend"):
            reset_schema(backend, "sqlserver")
    finally:
        backend.close()


def test_reset_schema_creates_duckdb_tables_and_idempotent() -> None:
    """``reset_schema`` must produce the 8 TPC-H tables on DuckDB and be
    safely re-runnable (idempotent drop-then-create)."""
    backend = DuckDBBackend(":memory:")
    try:
        reset_schema(backend, "duckdb")
        rows, _, _ = backend.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        )
        table_names = {r[0] for r in rows}
        assert {"region", "nation", "part", "supplier",
                "partsupp", "customer", "orders", "lineitem"} <= table_names

        # Second call must succeed (drops then recreates).
        reset_schema(backend, "duckdb")
        rows2, _, _ = backend.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        )
        assert {r[0] for r in rows2} == table_names
    finally:
        backend.close()


def test_load_table_duckdb_returns_row_count(tmp_path: Path) -> None:
    """``load_table_duckdb`` must COPY ``.tbl`` rows and return the count."""
    backend = DuckDBBackend(":memory:")
    try:
        reset_schema(backend, "duckdb")
        region_tbl = tmp_path / "region.tbl"
        region_tbl.write_text(
            textwrap.dedent(
                """\
                0|AFRICA|africa|
                1|AMERICA|america|
                2|ASIA|asia|
                3|EUROPE|europe|
                4|MIDDLE EAST|middle east|
                """
            )
        )
        n_rows = load_table_duckdb(backend, "region", region_tbl)
        assert n_rows == 5
        verify_rows, _, _ = backend.execute("SELECT COUNT(*) FROM region")
        assert verify_rows[0][0] == 5
    finally:
        backend.close()
