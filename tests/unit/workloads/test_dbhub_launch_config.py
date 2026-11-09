"""Tests for the opt-in dbhub PostgreSQL schema pinning."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from track_b import _mcp_launch  # noqa: E402


@pytest.fixture
def fake_node_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mcp_launch, "_resolve_node_binary", lambda _: "/bin/dbhub")


def test_write_dbhub_postgres_config_pins_requested_schema(tmp_path: Path) -> None:
    # Act
    config_path = _mcp_launch.write_dbhub_postgres_config(
        scale_factor=3, directory=tmp_path,
    )

    # Assert
    source = tomllib.loads(config_path.read_text())["sources"][0]
    assert source["search_path"] == "sf3"


def test_build_dbhub_launch_with_flag_unset_keeps_dsn_argument(
    fake_node_binary: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.delenv("DBHUB_SEARCH_PATH_CONFIG", raising=False)

    # Act
    spec = _mcp_launch._build_dbhub_launch(backend="postgres", scale_factor=3)

    # Assert
    assert spec.extra_args[1].startswith("--dsn=")


def test_build_dbhub_launch_with_flag_set_uses_config_for_postgres(
    fake_node_binary: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("DBHUB_SEARCH_PATH_CONFIG", "1")

    # Act
    spec = _mcp_launch._build_dbhub_launch(backend="postgres", scale_factor=3)

    # Assert
    assert spec.extra_args[1].startswith("--config=")


def test_build_dbhub_launch_with_flag_set_leaves_mysql_on_dsn(
    fake_node_binary: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("DBHUB_SEARCH_PATH_CONFIG", "1")

    # Act
    spec = _mcp_launch._build_dbhub_launch(backend="mysql", scale_factor=3)

    # Assert
    assert spec.extra_args[1].startswith("--dsn=")


def test_write_dbhub_postgres_config_connects_with_scale_factor_login(tmp_path: Path) -> None:
    # Act
    config_path = _mcp_launch.write_dbhub_postgres_config(
        scale_factor=3, directory=tmp_path,
    )

    # Assert
    source = tomllib.loads(config_path.read_text())["sources"][0]
    assert source["dsn"].startswith("postgresql://bench_sf3:")
