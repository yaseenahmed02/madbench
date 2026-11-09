"""Tests for the Track B workload loader (scripts/track_b/_workload.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from track_b._workload import WorkloadError, load_workload  # noqa: E402

_QUESTIONS = _REPO_ROOT / "workloads" / "track_b" / "questions.yaml"


def _write_yaml(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


_C4_C6_YAML = """
c4:
  - id: tb_c4_01
    target_subcat: s27
    question: "q4"
    gold_sql: |
      SELECT 4
c6:
  - id: tb_c6_01
    target_subcat: s36
    question: "q6"
    gold_sql: |
      SELECT 6
"""


def test_load_workload_questions_file_yields_fifty_questions():
    # Arrange
    paths = [_QUESTIONS]

    # Act
    questions = load_workload(paths)

    # Assert
    assert len(questions) == 50


def test_load_workload_reads_c4_and_c6_categories(tmp_path):
    # Arrange
    path = _write_yaml(tmp_path, "c4_c6.yaml", _C4_C6_YAML)

    # Act
    questions = load_workload([path])

    # Assert
    assert [q["category"] for q in questions] == ["c4", "c6"]


def test_load_workload_combines_files_in_order(tmp_path):
    # Arrange
    extra = _write_yaml(tmp_path, "extra.yaml", _C4_C6_YAML.replace("tb_c", "my_c"))

    # Act
    questions = load_workload([_QUESTIONS, extra])

    # Assert
    assert [q["id"] for q in questions[-2:]] == ["my_c4_01", "my_c6_01"]


def test_load_workload_strips_gold_sql_whitespace(tmp_path):
    # Arrange
    path = _write_yaml(tmp_path, "c4_c6.yaml", _C4_C6_YAML)

    # Act
    questions = load_workload([path])

    # Assert
    assert questions[0]["gold_sql"] == "SELECT 4"


def test_load_workload_with_duplicate_ids_raises_workload_error(tmp_path):
    # Arrange
    first = _write_yaml(tmp_path, "a.yaml", _C4_C6_YAML)
    second = _write_yaml(tmp_path, "b.yaml", _C4_C6_YAML)

    # Act / Assert
    with pytest.raises(WorkloadError, match="duplicate question ids"):
        load_workload([first, second])


def test_load_workload_with_no_category_keys_raises_workload_error(tmp_path):
    # Arrange
    path = _write_yaml(tmp_path, "empty.yaml", "schema: tpch\n")

    # Act / Assert
    with pytest.raises(WorkloadError, match="no c1-c6 questions"):
        load_workload([path])


def test_load_workload_ignores_non_category_keys(tmp_path):
    # Arrange
    body = "schema: tpch\nscale_factor_default: 1\n" + _C4_C6_YAML
    path = _write_yaml(tmp_path, "mixed.yaml", body)

    # Act
    questions = load_workload([path])

    # Assert
    assert len(questions) == 2
