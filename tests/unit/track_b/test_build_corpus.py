"""Tests for the Track B corpus merger (scripts/track_b/build_corpus.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "track_b" / "build_corpus.py"


@pytest.fixture(scope="module")
def corpus():
    spec = importlib.util.spec_from_file_location("build_corpus", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_corpus"] = module
    spec.loader.exec_module(module)
    return module


def _attempt(outcome: str, rep_idx: str = "0", query: str = "tb_c2_01") -> dict[str, str]:
    return {
        "model": "deepseek-v4-flash", "server": "dbhub", "backend": "mysql",
        "scale_factor": "1", "query": query, "rep_idx": rep_idx, "outcome": outcome,
    }


def test_select_corpus_keeps_last_copy_of_repeated_attempt(corpus):
    # Arrange
    rows = [_attempt("agent_error"), _attempt("correct")]

    # Act
    selected = corpus.select_corpus(rows)

    # Assert
    assert [row["outcome"] for row in selected] == ["correct"]


def test_select_corpus_keeps_distinct_attempts(corpus):
    # Arrange
    rows = [_attempt("correct", rep_idx="0"), _attempt("incorrect", rep_idx="1")]

    # Act
    selected = corpus.select_corpus(rows)

    # Assert
    assert len(selected) == 2


def test_select_corpus_with_no_rows_returns_empty_list(corpus):
    # Act
    selected = corpus.select_corpus([])

    # Assert
    assert selected == []
