"""Tests for the offline Track B re-scoring script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "track_b" / "rescore.py"


@pytest.fixture(scope="module")
def rescore():
    spec = importlib.util.spec_from_file_location("rescore", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["rescore"] = module
    spec.loader.exec_module(module)
    return module


def _write_trace(metrics_dir: Path, final_answer: object) -> None:
    trace_dir = metrics_dir / "raw_traces" / "cell"
    trace_dir.mkdir(parents=True)
    records = [{"type": "attempt_meta"}, {"type": "outcome", "final_answer": final_answer}]
    (trace_dir / "rep_000.jsonl").write_text("\n".join(json.dumps(r) for r in records))


def _row(outcome: str) -> dict[str, str]:
    return {
        "outcome": outcome,
        "verifier_diagnostic": "logged diagnostic",
        "trace_path": "results/x/raw_traces/cell/rep_000.jsonl",
        "query": "tb_c2_01",
        "backend": "mysql",
        "scale_factor": "1",
    }


def _use_gold(monkeypatch, rescore, rows: list[list[object]]) -> None:
    gold = rescore.GoldResult(
        query_id="tb_c2_01", backend="mysql", gold_sql="SELECT 1", columns=["a"], rows=rows
    )
    monkeypatch.setattr(rescore, "load_gold", lambda *_: gold)


def test_rescore_outcome_flips_json_text_answer_holding_gold_rows(rescore, tmp_path, monkeypatch):
    _write_trace(tmp_path, "[[1499.49]]")
    _use_gold(monkeypatch, rescore, [[1499.49]])

    outcome, _ = rescore.rescore_outcome(_row("incorrect"), tmp_path)

    assert outcome == "correct"


def test_rescore_outcome_keeps_wrong_answer_incorrect(rescore, tmp_path, monkeypatch):
    _write_trace(tmp_path, "[[7.0]]")
    _use_gold(monkeypatch, rescore, [[1499.49]])

    outcome, _ = rescore.rescore_outcome(_row("incorrect"), tmp_path)

    assert outcome == "incorrect"


@pytest.mark.parametrize("logged", ["correct", "agent_error", "unparseable"])
def test_rescore_outcome_leaves_other_outcomes_unchanged(rescore, tmp_path, logged):
    outcome, diagnostic = rescore.rescore_outcome(_row(logged), tmp_path)

    assert (outcome, diagnostic) == (logged, "logged diagnostic")


def test_read_final_answer_returns_none_without_outcome_record(rescore, tmp_path):
    trace = tmp_path / "rep.jsonl"
    trace.write_text(json.dumps({"type": "attempt_meta"}))

    assert rescore.read_final_answer(trace) is None
