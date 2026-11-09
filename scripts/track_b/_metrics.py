"""Per-trace metrics for Track B.

The sweep runner computes these once per (query, backend, rep) attempt and
writes one row to ``per_trace_metrics.csv``. Cell-level summaries are
aggregations over the reps.

Metric definitions:
  outcome          — 'correct' | 'incorrect' | 'unparseable' | 'timeout' |
                     'call_limit_exceeded' | 'agent_error'
  attempts         — total LLM->tool->result cycles
  R                — retry count (same tool re-issued after a failed call)
  gamma            — context decay: cumulative input tokens (proxy for
                     context window growth across turns)
  phi              — token inflation: total_input_tokens / total_output_tokens
                     (guards against div-by-zero -> reported as None)
  n_turns          — number of LLM turns (== attempts here; kept as an
                     explicit column for readability)
  n_tool_calls     — non-submit tool calls
  total_input_tokens, total_output_tokens, cost_usd, wall_clock_ms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrackBMetrics:
    """One row of Track B per-trace metrics."""

    cell_id: str
    server: str
    backend: str
    query: str
    category: str
    target_subcat: str
    scale_factor: int
    rep_idx: int
    outcome: str
    verifier_diagnostic: str
    attempts: int = 0
    retry_count: int = 0
    gamma_cumulative_input_tokens: int = 0
    phi_token_inflation: float | None = None
    n_turns: int = 0
    n_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cost_usd: float = 0.0
    wall_clock_ms: float = 0.0
    trace_path: str = ""
    final_answer_kind: str = ""
    tool_names: list[str] = field(default_factory=list)


PER_TRACE_COLUMNS: tuple[str, ...] = (
    "cell_id", "server", "backend", "query", "category", "target_subcat",
    "scale_factor", "rep_idx",
    "outcome", "verifier_diagnostic",
    "attempts", "retry_count",
    "gamma_cumulative_input_tokens", "phi_token_inflation",
    "n_turns", "n_tool_calls",
    "total_input_tokens", "total_output_tokens",
    "cost_usd", "wall_clock_ms",
    "final_answer_kind", "trace_path",
)


PER_CELL_COLUMNS: tuple[str, ...] = (
    "cell_id", "server", "backend", "query", "category", "target_subcat",
    "scale_factor", "n_reps",
    "n_correct", "n_incorrect", "n_unparseable", "n_error",
    "completion_rate",
    "mean_attempts", "mean_retry_count",
    "mean_input_tokens", "mean_output_tokens",
    "mean_cost_usd", "mean_wall_clock_ms",
)


def summarise_cell(rows: list[TrackBMetrics]) -> dict[str, Any]:
    """Aggregate reps of one cell into a single summary row."""
    if not rows:
        raise ValueError("summarise_cell requires at least one row")
    head = rows[0]
    n = len(rows)
    n_correct = sum(1 for r in rows if r.outcome == "correct")
    n_incorrect = sum(1 for r in rows if r.outcome == "incorrect")
    n_unparseable = sum(1 for r in rows if r.outcome == "unparseable")
    error_states = {"timeout", "call_limit_exceeded", "agent_error"}
    n_error = sum(1 for r in rows if r.outcome in error_states)
    return {
        "cell_id": head.cell_id, "server": head.server, "backend": head.backend,
        "query": head.query, "category": head.category,
        "target_subcat": head.target_subcat,
        "scale_factor": head.scale_factor, "n_reps": n,
        "n_correct": n_correct, "n_incorrect": n_incorrect,
        "n_unparseable": n_unparseable, "n_error": n_error,
        "completion_rate": n_correct / n if n else 0.0,
        "mean_attempts": _mean(r.attempts for r in rows),
        "mean_retry_count": _mean(r.retry_count for r in rows),
        "mean_input_tokens": _mean(r.total_input_tokens for r in rows),
        "mean_output_tokens": _mean(r.total_output_tokens for r in rows),
        "mean_cost_usd": _mean(r.cost_usd for r in rows),
        "mean_wall_clock_ms": _mean(r.wall_clock_ms for r in rows),
    }


def _mean(values) -> float:
    """Arithmetic mean; empty iterable -> 0.0."""
    xs = list(values)
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


__all__ = [
    "PER_CELL_COLUMNS", "PER_TRACE_COLUMNS", "TrackBMetrics", "summarise_cell",
]
