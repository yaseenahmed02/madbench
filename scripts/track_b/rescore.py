"""Re-score Track B attempts from their raw traces with the verifier.

Applies one verifier to every attempt under the given result roots, so all
reported outcomes come from a single scorer. Only logged ``incorrect`` attempts
are re-verified: a logged ``correct`` may rest on the harness executing a
submitted SQL string, which is not repeatable offline. Results are read and
never modified.

Reads ``<root>/<model>/<server>-<backend>/sf<N>/<server>/per_trace_metrics.csv``
(the layout ``run_track_b.sh`` writes) and the gold results captured by
``capture_gold_results.py``.

Usage:
    python scripts/track_b/rescore.py --output results/track_b/rescored.csv results/track_b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcp_dbs_bench.agent.correctness import GoldResult, verify  # noqa: E402

GOLD_ROOT = REPO_ROOT / "workloads" / "track_b"
METRICS_GLOB = "*/*/sf*/*/per_trace_metrics.csv"
RESCORED_OUTCOME = "incorrect"
MODEL_PATH_INDEX = -5


@dataclass(frozen=True)
class ScoredAttempt:
    """One attempt with its logged and final outcome."""

    model: str
    server: str
    backend: str
    scale_factor: int
    query: str
    category: str
    rep_idx: int
    logged_outcome: str
    outcome: str
    n_tool_calls: int
    total_input_tokens: int
    total_output_tokens: int
    cost_usd: float
    wall_clock_ms: float
    diagnostic: str


def read_final_answer(trace_path: Path) -> Any:
    """Return the final answer from an attempt's JSONL trace, or None if absent."""
    final_answer = None
    with trace_path.open(encoding="utf-8") as trace:
        for line in trace:
            record = json.loads(line)
            if record.get("type") == "outcome":
                final_answer = record.get("final_answer")
    return final_answer


def load_gold(query: str, backend: str, scale_factor: int) -> GoldResult:
    """Load the captured gold result for one question on one backend."""
    gold_path = GOLD_ROOT / f"gold_results_sf{scale_factor}" / f"{query}__{backend}.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    return GoldResult(
        query_id=query,
        backend=backend,
        gold_sql=gold["gold_sql"],
        columns=gold["columns"],
        rows=gold["rows"],
    )


def rescore_outcome(row: dict[str, str], metrics_dir: Path) -> tuple[str, str]:
    """Return the final outcome and diagnostic for one logged attempt."""
    if row["outcome"] != RESCORED_OUTCOME:
        return row["outcome"], row["verifier_diagnostic"]
    relative_trace = row["trace_path"].split("/raw_traces/")[-1]
    final_answer = read_final_answer(metrics_dir / "raw_traces" / relative_trace)
    gold = load_gold(row["query"], row["backend"], int(row["scale_factor"]))
    verdict, diagnostic = verify(final_answer, gold)
    return verdict, diagnostic


def score_attempt(row: dict[str, str], metrics_path: Path) -> ScoredAttempt:
    """Build the scored record for one row of a per-trace metrics file."""
    outcome, diagnostic = rescore_outcome(row, metrics_path.parent)
    cell_parts = metrics_path.parts
    return ScoredAttempt(
        model=cell_parts[MODEL_PATH_INDEX],
        server=row["server"],
        backend=row["backend"],
        scale_factor=int(row["scale_factor"]),
        query=row["query"],
        category=row["category"],
        rep_idx=int(row["rep_idx"]),
        logged_outcome=row["outcome"],
        outcome=outcome,
        n_tool_calls=int(row["n_tool_calls"] or 0),
        total_input_tokens=int(row["total_input_tokens"] or 0),
        total_output_tokens=int(row["total_output_tokens"] or 0),
        cost_usd=float(row["cost_usd"] or 0),
        wall_clock_ms=float(row["wall_clock_ms"] or 0),
        diagnostic=diagnostic,
    )


def iter_scored_attempts(result_roots: list[Path]) -> Iterator[ScoredAttempt]:
    """Yield every attempt under the given result roots, re-scored."""
    for root in result_roots:
        for metrics_path in sorted(root.glob(METRICS_GLOB)):
            with metrics_path.open(newline="", encoding="utf-8") as metrics:
                for row in csv.DictReader(metrics):
                    yield score_attempt(row, metrics_path)


def write_scored_attempts(attempts: list[ScoredAttempt], output_path: Path) -> None:
    """Write scored attempts as CSV, creating the parent directory if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(attempts[0])))
        writer.writeheader()
        writer.writerows(asdict(attempt) for attempt in attempts)


def main() -> None:
    """Re-score the given result roots and print the outcome changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attempts = list(iter_scored_attempts(args.roots))
    if not attempts:
        raise SystemExit(f"No attempts found under {args.roots}; check the result roots")
    write_scored_attempts(attempts, args.output)
    changes = Counter(
        (attempt.logged_outcome, attempt.outcome)
        for attempt in attempts
        if attempt.logged_outcome != attempt.outcome
    )
    print(f"Scored {len(attempts)} attempts into {args.output}; changes: {dict(changes)}")


if __name__ == "__main__":
    main()
