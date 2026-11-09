"""Merge re-scored Track B attempt files into one corpus with one row per attempt.

A resumed or repeated run can leave more than one copy of an attempt, keyed by
(model, server, backend, scale factor, question, repetition). Files are read in
the order given and the last copy of each attempt wins, so list the run that
should take precedence last.

Usage:
    python scripts/track_b/build_corpus.py --output data/track_b/corpus.csv \\
        results/track_b/rescored.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

AttemptKey = tuple[str, str, str, str, str, str]


def attempt_key(row: dict[str, str]) -> AttemptKey:
    """Return the (model, server, backend, scale factor, query, repetition) of an attempt."""
    return (
        row["model"], row["server"], row["backend"],
        row["scale_factor"], row["query"], row["rep_idx"],
    )


def select_corpus(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return one copy of each attempt, keeping the last copy seen.

    Args:
        rows: Re-scored attempts in precedence order (later rows win).

    Returns:
        The de-duplicated attempts, in first-seen order.
    """
    latest: dict[AttemptKey, dict[str, str]] = {}
    for row in rows:
        latest[attempt_key(row)] = row
    return list(latest.values())


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    """Read every row of the given CSV files, in order."""
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    """Read re-scored CSVs and write the de-duplicated corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    csv.field_size_limit(10**9)
    rows = read_rows(args.inputs)
    if not rows:
        raise SystemExit(f"No attempts found in {args.inputs}; run rescore.py first")
    selected = select_corpus(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(f"Selected {len(selected)} of {len(rows)} attempts into {args.output}")


if __name__ == "__main__":
    main()
