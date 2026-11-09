"""CSV writers for Track B sweep artifacts.

Line-buffered writers so a crash mid-run leaves a partial-but-parseable
per_trace_metrics.csv. Also emits the per-cell summary at end of run.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from track_b._metrics import (
    PER_CELL_COLUMNS, PER_TRACE_COLUMNS, TrackBMetrics, summarise_cell,
)


class PerTraceWriter:
    """Line-buffered CSV writer for per_trace_metrics.csv.

    Use as a context manager to ensure the file handle is closed.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "PerTraceWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._fh, fieldnames=list(PER_TRACE_COLUMNS))
        self._writer.writeheader()
        self._fh.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None

    def write(self, row: TrackBMetrics) -> None:
        """Append one metrics row to the CSV."""
        if self._writer is None:
            raise RuntimeError("PerTraceWriter not entered")
        payload = _row_to_csv(row)
        self._writer.writerow(payload)
        if self._fh is not None:
            self._fh.flush()


def write_per_cell_summary(path: Path, rows: Iterable[TrackBMetrics]) -> None:
    """Aggregate ``rows`` by cell_id and emit per_cell_summary.csv."""
    grouped: dict[str, list[TrackBMetrics]] = {}
    for row in rows:
        grouped.setdefault(row.cell_id, []).append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PER_CELL_COLUMNS))
        writer.writeheader()
        for cell_id in sorted(grouped):
            writer.writerow(summarise_cell(grouped[cell_id]))


def _row_to_csv(row: TrackBMetrics) -> dict[str, str]:
    """Serialise a TrackBMetrics into the string-only shape csv.DictWriter wants."""
    raw = asdict(row)
    out: dict[str, str] = {}
    for col in PER_TRACE_COLUMNS:
        val = raw.get(col)
        if val is None:
            out[col] = ""
        elif isinstance(val, list):
            out[col] = ";".join(str(v) for v in val)
        elif isinstance(val, str):
            out[col] = val
        else:
            out[col] = str(val)
    return out


__all__ = ["PerTraceWriter", "write_per_cell_summary"]
