"""Checks that the shipped Track A data uses the published PTR definition."""
from __future__ import annotations

import csv
from pathlib import Path

TRACES_CSV = Path(__file__).resolve().parents[3] / "data" / "track_a" / "traces.csv"
TOLERANCE = 1e-9


def test_published_ptr_equals_total_minus_db_over_db():
    with TRACES_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mismatches = [
        row["cell_id"] for row in rows
        if abs(float(row["PTR"]) - (float(row["T_total_ms"]) - float(row["T_db_ms"])) / float(row["T_db_ms"])) > TOLERANCE
    ]
    assert mismatches == []
