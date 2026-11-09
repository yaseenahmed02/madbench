"""Canonical result-set normalisation for Track B.

Gold results are captured once per (query, backend, SF) and stored as JSON.
Agent answers arrive at verification time in different shapes but must
compare equal to the gold set. Both sides go through the same normaliser:

- Column names lowercased and stripped
- Floats rounded to ``FLOAT_DECIMALS`` places
- Decimals converted to floats
- Rows sorted deterministically when the gold SQL does not enforce order

This keeps the gold JSON stable across DB drivers (psycopg returns
``Decimal``, DuckDB returns ``float``, pymysql returns strings for some
types) and keeps the verifier deterministic across replays.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

FLOAT_DECIMALS: int = 4
_ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def normalise_columns(columns: list[str]) -> list[str]:
    """Lowercase and strip column names for portable comparison."""
    return [str(c).strip().lower() for c in columns]


def normalise_row(row: list[Any]) -> list[Any]:
    """Convert Decimals to floats, round floats, leave other types unchanged."""
    return [_normalise_cell(v) for v in row]


def normalise_rows(
    rows: list[list[Any]], *, gold_sql: str,
) -> list[list[Any]]:
    """Normalise every cell; sort rows deterministically for unordered SQL.

    Args:
        rows: List of rows returned by the database driver.
        gold_sql: The pre-registered gold SQL. If it contains ``ORDER BY``
            the row order is preserved; otherwise rows are sorted by their
            stringified cell values so gold files are diff-stable.
    """
    normalised = [normalise_row(r) for r in rows]
    if _ORDER_BY_PATTERN.search(gold_sql):
        return normalised
    return sorted(normalised, key=_row_sort_key)


def _normalise_cell(value: Any) -> Any:
    """Coerce one cell into a JSON-friendly, comparison-stable form."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        return round(float(value), FLOAT_DECIMALS)
    if isinstance(value, float):
        return round(value, FLOAT_DECIMALS)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _row_sort_key(row: list[Any]) -> tuple[str, ...]:
    """Stable sort key that avoids TypeError on mixed types (None vs str)."""
    return tuple("" if v is None else repr(v) for v in row)


__all__ = [
    "FLOAT_DECIMALS", "normalise_columns", "normalise_row", "normalise_rows",
]
