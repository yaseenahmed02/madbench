"""TPC-H query index: 5 queries x 3 dialects = 15 SQL files.

Each query has one SQL file per dialect and an expected result cardinality
per scale factor.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

QUERY_IDS: tuple[str, ...] = ("Q1", "Q3", "Q6", "Q9", "Q21")
DIALECTS: tuple[str, ...] = ("postgres", "mysql", "duckdb")

# Expected result cardinalities per (query, scale_factor).
_EXPECTED_CARDINALITY: dict[tuple[str, int], int] = {
    ("Q1", 1): 4, ("Q1", 3): 4, ("Q1", 10): 4,
    ("Q3", 1): 10, ("Q3", 3): 10, ("Q3", 10): 10,
    ("Q6", 1): 1, ("Q6", 3): 1, ("Q6", 10): 1,
    ("Q9", 1): 175, ("Q9", 3): 175, ("Q9", 10): 175,
    ("Q21", 1): 100, ("Q21", 3): 100, ("Q21", 10): 411,
}

_QUERIES_DIR = Path(__file__).resolve().parent / "queries"


@lru_cache(maxsize=64)
def load_query(query_id: str, dialect: str) -> str:
    """Return the SQL text for a (query_id, dialect) pair.

    Raises:
        ValueError: If ``query_id`` or ``dialect`` is unknown.
        FileNotFoundError: If the SQL file is missing on disk.
    """
    if query_id not in QUERY_IDS:
        raise ValueError(f"Unknown query_id {query_id!r}; valid: {QUERY_IDS}")
    if dialect not in DIALECTS:
        raise ValueError(f"Unknown dialect {dialect!r}; valid: {DIALECTS}")
    path = _QUERIES_DIR / dialect / f"{query_id}.sql"
    if not path.is_file():
        raise FileNotFoundError(f"TPC-H query file not found: {path}")
    return path.read_text()


def expected_cardinality(query_id: str, scale_factor: int) -> int:
    """Return the expected row count for (query, SF).

    Raises:
        KeyError: If the (query, SF) combination is not in the table.
    """
    key = (query_id, scale_factor)
    if key not in _EXPECTED_CARDINALITY:
        raise KeyError(
            f"No published cardinality for query {query_id} at SF{scale_factor}. "
            f"Valid keys: {sorted(_EXPECTED_CARDINALITY)}"
        )
    return _EXPECTED_CARDINALITY[key]


QUERIES: dict[str, dict[str, str]] = {
    query_id: {dialect: load_query(query_id, dialect) for dialect in DIALECTS}
    for query_id in QUERY_IDS
}
