"""Correctness verification for Track B agent answers.

The Track B agent loop terminates when the LLM calls ``submit_answer``.
That call carries a payload of unknown shape: the agent may submit a SQL
string (expected to be executed), a scalar value (single-cell result), or
a list of rows. This module normalises all three shapes and compares them
against a captured gold result set.

Design goals
------------
- Pure, testable comparison logic. No network, no filesystem side-effects.
- SQL-shape answers are executed only when an ``sql_executor`` callable is
  provided; otherwise they are marked ``unparseable`` with a clear message.
- Row ordering is respected only when the gold SQL contains ``ORDER BY``.
  Otherwise the two result sets are compared as multisets.
- Column ordering is ignored: canonicalise the gold schema by lowercased
  column names and reorder the agent rows if needed.
- Numeric tolerance applies to any pair of values that both parse as
  numbers; strings compare case-insensitively after trim.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Verdict = Literal["correct", "incorrect", "unparseable"]

# Matches ``ORDER BY`` outside quoted strings — sufficient for TPC-H gold SQL,
# which does not embed the phrase inside a literal.
_ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
# Node's PostgreSQL driver turns a DATE into a JS Date, so servers built on it
# return 1995-01-01 as 1995-01-01T00:00:00.000Z. Both spell the same value.
_MIDNIGHT_TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})t00:00:00(?:\.0+)?z?$")


@dataclass(frozen=True, slots=True)
class GoldResult:
    """Captured result set for one (query, backend) pair.

    Attributes:
        query_id: Track B query identifier, e.g. ``tb_c2_01``.
        backend: One of ``postgres`` / ``mysql`` / ``duckdb``.
        gold_sql: The pre-registered gold SQL that produced these rows.
        columns: Column names in the order returned by the database.
        rows: Row payload; each row is a list of scalars (JSON-normalised).
    """

    query_id: str
    backend: str
    gold_sql: str
    columns: list[str]
    rows: list[list[Any]]


def verify(
    agent_final_answer: Any,
    gold: GoldResult,
    *,
    tolerance: float = 1e-4,
    sql_executor: Callable[[str], tuple[list[str], list[list[Any]]]] | None = None,
) -> tuple[Verdict, str]:
    """Compare an agent's submitted answer against the captured gold result.

    Args:
        agent_final_answer: Whatever the agent passed to ``submit_answer``.
            Accepted shapes: SQL string, scalar, list of rows, dict with
            ``columns``/``rows`` or ``result``/``answer`` keys.
        gold: The captured gold result set for this (query, backend).
        tolerance: Numeric tolerance for floating-point comparisons.
        sql_executor: Optional callable that runs a SQL string against the
            same backend and returns ``(columns, rows)``. Required when the
            agent submits a SQL string; otherwise SQL-shape answers cannot
            be verified.

    Returns:
        ``(verdict, diagnostic)`` where verdict is one of
        ``"correct" | "incorrect" | "unparseable"``.
    """
    if agent_final_answer is None:
        return "unparseable", "agent submitted None"
    normalised, shape_msg = _normalise_agent_answer(agent_final_answer, sql_executor)
    if normalised is None:
        return "unparseable", shape_msg
    agent_columns, agent_rows = normalised
    return _compare_result_sets(
        agent_columns=agent_columns,
        agent_rows=agent_rows,
        gold=gold,
        tolerance=tolerance,
    )


def _normalise_agent_answer(
    answer: Any,
    sql_executor: Callable[[str], tuple[list[str], list[list[Any]]]] | None,
) -> tuple[tuple[list[str], list[list[Any]]] | None, str]:
    """Convert an agent answer into ``(columns, rows)`` or explain why not.

    Recognised shapes:
      * SQL string  -> execute via ``sql_executor`` (or unparseable if absent)
      * JSON string -> parsed, then treated as the list or dict it holds
      * scalar      -> single-cell row list
      * list        -> row list (possibly with column names in first row)
      * dict        -> ``columns`` + ``rows`` / ``result`` / ``answer``
    """
    if isinstance(answer, str):
        return _normalise_string_answer(answer, sql_executor)
    if isinstance(answer, (int, float, bool)):
        return ([""], [[answer]]), "scalar answer"
    if isinstance(answer, dict):
        return _normalise_dict_answer(answer, sql_executor)
    if isinstance(answer, Sequence):
        return _normalise_row_list(list(answer))
    return None, f"unsupported answer type: {type(answer).__name__}"


def _normalise_string_answer(
    answer: str,
    sql_executor: Callable[[str], tuple[list[str], list[list[Any]]]] | None,
) -> tuple[tuple[list[str], list[list[Any]]] | None, str]:
    """Route a string answer either through SQL execution or scalar wrap."""
    stripped = answer.strip()
    if _looks_like_sql(stripped):
        if sql_executor is None:
            return None, "agent submitted SQL but no sql_executor was provided"
        try:
            cols, rows = sql_executor(stripped)
        except Exception as exc:
            return None, f"sql_executor raised {type(exc).__name__}: {exc}"
        return (list(cols), [list(r) for r in rows]), "SQL executed"
    parsed = _parse_json_rows(stripped)
    if parsed is not None:
        return _normalise_agent_answer(parsed, sql_executor)
    return ([""], [[stripped]]), "string scalar answer"


def _parse_json_rows(candidate: str) -> list[Any] | dict[str, Any] | None:
    """Return the structure a JSON-encoded answer holds, or None if it is text.

    Some models send ``submit_answer`` its rows as JSON text. The rows are the
    same answer, so scoring the text as one scalar would mark it wrong.
    """
    if not candidate.startswith(("[", "{")):
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, (list, dict)) else None


def _normalise_dict_answer(
    answer: dict[str, Any],
    sql_executor: Callable[[str], tuple[list[str], list[list[Any]]]] | None,
) -> tuple[tuple[list[str], list[list[Any]]] | None, str]:
    """Unpack the standard ``{columns, rows}`` shape and its common aliases."""
    if "columns" in answer and "rows" in answer:
        cols = [str(c) for c in answer["columns"]]
        rows = [list(r) for r in answer["rows"]]
        return (cols, rows), "dict {columns, rows}"
    for key in ("result", "answer", "data"):
        if key in answer:
            return _normalise_agent_answer(answer[key], sql_executor)
    return None, f"dict answer missing recognised keys: {sorted(answer)!r}"


def _normalise_row_list(
    rows: list[Any],
) -> tuple[tuple[list[str], list[list[Any]]] | None, str]:
    """Interpret a list as either a row list or a single row of scalars.

    Detects the common LLM habit of prepending a column-header row: if the
    first row is all-string and at least one subsequent row has non-string
    cells (or all rows have identical first-row values), it is peeled off
    as the ``columns`` argument.
    """
    if not rows:
        return ([""], []), "empty list answer"
    first = rows[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        normalised = [list(r) for r in rows]
        if _looks_like_header_row(normalised):
            cols = [str(v) for v in normalised[0]]
            return (cols, normalised[1:]), "list of rows with header row peeled"
        width = len(normalised[0])
        cols = [""] * width
        return (cols, normalised), "list of rows"
    return ([""], [[v] for v in rows]), "list of scalars (as column)"


def _looks_like_header_row(rows: list[list[Any]]) -> bool:
    """Heuristic: first row is all-string, and later rows contain non-strings.

    LLMs often submit ``[["count"], [150000]]``; without this heuristic
    the verifier double-counts the header row and reports a row-count
    mismatch. We only peel the header when we are highly confident: every
    cell in row 0 is a str, at least one subsequent cell is not a str,
    and every row has the same width.
    """
    if len(rows) < 2:
        return False
    header = rows[0]
    if not all(isinstance(v, str) for v in header):
        return False
    width = len(header)
    if not all(len(r) == width for r in rows[1:]):
        return False
    has_non_string_cell = any(
        not isinstance(v, str) for row in rows[1:] for v in row
    )
    return has_non_string_cell


def _looks_like_sql(candidate: str) -> bool:
    """Very light SQL sniff: starts with a SELECT/WITH/SHOW keyword."""
    head = candidate.lstrip().lower()
    return head.startswith(("select ", "with ", "show ", "select\n", "with\n"))


def _compare_result_sets(
    *,
    agent_columns: list[str],
    agent_rows: list[list[Any]],
    gold: GoldResult,
    tolerance: float,
) -> tuple[Verdict, str]:
    """Reconcile shape, column order, and row order then compare."""
    if len(gold.rows) != len(agent_rows):
        return "incorrect", (
            f"row count mismatch: gold={len(gold.rows)}, agent={len(agent_rows)}"
        )
    if not gold.rows:
        return "correct", "both empty"
    width = len(gold.rows[0])
    for i, row in enumerate(agent_rows):
        if len(row) != width:
            return "incorrect", (
                f"row {i} width mismatch: gold={width}, agent={len(row)}"
            )
    permuted_rows = _align_columns(agent_columns, agent_rows, gold.columns)
    order_sensitive = bool(_ORDER_BY_PATTERN.search(gold.gold_sql))
    return _compare_rows(
        gold_rows=gold.rows,
        agent_rows=permuted_rows,
        order_sensitive=order_sensitive,
        tolerance=tolerance,
    )


def _align_columns(
    agent_columns: list[str],
    agent_rows: list[list[Any]],
    gold_columns: list[str],
) -> list[list[Any]]:
    """Permute agent columns to match gold when column names line up.

    Falls back to the agent's original column order when either side has
    unnamed columns or names do not match — comparison then proceeds
    positionally on cell values.
    """
    if not gold_columns or not agent_columns:
        return agent_rows
    if len(gold_columns) != len(agent_columns):
        return agent_rows
    gold_names = [_canon_col(c) for c in gold_columns]
    agent_names = [_canon_col(c) for c in agent_columns]
    if any(n == "" for n in gold_names) or any(n == "" for n in agent_names):
        return agent_rows
    if sorted(gold_names) != sorted(agent_names):
        return agent_rows
    permutation = [agent_names.index(name) for name in gold_names]
    return [[row[i] for i in permutation] for row in agent_rows]


def _canon_col(name: str) -> str:
    """Canonical form for column-name comparison."""
    return str(name).strip().lower()


def _compare_rows(
    *,
    gold_rows: list[list[Any]],
    agent_rows: list[list[Any]],
    order_sensitive: bool,
    tolerance: float,
) -> tuple[Verdict, str]:
    """Compare row lists either as sequences or as multisets."""
    if order_sensitive:
        for i, (g_row, a_row) in enumerate(zip(gold_rows, agent_rows, strict=True)):
            for j, (g_val, a_val) in enumerate(zip(g_row, a_row, strict=True)):
                if not _cells_equal(g_val, a_val, tolerance):
                    return "incorrect", (
                        f"cell mismatch at row {i} col {j}: "
                        f"gold={g_val!r}, agent={a_val!r}"
                    )
        return "correct", "row-by-row ordered match"
    return _compare_as_multiset(gold_rows, agent_rows, tolerance)


def _compare_as_multiset(
    gold_rows: list[list[Any]],
    agent_rows: list[list[Any]],
    tolerance: float,
) -> tuple[Verdict, str]:
    """Match every gold row to a distinct agent row modulo tolerance."""
    unmatched_agent = list(range(len(agent_rows)))
    for g_row in gold_rows:
        match_idx = _find_row_match(g_row, agent_rows, unmatched_agent, tolerance)
        if match_idx is None:
            return "incorrect", (
                f"gold row {g_row!r} has no match in agent rows"
            )
        unmatched_agent.remove(match_idx)
    return "correct", "multiset match"


def _find_row_match(
    gold_row: list[Any],
    agent_rows: list[list[Any]],
    candidates: list[int],
    tolerance: float,
) -> int | None:
    """Return the index of an agent row that equals ``gold_row``, or None."""
    for idx in candidates:
        if _rows_equal(gold_row, agent_rows[idx], tolerance):
            return idx
    return None


def _rows_equal(a: list[Any], b: list[Any], tolerance: float) -> bool:
    """Positional cell-by-cell equality with tolerance."""
    if len(a) != len(b):
        return False
    return all(_cells_equal(x, y, tolerance) for x, y in zip(a, b, strict=True))


def _cells_equal(a: Any, b: Any, tolerance: float) -> bool:
    """Equality with numeric tolerance and lenient string handling."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_num = _coerce_number(a)
    b_num = _coerce_number(b)
    if a_num is not None and b_num is not None:
        return abs(a_num - b_num) <= tolerance
    return _canon_str(a) == _canon_str(b)


def _coerce_number(value: Any) -> float | None:
    """Return ``value`` as float if it parses as a number, else None.

    Booleans are excluded so ``True == 1`` does not accidentally succeed
    across cell types.
    """
    if isinstance(value, bool):
        return None
    # Database drivers return NUMERIC/DECIMAL columns as Decimal, so executed
    # SQL answers must compare numerically against the float gold values.
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _canon_str(value: Any) -> str:
    """Canonical string form for lenient equality."""
    canonical = str(value).strip().lower()
    return _MIDNIGHT_TIMESTAMP_PATTERN.sub(r"\1", canonical)


__all__ = ["GoldResult", "Verdict", "verify"]
