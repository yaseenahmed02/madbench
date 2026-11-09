"""Parse upstream MCP server text payloads into row+column tuples.

Each upstream server emits a different text shape inside the ``content[].text``
block of a JSON-RPC ``tools/call`` response:

  - mcp-alchemy: vertical text ("1. row\\ncol: value\\n\\nResult: N rows")
  - dbhub: JSON envelope {"success": ..., "data": {"rows": [...], "count": ...}}
  - mcp-server-postgres: JSON array of row-objects

The harness's ``ToolCallResponse`` contract requires rows as tuples and
column names as a tuple of strings, so the wrapper must re-parse the text
payload back to that shape. ``raw_bytes`` on the response object remains the
exact wire payload of the JSON-RPC envelope, untouched.
"""

from __future__ import annotations

import json
import re
from typing import Any

_ALCHEMY_RESULT_RE = re.compile(r"^Result:\s*\d+\s*rows?\s*$", re.MULTILINE)


def parse_dbhub_payload(text: str) -> tuple[list[tuple[Any, ...]], tuple[str, ...]]:
    """Parse dbhub's text payload: a JSON envelope with rows[] and count.

    Examples:
        >>> txt = '{"success": true, "data": {"rows": [{"x": 1}], "count": 1}}'
        >>> rows, cols = parse_dbhub_payload(txt)
        >>> rows
        [(1,)]
        >>> cols
        ('x',)
    """
    payload = _safe_json_loads(text, label="dbhub")
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    rows_raw = data.get("rows", []) if isinstance(data, dict) else []
    if not isinstance(rows_raw, list):
        return [], ()
    return _rows_from_row_objects(rows_raw)


def parse_anthropic_pg_payload(text: str) -> tuple[list[tuple[Any, ...]], tuple[str, ...]]:
    """Parse mcp-server-postgres' text payload: a JSON array of row objects."""
    payload = _safe_json_loads(text, label="anthropic-pg")
    if not isinstance(payload, list):
        return [], ()
    return _rows_from_row_objects(payload)



def parse_mcp_alchemy_payload(text: str) -> tuple[list[tuple[Any, ...]], tuple[str, ...]]:
    """Parse mcp-alchemy's vertical text payload.

    Format::

        1. row
        col_a: value_a
        col_b: value_b

        2. row
        col_a: value_a2
        col_b: value_b2

        Result: 2 rows

    Returns string-typed cells; the upstream emits a textual representation
    so numeric type recovery is best-effort. For ``SELECT 1`` and similar
    smoke queries the cells are simple integers that we coerce.
    """
    body, _, _ = text.partition("Result:")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    row_blocks = [b for b in blocks if re.match(r"^\d+\.\s*row", b)]
    columns: list[str] = []
    rows: list[tuple[Any, ...]] = []
    for block in row_blocks:
        lines = block.splitlines()[1:]  # skip "N. row"
        cells: dict[str, Any] = {}
        for line in lines:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            cells[key.strip()] = _coerce_cell(value.strip())
        if not columns:
            columns = list(cells.keys())
        rows.append(tuple(cells.get(c) for c in columns))
    return rows, tuple(columns)


def _rows_from_row_objects(
    objs: list[Any],
) -> tuple[list[tuple[Any, ...]], tuple[str, ...]]:
    """Convert a list of ``{col: value}`` dicts to tuples + column tuple."""
    if not objs or not isinstance(objs[0], dict):
        return [], ()
    columns = tuple(objs[0].keys())
    rows = [tuple(obj.get(c) for c in columns) for obj in objs if isinstance(obj, dict)]
    return rows, columns


def _safe_json_loads(text: str, *, label: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {} if label != "anthropic-pg" else []


def _coerce_cell(value: str) -> Any:
    """Best-effort scalar coercion for mcp-alchemy's textual cell values."""
    if value in ("", "None", "null", "NULL"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


__all__ = [
    "parse_anthropic_pg_payload",
    "parse_dbhub_payload",
    "parse_mcp_alchemy_payload",
]
