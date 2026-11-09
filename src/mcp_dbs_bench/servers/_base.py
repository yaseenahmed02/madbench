"""Shared utilities for built-in MCP server adapters.

Two flavors of adapter are supported:

1. **Subprocess adapters** — wrap an upstream MCP server (Node, Go, Python)
   running as a stdio child process. The adapter speaks JSON-RPC 2.0 over
   stdin/stdout, timestamps each message at the byte boundary, and recovers
   the protocol phase boundaries from the message sequence.

2. **In-process adapters** — for upstream servers whose source is a Python
   library, the adapter calls the library directly. Phase boundaries are
   explicit because the adapter controls the code path. Traces are still
   wire-format identical because the JSON-RPC envelope is constructed
   manually before tokenization / byte-count measurement.

Both flavors satisfy the same plugin Protocol. The trace shape is invariant
across them; only T_xport differs (sub-microsecond for in-process, real IPC
for subprocess).
"""

from __future__ import annotations

import json
from typing import Any


def encode_jsonrpc_tools_list_response(tools: list[dict[str, Any]], request_id: int = 1) -> bytes:
    """Build a JSON-RPC 2.0 `tools/list` response envelope.

    Used by in-process adapters so the byte count and token count match what
    a stdio-emitting server would produce.
    """
    envelope = {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}}
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def encode_jsonrpc_tool_call_response(
    *,
    columns: list[str],
    rows: list[tuple[Any, ...]],
    request_id: int = 1,
    output_format: str = "json_flat_table",
) -> bytes:
    """Build a JSON-RPC 2.0 `tools/call` response with the given result set.

    Output shape is controlled by ``output_format``:
        - json_flat_table:  {"columns": [...], "rows": [[...], ...]}
        - json_per_row:     [{"col1": v1, "col2": v2}, ...]
        - csv:              header row + CSV
    """
    if output_format == "json_per_row":
        body: Any = [dict(zip(columns, row, strict=False)) for row in rows]
    elif output_format == "csv":
        lines = [",".join(columns)]
        lines.extend(",".join(_csv_cell(v) for v in row) for row in rows)
        body = "\n".join(lines)
    else:  # default json_flat_table
        body = {"columns": list(columns), "rows": [list(r) for r in rows]}

    envelope = {"jsonrpc": "2.0", "id": request_id, "result": body}
    return json.dumps(envelope, separators=(",", ":"), default=_json_default).encode("utf-8")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    if any(c in s for c in (",", '"', "\n")):
        return '"' + s.replace('"', '""') + '"'
    return s


def _json_default(value: Any) -> Any:
    """JSON encoder fallback for Decimal, datetime, etc."""
    import datetime
    from decimal import Decimal

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return str(value)
