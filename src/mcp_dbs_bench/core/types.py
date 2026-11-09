"""Plugin-SDK data types — the wire format between plugins and the harness.

All carriers are frozen dataclasses (immutable, slots-enabled) so a plugin
cannot mutate them mid-tool-call and corrupt the trace record. Construction-
time validation enforces non-negative counts and required enum membership.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputFormat(str, Enum):
    """Self-declared output format. Used to compute IM-1 (phi)
    and CM-2 (rho) without parsing semantics the harness does not control."""

    JSON_PER_ROW = "json_per_row"
    JSON_COLUMNAR = "json_columnar"
    JSON_FLAT_TABLE = "json_flat_table"
    CSV = "csv"
    TSV = "tsv"
    VERTICAL_TEXT = "vertical_text"
    MARKDOWN = "markdown"


class ConnectionModel(str, Enum):
    """Connection model the server uses."""

    IMPLICIT_POOL = "implicit_pool"
    GATEWAY = "gateway"
    EXPLICIT_LIFECYCLE = "explicit_lifecycle"
    SESSION = "session"
    CENTRAL_POOL = "central_pool"
    SQLALCHEMY = "sqlalchemy"
    DIRECT = "direct"


@dataclass(frozen=True, slots=True)
class ServerCapabilities:
    """Self-declared properties of an MCP server, used to compute tier-1 metrics.

    Validation:
        declared_tool_count: must be non-negative. Validated against the live
            tools/list response by ``validate_plugin``.
    """

    output_format: OutputFormat
    connection_model: ConnectionModel
    declared_tool_count: int
    supports_streaming: bool
    safety_mechanism: str
    schema_depth: str

    def __post_init__(self) -> None:
        if self.declared_tool_count < 0:
            raise ValueError(
                f"declared_tool_count must be non-negative; got {self.declared_tool_count}"
            )
        if not self.safety_mechanism:
            raise ValueError("safety_mechanism must be a non-empty string")
        if self.schema_depth not in {"none", "basic", "rich", "hidden"}:
            raise ValueError(
                f"schema_depth must be one of none|basic|rich|hidden; got {self.schema_depth!r}"
            )


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """Outbound tool/call invocation: name + JSON-encodable arguments.

    `byte_size()` returns the JSON-canonical byte count used as the
    `request.bytes` attribute on the `mcp.tool_call` span.
    """

    name: str
    arguments: dict[str, Any]

    def byte_size(self) -> int:
        """Return the canonical JSON encoded byte length (sorted keys)."""
        payload = {"name": self.name, "arguments": self.arguments}
        return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class ToolCallResponse:
    """Inbound tool/call result: materialized rows plus the raw payload bytes.

    `raw_bytes` is what arrived on the wire; the metric layer uses it to
    compute IM-2 (B_mcp), IM-3 inputs, and tokenization (IM-1, IM-9, IM-8).
    `rows` is the parsed view used to compute `tau*_result`.
    """

    rows: list[tuple[Any, ...]]
    raw_bytes: bytes
    column_names: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def byte_size(self) -> int:
        return len(self.raw_bytes)


@dataclass(frozen=True, slots=True)
class ToolManifest:
    """tools/list response: the set of tools the server exposes.

    `raw_bytes` preserves the wire payload so the harness can record
    `response.bytes` and `response.tokens` on the `mcp.discovery` span.
    """

    tools: list[dict[str, Any]] = field(default_factory=list)
    raw_bytes: bytes = b""

    @property
    def tool_count(self) -> int:
        return len(self.tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(t.get("name", "") for t in self.tools)

    @classmethod
    def from_jsonrpc(cls, payload: dict[str, Any]) -> ToolManifest:
        """Construct from a JSON-RPC 2.0 ``tools/list`` response envelope.

        Args:
            payload: Parsed JSON-RPC response. Must contain ``result.tools``.

        Returns:
            A ToolManifest. ``raw_bytes`` is set to the JSON encoding of the
            full payload so the harness can compute response.bytes.

        Raises:
            ValueError: If ``payload`` is missing the result.tools field.
        """
        result = payload.get("result")
        if not isinstance(result, dict) or "tools" not in result:
            raise ValueError(
                "JSON-RPC payload missing 'result.tools' field; "
                f"keys present: {sorted(payload.keys())}"
            )
        tools = result["tools"]
        if not isinstance(tools, list):
            raise ValueError(f"result.tools must be a list; got {type(tools).__name__}")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(tools=tools, raw_bytes=raw)


@dataclass(slots=True)
class ConnectionHandle:
    """Opaque per-server connection state.

    The harness treats this as a black box; plugins put whatever they need
    inside `value` (a client object, a process handle, an asyncio queue).
    Not frozen because the plugin may need to mutate connection-internal state.
    """

    value: Any
