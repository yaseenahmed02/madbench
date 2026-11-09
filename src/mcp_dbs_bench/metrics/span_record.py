"""SpanRecord: the validated in-memory span shape every metric consumes.

This is the boundary between the instrumentation layer (which emits OTel spans)
and the metric layer (which projects spans to numbers). Decoupling the metric
layer from the OTel SDK keeps it pure and testable.

When core/spans.py lands from the instrumentation owner, this module will
re-export from there for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp_dbs_bench.core.clock import NS_PER_MS


@dataclass(frozen=True, slots=True)
class SpanRecord:
    """A single OTel-style span, normalized for metric computation.

    Attributes are stored as a plain ``dict[str, Any]`` keyed by the OTel
    dotted attribute name (e.g. ``"response.tokens"``). Callers are expected
    to validate attribute presence via the Pydantic schemas in
    ``instrumentation/span_schema.py`` before constructing a SpanRecord.

    Attributes:
        name: OTel span name (e.g. ``"mcp.tool_call"``).
        start_ns: Monotonic-clock start timestamp in nanoseconds.
        end_ns: Monotonic-clock end timestamp in nanoseconds. Must be >= start_ns.
        attributes: Mapping of attribute name to value.
        parent_span_id: Optional id of the parent span; None for root spans.
        span_id: Optional id of this span. Used to reconstruct trees.
    """

    name: str
    start_ns: int
    end_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_span_id: str | None = None
    span_id: str | None = None

    def __post_init__(self) -> None:
        if self.end_ns < self.start_ns:
            raise ValueError(
                f"end_ns ({self.end_ns}) must be >= start_ns ({self.start_ns}) "
                f"for span '{self.name}'"
            )

    @property
    def duration_ns(self) -> int:
        """Span duration in nanoseconds."""
        return self.end_ns - self.start_ns

    @property
    def duration_ms(self) -> float:
        """Span duration in milliseconds."""
        return self.duration_ns / NS_PER_MS
