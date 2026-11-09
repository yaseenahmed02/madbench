"""Phase enum — closed set of protocol-tax phases.

The enum is closed (not extensible by plugins) because the accounting identity
T_total = T_db + sum(phases) + epsilon sums over exactly these phases. Adding a
phase would change the meaning of $T_{\\text{total}}$ and invalidate cross-server comparisons.
"""

from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    """One of the six measurable protocol-tax phases.

    Each phase maps 1:1 to an OTel span name. The mapping is the
    contract between the plugin SDK (which yields PhaseContext entries) and
    the instrumentation layer (which emits spans).
    """

    DISCOVERY = "discovery"
    LIFECYCLE = "lifecycle"
    HANDOVER = "handover"
    SERIALIZE = "serialize"
    PARSE = "parse"
    TRANSPORT = "transport"

    @property
    def span_name(self) -> str:
        """Return the OTel span name for this phase."""
        return _PHASE_TO_SPAN[self]

    @classmethod
    def from_span_name(cls, span_name: str) -> Phase:
        """Inverse of `span_name`. Raises KeyError on unknown span names."""
        try:
            return _SPAN_TO_PHASE[span_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown phase span name '{span_name}'. "
                f"Valid: {sorted(_SPAN_TO_PHASE)}"
            ) from exc


_PHASE_TO_SPAN: dict[Phase, str] = {
    Phase.DISCOVERY: "mcp.discovery",
    Phase.LIFECYCLE: "mcp.lifecycle.connect",
    Phase.HANDOVER: "mcp.tool_call",
    Phase.SERIALIZE: "mcp.serialization",
    Phase.PARSE: "mcp.parse",
    Phase.TRANSPORT: "mcp.transport",
}

_SPAN_TO_PHASE: dict[str, Phase] = {span: phase for phase, span in _PHASE_TO_SPAN.items()}
