"""Track A harness: deterministic SQL-through-MCP-server execution.

The harness is a pure orchestrator. It composes a registered server plugin
with a registered backend, runs one ``mcp.interaction`` end-to-end, and
emits OTel spans through the OTelPhaseContext. The harness is async because
the MCP wire protocol is naturally async; one query at a time per process.
"""

from __future__ import annotations

from mcp_dbs_bench.harness.runner import (
    InteractionResult,
    TrackARunner,
    randomized_run_order,
)

__all__ = ["InteractionResult", "TrackARunner", "randomized_run_order"]
