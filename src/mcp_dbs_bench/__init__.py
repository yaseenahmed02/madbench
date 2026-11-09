"""MADBench: Protocol Tax decomposition benchmark for MCP database servers.

The public API surfaces the core data model (spans, phases) and the plugin
SDK for third-party MCP server adapters. The harness and instrumentation are
driven by the scripts under ``scripts/``.

Stable public surface:
    Plugin SDK
        ``MCPServerPlugin``         - structural Protocol every adapter satisfies
        ``Phase``                   - the six phase enum members + ``mcp.interaction``
        ``PhaseContext``            - plugin-facing span recorder protocol
        ``PhaseRecord``             - immutable record of one phase entry

    Types
        ``ConnectionHandle``        - opaque per-connection plugin state
        ``ConnectionModel``         - implicit-pool / gateway / etc.
        ``OutputFormat``            - JSON / CSV / TSV / markdown
        ``ServerCapabilities``      - plugin-emitted capability declaration
        ``ToolCallRequest``         - outbound tools/call envelope
        ``ToolCallResponse``        - inbound tools/call result
        ``ToolManifest``            - tools/list response

    Plugin validation
        ``validate_plugin``         - plugin conformance check
"""

from __future__ import annotations

__version__ = "1.0.0"

from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseContext, PhaseRecord
from mcp_dbs_bench.core.plugin import MCPServerPlugin
from mcp_dbs_bench.core.types import (
    ConnectionHandle,
    ConnectionModel,
    OutputFormat,
    ServerCapabilities,
    ToolCallRequest,
    ToolCallResponse,
    ToolManifest,
)
from mcp_dbs_bench.core.validate import (
    ValidationReport,
    validate_plugin,
)

__all__ = [
    "ConnectionHandle",
    "ConnectionModel",
    "MCPServerPlugin",
    "OutputFormat",
    "Phase",
    "PhaseContext",
    "PhaseRecord",
    "ServerCapabilities",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolManifest",
    "ValidationReport",
    "__version__",
    "validate_plugin",
]
