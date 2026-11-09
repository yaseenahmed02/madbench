"""Core data model: spans, phases, timing decomposition, plugin protocol.

This package is the dependency root. Higher layers (instrumentation, harness,
backends, servers) depend downward on `core`; `core` depends only on the
standard library and Pydantic.

Public symbols listed in ``__all__`` form the stable import surface. The
submodules remain reachable for power users, but new code should import from
the package root.
"""

from __future__ import annotations

from mcp_dbs_bench.core.clock import monotonic_now_ns, ns_to_ms
from mcp_dbs_bench.core.exceptions import (
    BackendDriverMissingError,
    CalibrationError,
    ConfigurationError,
    ExclusivityViolationError,
    MCPDBSBenchError,
    PhaseBoundaryError,
    PluginRegistrationError,
    ResidualExceededError,
)
from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseContext, PhaseContextImpl, PhaseRecord
from mcp_dbs_bench.core.plugin import MCPServerPlugin
from mcp_dbs_bench.core.timing import TimingDecomposition, decompose
from mcp_dbs_bench.core.types import (
    ConnectionHandle,
    ConnectionModel,
    OutputFormat,
    ServerCapabilities,
    ToolCallRequest,
    ToolCallResponse,
    ToolManifest,
)

__all__ = [
    "BackendDriverMissingError",
    "CalibrationError",
    "ConfigurationError",
    "ConnectionHandle",
    "ConnectionModel",
    "ExclusivityViolationError",
    "MCPDBSBenchError",
    "MCPServerPlugin",
    "OutputFormat",
    "Phase",
    "PhaseBoundaryError",
    "PhaseContext",
    "PhaseContextImpl",
    "PhaseRecord",
    "PluginRegistrationError",
    "ResidualExceededError",
    "ServerCapabilities",
    "TimingDecomposition",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolManifest",
    "decompose",
    "monotonic_now_ns",
    "ns_to_ms",
]
