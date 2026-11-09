"""Public-API smoke tests for ``mcp_dbs_bench.core``.

Every documented public symbol must import cleanly from the package root.
"""

from __future__ import annotations


def test_phase_imports_from_package_root():
    from mcp_dbs_bench.core import Phase  # noqa: F401


def test_phase_context_protocols_import_from_package_root():
    from mcp_dbs_bench.core import PhaseContext, PhaseContextImpl, PhaseRecord  # noqa: F401


def test_plugin_protocol_imports_from_package_root():
    from mcp_dbs_bench.core import MCPServerPlugin  # noqa: F401


def test_data_carriers_import_from_package_root():
    from mcp_dbs_bench.core import (  # noqa: F401
        ConnectionHandle,
        ConnectionModel,
        OutputFormat,
        ServerCapabilities,
        ToolCallRequest,
        ToolCallResponse,
        ToolManifest,
    )


def test_exceptions_import_from_package_root():
    from mcp_dbs_bench.core import (  # noqa: F401
        BackendDriverMissingError,
        CalibrationError,
        ConfigurationError,
        ExclusivityViolationError,
        MCPDBSBenchError,
        PhaseBoundaryError,
        PluginRegistrationError,
        ResidualExceededError,
    )


def test_clock_helpers_import_from_package_root():
    from mcp_dbs_bench.core import monotonic_now_ns, ns_to_ms  # noqa: F401


def test_timing_helpers_import_from_package_root():
    from mcp_dbs_bench.core import TimingDecomposition, decompose  # noqa: F401


def test_re_exported_symbols_match_submodules():
    """Re-exports must be the SAME object as the submodule definition."""
    from mcp_dbs_bench import core
    from mcp_dbs_bench.core import phase, phase_context, plugin

    assert core.Phase is phase.Phase
    assert core.PhaseContext is phase_context.PhaseContext
    assert core.MCPServerPlugin is plugin.MCPServerPlugin
