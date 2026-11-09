"""Tests for the custom exception hierarchy.

The exception hierarchy is the contract for error handling across modules:
every domain error inherits from MCPDBSBenchError so callers can use a single
`except MCPDBSBenchError` to catch everything domain-specific.
"""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
    "exception_cls",
    [
        BackendDriverMissingError,
        CalibrationError,
        ConfigurationError,
        ExclusivityViolationError,
        PhaseBoundaryError,
        PluginRegistrationError,
        ResidualExceededError,
    ],
)
def test_all_domain_errors_inherit_from_root(exception_cls):
    assert issubclass(exception_cls, MCPDBSBenchError)


def test_root_inherits_from_exception():
    assert issubclass(MCPDBSBenchError, Exception)


def test_residual_exceeded_carries_residual_and_total():
    err = ResidualExceededError(residual_ms=12.5, total_ms=100.0, threshold=0.05)
    assert err.residual_ms == 12.5
    assert err.total_ms == 100.0
    assert err.threshold == 0.05
    # Message should be actionable (mentions the fraction and threshold).
    assert "0.05" in str(err)
    assert "0.125" in str(err) or "12.5%" in str(err)


def test_exclusivity_violation_carries_phase_pair():
    err = ExclusivityViolationError(
        phase_a="discovery", phase_b="handover", overlap_ns=500_000
    )
    assert err.phase_a == "discovery"
    assert err.phase_b == "handover"
    assert err.overlap_ns == 500_000
    assert "discovery" in str(err)
    assert "handover" in str(err)


def test_phase_boundary_error_identifies_phase():
    err = PhaseBoundaryError("LIFECYCLE", "entered without exit")
    assert err.phase == "LIFECYCLE"
    assert "LIFECYCLE" in str(err)


def test_backend_driver_missing_carries_install_hint():
    err = BackendDriverMissingError("postgres", "pip install 'mcp-dbs-bench[postgres]'")
    assert err.backend == "postgres"
    assert "pip install" in str(err)
