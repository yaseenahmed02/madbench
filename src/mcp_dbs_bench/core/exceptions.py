"""Custom exception hierarchy for MCP-DBS-Bench.

All domain errors derive from MCPDBSBenchError so callers can catch the entire
family with one clause. Each subclass carries actionable context (which phase,
which server, which threshold) so error messages can answer: what happened,
what was expected, what was the input, what should the caller do.
"""

from __future__ import annotations


class MCPDBSBenchError(Exception):
    """Root of the MCP-DBS-Bench exception hierarchy."""


class ConfigurationError(MCPDBSBenchError):
    """Configuration validation failed at startup (Pydantic model errors,
    missing required fields, mutually-exclusive options set, etc.)."""


class PluginRegistrationError(MCPDBSBenchError):
    """A plugin entry point failed to load or does not conform to
    `MCPServerPlugin`. Raised during registry scan, before any measurement."""


class BackendDriverMissingError(MCPDBSBenchError):
    """A backend's optional driver (psycopg, pymysql, etc.) is not installed.

    Attributes:
        backend: The backend identifier (e.g. ``"postgres"``).
        install_hint: Shell command to install the missing extra.
    """

    def __init__(self, backend: str, install_hint: str) -> None:
        self.backend = backend
        self.install_hint = install_hint
        super().__init__(
            f"{backend} backend requires an optional driver that is not installed. "
            f"Install it with: {install_hint}"
        )


class ExclusivityViolationError(MCPDBSBenchError):
    """Property 2 (Exclusivity) failed: two non-transport phases overlapped.

    The PhaseContext enforces this at runtime; the calibration pipeline also
    re-checks it post-trace. Either source raises this exception.

    Attributes:
        phase_a: Name of the first overlapping phase.
        phase_b: Name of the second overlapping phase.
        overlap_ns: Overlap duration in nanoseconds. ``None`` when the
            violation is detected at enter-time (the second phase has not
            yet completed, so a literal overlap cannot be computed). The
            post-hoc verifier (:func:`instrumentation.properties.verify_exclusivity`)
            populates the literal value when re-checking the trace.
    """

    def __init__(
        self, phase_a: str, phase_b: str, overlap_ns: int | None = None
    ) -> None:
        self.phase_a = phase_a
        self.phase_b = phase_b
        self.overlap_ns = overlap_ns
        if overlap_ns is None:
            detail = (
                "second phase still open; post-hoc overlap available via "
                "verify_exclusivity"
            )
        else:
            detail = f"by {overlap_ns} ns"
        super().__init__(
            f"Property 2 violation: phases '{phase_a}' and '{phase_b}' overlap "
            f"{detail}. Non-transport phases must be pairwise disjoint."
        )


class ResidualExceededError(MCPDBSBenchError):
    """Property 1 (Completeness) failed: |epsilon|/T_total >= threshold.

    Attributes:
        residual_ms: epsilon, the unaccounted time, in milliseconds.
        total_ms: T_total, the wall-clock duration of the interaction.
        threshold: The fraction (default 0.05).
    """

    def __init__(self, residual_ms: float, total_ms: float, threshold: float) -> None:
        self.residual_ms = residual_ms
        self.total_ms = total_ms
        self.threshold = threshold
        fraction = abs(residual_ms) / total_ms if total_ms > 0 else float("inf")
        super().__init__(
            f"Property 1 violation: |epsilon|/T_total = {fraction:.4f} "
            f"exceeds threshold {threshold:.2f} "
            f"(epsilon={residual_ms:.3f} ms, T_total={total_ms:.3f} ms)"
        )


class PhaseBoundaryError(MCPDBSBenchError):
    """A phase was entered without a matching exit, or exited without entry.

    Attributes:
        phase: The phase identifier (e.g. ``"DISCOVERY"``).
    """

    def __init__(self, phase: str, detail: str) -> None:
        self.phase = phase
        super().__init__(f"Phase '{phase}' boundary error: {detail}")


class CalibrationError(MCPDBSBenchError):
    """Calibration (delta_obs > threshold) failed for a (server, backend, query)
    combination. Primary measurement must not proceed for the failing cell."""
