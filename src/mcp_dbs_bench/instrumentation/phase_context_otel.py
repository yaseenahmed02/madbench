"""OTel binding for the PhaseContext state machine.

Adds two responsibilities to the pure-data PhaseContextImpl:

  1. Open / close an OTel span for each phase, using `Phase.span_name`.
  2. Forward the optional attr dict onto the span at exit time.

The harness wires one OTelPhaseContext per `mcp.interaction`. Plugins use
the `span(phase, **attrs)` context manager; the harness inspects
``self.records`` to feed the timing decomposition.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseContextImpl, PhaseRecord
from mcp_dbs_bench.instrumentation.tracer import OTelTracer


class OTelPhaseContext:
    """OTel-emitting PhaseContext. Conforms to `core.phase_context.PhaseContext`.

    Composition over inheritance: a PhaseContextImpl owns the state machine
    and exclusivity enforcement; this class adds OTel span emission as the
    second responsibility. Keeping them separated means pure-data tests
    don't need the OTel SDK.
    """

    def __init__(self, tracer: OTelTracer) -> None:
        self._tracer = tracer
        self._inner = PhaseContextImpl()

    @property
    def records(self) -> list[PhaseRecord]:
        return self._inner.records

    def enter(self, phase: Phase, *, attrs: dict[str, Any] | None = None) -> None:
        # Bare enter/exit support exists for parity with the Protocol, but
        # callers should prefer `span()` so OTel span lifecycle is automatic.
        self._inner.enter(phase, attrs=attrs)

    def exit(self, phase: Phase, *, attrs: dict[str, Any] | None = None) -> None:
        self._inner.exit(phase, attrs=attrs)

    @contextmanager
    def span(self, phase: Phase, **attrs: Any) -> Iterator[dict[str, Any]]:
        """Open an OTel span and a PhaseContextImpl interval, exit both on close.

        Yields the live attribute dict (see ``PhaseContextImpl.span``) so
        callers may set attributes that are only known after the wrapped
        code runs (e.g. ``t_db_ms`` after ``backend.execute`` returns).
        Both the OTel span and the PhaseRecord receive the final attribute
        values at close time.
        """
        live_attrs = dict(attrs)
        with self._tracer.start_as_current_span(phase.span_name) as otel_span:
            self._inner.enter(phase, attrs=live_attrs)
            try:
                yield live_attrs
            finally:
                for key, value in live_attrs.items():
                    otel_span.set_attribute(key, value)
                self._inner.exit(phase, attrs=live_attrs)
