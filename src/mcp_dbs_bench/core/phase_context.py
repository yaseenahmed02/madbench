"""PhaseContext: the load-bearing state machine for Property 2 enforcement.

Plugins do not call OTel APIs directly. Instead they call `enter(phase)` and
`exit(phase)` (or use the `span(phase)` context manager). This class:

  1. Records monotonic timestamps on each call.
  2. Enforces exclusivity: at most one non-transport phase open at a time.
   (TRANSPORT is exempt because the timing model allows it to overlap.)
  3. Detects unmatched enter/exit pairs (PhaseBoundaryError).
  4. Surfaces a list of PhaseRecords that the harness translates into spans.

This is a pure data-structure module — no OTel imports, no I/O. The OTel
binding lives in `instrumentation/phase_context_otel.py`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from mcp_dbs_bench.core.clock import monotonic_now_ns
from mcp_dbs_bench.core.exceptions import (
    ExclusivityViolationError,
    PhaseBoundaryError,
)
from mcp_dbs_bench.core.phase import Phase


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    """Immutable record of one phase entry — output of PhaseContext.

    The harness translates PhaseRecords into OTel spans. The metric layer
    aggregates them into the eight terms of the timing identity.
    """

    phase: Phase
    start_ns: int
    end_ns: int
    attrs: dict[str, Any]

    def __post_init__(self) -> None:
        if self.end_ns < self.start_ns:
            raise ValueError(
                f"end_ns ({self.end_ns}) must be >= start_ns ({self.start_ns}) "
                f"for phase {self.phase.name}"
            )

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


class PhaseContext(Protocol):
    """Plugin-facing hook for marking phase boundaries.

    The plugin MUST call `enter(phase)` immediately before, and `exit(phase)`
    immediately after, the code region whose wall-clock time defines the
    phase. The harness records timestamps and enforces
    Property 2 (exclusivity) on each call.
    """

    def enter(self, phase: Phase, *, attrs: dict[str, Any] | None = None) -> None: ...

    def exit(self, phase: Phase, *, attrs: dict[str, Any] | None = None) -> None: ...

    def span(self, phase: Phase, **attrs: Any) -> Any: ...


class PhaseContextImpl:
    """Concrete PhaseContext that records intervals in memory.

    TRANSPORT is allowed to overlap any other phase because the span model places
    `mcp.transport` as a sibling of `mcp.tool_call`. All other phases must
    be strictly disjoint; overlap raises ExclusivityViolationError immediately.
    """

    def __init__(self) -> None:
        self._records: list[PhaseRecord] = []
        # Map of currently-open phase -> (start_ns, attrs).
        self._open: dict[Phase, tuple[int, dict[str, Any]]] = {}

    @property
    def records(self) -> list[PhaseRecord]:
        """Return a defensive copy of the recorded phase intervals."""
        return list(self._records)

    def enter(self, phase: Phase, *, attrs: dict[str, Any] | None = None) -> None:
        """Mark the start of a phase. See class docstring for exclusivity rules.

        Raises:
            ExclusivityViolationError: A non-TRANSPORT phase is already open
                and the new phase is also not TRANSPORT.
            PhaseBoundaryError: This phase is already open (double-enter).
        """
        if phase in self._open:
            raise PhaseBoundaryError(
                phase.name, f"already entered at ns={self._open[phase][0]}"
            )
        if phase is not Phase.TRANSPORT:
            for open_phase in self._open:
                if open_phase is not Phase.TRANSPORT:
                    # overlap_ns is None at enter time: the second phase
                    # has not yet completed so we cannot compute a literal
                    # overlap. The post-hoc verifier populates this when
                    # re-checking the trace.
                    raise ExclusivityViolationError(
                        phase_a=open_phase.name.lower(),
                        phase_b=phase.name.lower(),
                        overlap_ns=None,
                    )
        self._open[phase] = (monotonic_now_ns(), dict(attrs) if attrs else {})

    def exit(self, phase: Phase, *, attrs: dict[str, Any] | None = None) -> None:
        """Mark the end of a phase. Merges optional exit-time attrs into the record.

        Raises:
            PhaseBoundaryError: This phase was not previously entered.
        """
        if phase not in self._open:
            open_names = sorted(p.name for p in self._open)
            raise PhaseBoundaryError(
                phase.name,
                f"exit without matching enter (currently open: {open_names or 'none'})",
            )
        start_ns, merged_attrs = self._open.pop(phase)
        if attrs:
            merged_attrs.update(attrs)
        self._records.append(
            PhaseRecord(
                phase=phase,
                start_ns=start_ns,
                end_ns=monotonic_now_ns(),
                attrs=merged_attrs,
            )
        )

    @contextmanager
    def span(self, phase: Phase, **attrs: Any) -> Iterator[dict[str, Any]]:
        """Scope guard: enter on ``__enter__``, exit on ``__exit__``.

        The recommended plugin-author idiom. Guarantees ``exit()`` runs
        even if the wrapped code raises, so the records list stays
        consistent.

        Yields the live attribute dict so callers can populate values that
        are only known after ``backend.execute`` returns (e.g. ``t_db_ms``
        on the SERIALIZE phase under the subprocess adapter). Any
        keys added to the yielded dict are merged into the phase's record
        at exit time.
        """
        live_attrs = dict(attrs)
        self.enter(phase, attrs=live_attrs)
        try:
            yield live_attrs
        finally:
            self.exit(phase, attrs=live_attrs)
