"""8-term timing decomposition and transport overlap correction.

Pure functions over PhaseRecords. No I/O, no clock calls (records already
carry monotonic timestamps). The metric layer consumes TimingDecomposition
to compute PTR, phi, gamma and the phase-dominance fractions f_i.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mcp_dbs_bench.core.clock import ns_to_ms
from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseRecord

PHASE_TERMS: tuple[str, ...] = (
    "t_db_ms",
    "t_disc_ms",
    "t_life_ms",
    "t_hand_ms",
    "t_ser_ms",
    "t_parse_ms",
    "t_xport_excl_ms",
    "epsilon_ms",
)


@dataclass(frozen=True, slots=True)
class Interval:
    """Half-open monotonic interval [start_ns, end_ns)."""

    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)


def interval_overlap_ns(a: Interval, b: Interval) -> int:
    """Return the overlap (intersection length) of two intervals in nanoseconds."""
    start = max(a.start_ns, b.start_ns)
    end = min(a.end_ns, b.end_ns)
    return max(0, end - start)


def exclusive_transport_time_ns(
    xport: Interval, handover: Interval, parse: Interval
) -> int:
    """Exclusive transport time: T_xport_excl = T_xport_wall - T_x_int_h - T_x_int_p.

    Under single-machine stdio, the correction is typically negligible (< 1%
    of xport.duration_ns); a larger gap is a calibration warning.
    """
    return (
        xport.duration_ns
        - interval_overlap_ns(xport, handover)
        - interval_overlap_ns(xport, parse)
    )


@dataclass(frozen=True, slots=True)
class TimingDecomposition:
    """The eight terms of the timing identity.

    All values are in milliseconds. ``epsilon_ms`` is the residual:
    T_total minus the sum of T_db and the six named phase durations.
    """

    t_total_ms: float
    t_db_ms: float
    t_disc_ms: float
    t_life_ms: float
    t_hand_ms: float
    t_ser_ms: float
    t_parse_ms: float
    t_xport_excl_ms: float
    epsilon_ms: float

    @property
    def residual_fraction(self) -> float:
        """|epsilon| / T_total. Used by Property 1 (Completeness)."""
        if self.t_total_ms <= 0:
            return float("inf")
        return abs(self.epsilon_ms) / self.t_total_ms

    def passes_completeness(self, threshold: float = 0.05) -> bool:
        """True iff |epsilon|/T_total < threshold."""
        return self.residual_fraction < threshold

    def to_dict(self) -> dict[str, float]:
        """Return a dict view with the canonical column keys (PHASE_TERMS plus t_total)."""
        return asdict(self)


def decompose(
    records: list[PhaseRecord], *, t_total_ms: float, t_db_ms: float
) -> TimingDecomposition:
    """Reduce a list of PhaseRecords into the 8-term timing decomposition.

    The subprocess adapter wraps its local ``backend.execute`` call inside
    the SERIALIZE phase so the wall time is accounted in
    the identity rather than absorbed as epsilon. The SERIALIZE phase
    carries a ``t_db_ms`` attribute equal to that local execution; the
    identity must not double-count it. We subtract ``t_db_ms`` from the
    SERIALIZE total so T_db appears as its own term exactly once.

    Args:
        records: All PhaseRecords for one mcp.interaction.
        t_total_ms: End-to-end wall-clock duration of the interaction.
        t_db_ms: Time inside the DBMS driver (from the backend adapter,
            not from a Phase). Query planning time is included.

    Returns:
        A TimingDecomposition with the six phase sums, T_db, T_total, and
        the residual epsilon.

    Raises:
        ValueError: If ``t_total_ms`` or ``t_db_ms`` is negative.
    """
    if t_total_ms < 0:
        raise ValueError(f"t_total_ms must be non-negative; got {t_total_ms}")
    if t_db_ms < 0:
        raise ValueError(f"t_db_ms must be non-negative; got {t_db_ms}")

    by_phase = _group_records_by_phase(records)
    sums = _sum_phase_durations(by_phase)
    t_db_in_ser_ms = _sum_attr(records, Phase.SERIALIZE, "t_db_ms")
    t_ser_ms = max(0.0, sums["ser"] - t_db_in_ser_ms)
    accounted = (
        t_db_ms + sums["disc"] + sums["life"] + sums["hand"]
        + t_ser_ms + sums["parse"] + sums["xport_excl"]
    )

    return TimingDecomposition(
        t_total_ms=t_total_ms,
        t_db_ms=t_db_ms,
        t_disc_ms=sums["disc"],
        t_life_ms=sums["life"],
        t_hand_ms=sums["hand"],
        t_ser_ms=t_ser_ms,
        t_parse_ms=sums["parse"],
        t_xport_excl_ms=sums["xport_excl"],
        epsilon_ms=t_total_ms - accounted,
    )


def _group_records_by_phase(
    records: list[PhaseRecord],
) -> dict[Phase, list[Interval]]:
    """Bucket PhaseRecords by Phase, preserving order."""
    by_phase: dict[Phase, list[Interval]] = {p: [] for p in Phase}
    for record in records:
        by_phase[record.phase].append(Interval(record.start_ns, record.end_ns))
    return by_phase


def _sum_phase_durations(
    by_phase: dict[Phase, list[Interval]],
) -> dict[str, float]:
    """Sum per-phase durations and compute exclusive transport."""
    return {
        "disc": _sum_durations_ms(by_phase[Phase.DISCOVERY]),
        "life": _sum_durations_ms(by_phase[Phase.LIFECYCLE]),
        "hand": _sum_durations_ms(by_phase[Phase.HANDOVER]),
        "ser":  _sum_durations_ms(by_phase[Phase.SERIALIZE]),
        "parse": _sum_durations_ms(by_phase[Phase.PARSE]),
        "xport_excl": _exclusive_transport_total_ms(
            xports=by_phase[Phase.TRANSPORT],
            handovers=by_phase[Phase.HANDOVER],
            parses=by_phase[Phase.PARSE],
        ),
    }


def _sum_attr(records: list[PhaseRecord], phase: Phase, key: str) -> float:
    """Sum a numeric attribute across all records of ``phase``.

    Used by the double-count corrections in :func:`decompose`. Records
    missing the key contribute zero. Non-numeric values are skipped to
    keep the decomposition robust against plugin metadata leakage.
    """
    total = 0.0
    for record in records:
        if record.phase is not phase:
            continue
        value = record.attrs.get(key)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def _sum_durations_ms(intervals: list[Interval]) -> float:
    return sum(ns_to_ms(iv.duration_ns) for iv in intervals)


def _exclusive_transport_total_ms(
    *,
    xports: list[Interval],
    handovers: list[Interval],
    parses: list[Interval],
) -> float:
    """Sum exclusive transport across all transport intervals.

    Each transport interval is corrected against all handover and parse
    intervals because a single transport may span multiple tool calls (e.g.
    streaming server with multiple in-flight requests).
    """
    if not xports:
        return 0.0
    total_ns = 0
    for xport in xports:
        wall = xport.duration_ns
        overlap_h = sum(interval_overlap_ns(xport, h) for h in handovers)
        overlap_p = sum(interval_overlap_ns(xport, p) for p in parses)
        total_ns += max(0, wall - overlap_h - overlap_p)
    return ns_to_ms(total_ns)
