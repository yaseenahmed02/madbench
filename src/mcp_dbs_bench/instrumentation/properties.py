"""Property 1/2/3 verification.

These are runnable functions, not assertions, because the harness must
classify failing runs (flag + exclude from primary analysis) without
aborting the entire batch. The functions return result dataclasses that the
harness inspects and writes into the run manifest.

Property 1 (Completeness):   |epsilon| / T_total < threshold
Property 2 (Exclusivity):    non-TRANSPORT phases pairwise disjoint
                             (including TRANSPORT-TRANSPORT pairs)
Property 3 (Observability):  every span has positive monotone timestamps
                             drawn from ``time.monotonic_ns`` (not wall-clock)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations

from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseRecord
from mcp_dbs_bench.core.timing import Interval, interval_overlap_ns

# Process-start monotonic anchor. Every span recorded
# by the harness's PhaseContext uses ``core.clock.monotonic_now_ns``; that
# value is always >= this anchor and well below ``2 * monotonic_now_ns``.
# Wall-clock ``time.time_ns()`` returns nanoseconds since the Unix epoch
# (~1.7e18 in 2026), which is orders of magnitude above any monotonic
# value on any reasonable host. The verifier compares against this anchor
# to reject wall-clock-derived timestamps.
_PROCESS_START_MONOTONIC_NS: int = time.monotonic_ns()

# A wall-clock timestamp in nanoseconds will be a Unix-epoch ns value
# (~1.78e18 in 2026, growing by ~3.15e16/year). Monotonic counters on
# macOS/Linux measure since boot or a system reference, never exceed
# system uptime in ns (typically ~1e15 even on long-running servers), and
# are always far below the wall-clock value. We use a 30-year cutoff
# (9.46e17 ns) — well above any plausible monotonic value (would require
# 30 years of uninterrupted uptime) and well below any current-epoch
# wall-clock value (1.7e18 today, only growing). The verifier rejects
# anything above this cutoff as not-from-monotonic-clock.
_MAX_PLAUSIBLE_MONOTONIC_NS: int = 30 * 365 * 24 * 3600 * 1_000_000_000  # 30 years


@dataclass(frozen=True, slots=True)
class CompletenessResult:
    """Outcome of Property 1 verification."""

    epsilon_ms: float
    total_ms: float
    fraction: float
    threshold: float
    passes: bool


@dataclass(frozen=True, slots=True)
class ExclusivityViolation:
    """One overlapping phase pair detected by Property 2."""

    phase_a: str
    phase_b: str
    overlap_ns: int


def verify_completeness(
    *,
    t_total_ms: float,
    t_db_ms: float,
    t_disc_ms: float,
    t_life_ms: float,
    t_hand_ms: float,
    t_ser_ms: float,
    t_parse_ms: float,
    t_xport_excl_ms: float,
    threshold: float = 0.05,
    absolute_floor_ms: float = 1.0,
) -> CompletenessResult:
    """Property 1: completeness budget on ε = T_total - sum-of-phases.

    Instrumentation noise is bounded in *absolute* terms (~0.3-0.5 ms per
    rep), so on very small T_total (e.g., DuckDB Q1 SF1 at ~5 ms), a
    relative-only threshold flags cells that are otherwise clean. The rule is:

        |ε|/T_total < threshold  OR  |ε| < absolute_floor_ms

    A cell PASSES if EITHER bound holds. Default floor = 1.0 ms (~2x the
    observed worst-case instrumentation overhead). Set ``absolute_floor_ms=0``
    for a relative-only check.

    Returns:
        CompletenessResult with epsilon, fraction, and a pass/fail flag.
        Caller decides whether to exclude the run; this function never raises.
    """
    accounted = (
        t_db_ms + t_disc_ms + t_life_ms + t_hand_ms + t_ser_ms + t_parse_ms + t_xport_excl_ms
    )
    epsilon = t_total_ms - accounted
    if t_total_ms <= 0:
        # Degenerate measurement — fail regardless of absolute_floor_ms.
        return CompletenessResult(
            epsilon_ms=epsilon,
            total_ms=t_total_ms,
            fraction=float("inf"),
            threshold=threshold,
            passes=False,
        )
    fraction = abs(epsilon) / t_total_ms
    passes = fraction < threshold or abs(epsilon) < absolute_floor_ms
    return CompletenessResult(
        epsilon_ms=epsilon,
        total_ms=t_total_ms,
        fraction=fraction,
        threshold=threshold,
        passes=passes,
    )


def verify_exclusivity(records: list[PhaseRecord]) -> list[ExclusivityViolation]:
    """Property 2: phases must be pairwise disjoint, with one exception.

    TRANSPORT is permitted to overlap HANDOVER and PARSE because
    a stdio frame is in flight while the harness simultaneously marshals the
    outbound request or parses the inbound envelope. TRANSPORT spans must
    still be disjoint from each other (one stdio roundtrip at a time per
    connection) and from LIFECYCLE / DISCOVERY / SERIALIZE.

    Every pair is checked except the two documented exceptions.

    Returns:
        List of overlapping pairs; empty list means the property holds.
    """
    indexed = [(r.phase, Interval(r.start_ns, r.end_ns)) for r in records]
    violations: list[ExclusivityViolation] = []
    for (phase_a, iv_a), (phase_b, iv_b) in combinations(indexed, 2):
        if _is_permitted_overlap(phase_a, phase_b):
            continue
        overlap = interval_overlap_ns(iv_a, iv_b)
        if overlap > 0:
            violations.append(
                ExclusivityViolation(
                    phase_a=phase_a.value,
                    phase_b=phase_b.value,
                    overlap_ns=overlap,
                )
            )
    return violations


_PERMITTED_TRANSPORT_OVERLAPS: frozenset[Phase] = frozenset(
    {Phase.HANDOVER, Phase.PARSE}
)


def _is_permitted_overlap(phase_a: Phase, phase_b: Phase) -> bool:
    """Return True if (phase_a, phase_b) is a permitted overlap.

    TRANSPORT may overlap HANDOVER and PARSE. All other pairs
    (including TRANSPORT-TRANSPORT, TRANSPORT-DISCOVERY, TRANSPORT-LIFECYCLE,
    TRANSPORT-SERIALIZE) must be strictly disjoint.
    """
    if phase_a is Phase.TRANSPORT and phase_b in _PERMITTED_TRANSPORT_OVERLAPS:
        return True
    if phase_b is Phase.TRANSPORT and phase_a in _PERMITTED_TRANSPORT_OVERLAPS:
        return True
    return False


def verify_observability(records: list[PhaseRecord]) -> list[str]:
    """Property 3: every span has a real hardware (monotonic) timestamp.

    All phase timestamps must originate from
    ``time.monotonic_ns`` (or equivalent monotonic source). The verifier
    enforces:

      - ``start_ns > 0``
      - ``end_ns >= start_ns``
      - ``start_ns`` and ``end_ns`` are within the monotonic-clock range
        bounded by ``_MAX_PLAUSIBLE_MONOTONIC_NS``. Wall-clock timestamps
        (``time.time_ns``) exceed this bound by definition and are rejected.

    Args:
        records: PhaseRecords produced by a PhaseContext.

    Returns:
        List of error messages (empty list = property holds).
    """
    errors: list[str] = []
    for record in records:
        if record.start_ns <= 0:
            errors.append(f"{record.phase.name}: missing or zero start_ns")
            continue
        if record.end_ns < record.start_ns:
            errors.append(
                f"{record.phase.name}: end_ns ({record.end_ns}) "
                f"< start_ns ({record.start_ns})"
            )
            continue
        if record.start_ns > _MAX_PLAUSIBLE_MONOTONIC_NS:
            errors.append(
                f"{record.phase.name}: start_ns ({record.start_ns}) exceeds "
                f"plausible monotonic range (cutoff "
                f"{_MAX_PLAUSIBLE_MONOTONIC_NS}); likely a wall-clock "
                "value from time.time_ns rather than time.monotonic_ns. "
                "Property 3 requires monotonic timestamps."
            )
            continue
        if record.end_ns > _MAX_PLAUSIBLE_MONOTONIC_NS:
            errors.append(
                f"{record.phase.name}: end_ns ({record.end_ns}) exceeds "
                f"plausible monotonic range; likely a wall-clock value."
            )
    return errors
