"""Tests for Property 1/2/3 verification."""

from __future__ import annotations

import pytest

from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseRecord
from mcp_dbs_bench.core.timing import Interval
from mcp_dbs_bench.instrumentation.properties import (
    CompletenessResult,
    ExclusivityViolation,
    verify_completeness,
    verify_exclusivity,
    verify_observability,
)


def _record(phase: Phase, start_ms: float, end_ms: float) -> PhaseRecord:
    return PhaseRecord(
        phase=phase,
        start_ns=int(start_ms * 1_000_000),
        end_ns=int(end_ms * 1_000_000),
        attrs={},
    )


# ---------- Property 1: Completeness ----------


def test_completeness_passes_when_residual_under_threshold():
    result = verify_completeness(
        t_total_ms=100.0,
        t_db_ms=30.0,
        t_disc_ms=10.0,
        t_life_ms=5.0,
        t_hand_ms=5.0,
        t_ser_ms=20.0,
        t_parse_ms=20.0,
        t_xport_excl_ms=8.0,
        threshold=0.05,
    )
    # accounted = 98; epsilon = 2; fraction = 0.02
    assert isinstance(result, CompletenessResult)
    assert result.passes is True
    assert result.fraction == pytest.approx(0.02)


def test_completeness_fails_when_residual_above_threshold():
    result = verify_completeness(
        t_total_ms=100.0,
        t_db_ms=30.0,
        t_disc_ms=10.0,
        t_life_ms=5.0,
        t_hand_ms=5.0,
        t_ser_ms=10.0,
        t_parse_ms=10.0,
        t_xport_excl_ms=10.0,
        threshold=0.05,
    )
    # accounted = 80; epsilon = 20; fraction = 0.2
    assert result.passes is False


def test_completeness_handles_zero_total():
    result = verify_completeness(
        t_total_ms=0.0,
        t_db_ms=0.0,
        t_disc_ms=0.0,
        t_life_ms=0.0,
        t_hand_ms=0.0,
        t_ser_ms=0.0,
        t_parse_ms=0.0,
        t_xport_excl_ms=0.0,
    )
    assert result.fraction == float("inf")
    assert result.passes is False


# ---------- Property 2: Exclusivity ----------


def test_exclusivity_passes_for_disjoint_phases():
    records = [
        _record(Phase.DISCOVERY, 0.0, 10.0),
        _record(Phase.HANDOVER, 10.0, 20.0),
        _record(Phase.SERIALIZE, 20.0, 30.0),
    ]
    assert verify_exclusivity(records) == []


def test_exclusivity_flags_overlap_between_non_transport_phases():
    records = [
        _record(Phase.DISCOVERY, 0.0, 20.0),
        _record(Phase.HANDOVER, 10.0, 30.0),  # overlaps 10 ms with DISCOVERY
    ]
    violations = verify_exclusivity(records)
    assert len(violations) == 1
    v = violations[0]
    assert isinstance(v, ExclusivityViolation)
    # Either order is fine.
    pair = {v.phase_a, v.phase_b}
    assert pair == {"discovery", "handover"}
    assert v.overlap_ns == 10_000_000


def test_exclusivity_allows_transport_to_overlap_handover():
    # transport may overlap handover.
    records = [
        _record(Phase.HANDOVER, 0.0, 30.0),
        _record(Phase.TRANSPORT, 10.0, 25.0),
    ]
    assert verify_exclusivity(records) == []


def test_exclusivity_allows_transport_to_overlap_parse():
    records = [
        _record(Phase.PARSE, 0.0, 30.0),
        _record(Phase.TRANSPORT, 10.0, 25.0),
    ]
    assert verify_exclusivity(records) == []


# ---------- Property 3: Observability ----------


def test_observability_passes_for_valid_records():
    records = [
        _record(Phase.DISCOVERY, 1.0, 2.0),
        _record(Phase.HANDOVER, 5.0, 7.0),
    ]
    assert verify_observability(records) == []


def test_observability_handles_empty_record_list():
    assert verify_observability([]) == []


def test_observability_does_not_raise_for_known_phases():
    # Phase enum is closed; all phase names are observable by construction.
    # Start at 100 ms so every record has positive timestamps.
    records = [
        _record(p, 100.0 + float(i), 100.0 + float(i + 1))
        for i, p in enumerate(Phase)
    ]
    assert verify_observability(records) == []


# ---------- Property 3 (Observability): wall-clock rejection ----------


def test_observability_rejects_wall_clock_timestamps():
    """Property 3 requires monotonic timestamps.

    Wall-clock (``time.time_ns()``) returns CLOCK_REALTIME, which can jump
    backwards on NTP sync. The verifier must reject any record whose start
    falls outside the harness's recorded monotonic epoch.
    """
    import time

    wall_ns = time.time_ns()  # CLOCK_REALTIME — orders of magnitude larger
    wall_record = PhaseRecord(
        phase=Phase.DISCOVERY,
        start_ns=wall_ns,
        end_ns=wall_ns + 1_000_000,
        attrs={},
    )
    errors = verify_observability([wall_record])
    assert any("wall" in e.lower() or "monotonic" in e.lower() for e in errors), (
        f"verifier accepted wall-clock timestamps without flagging; errors={errors}"
    )


def test_observability_accepts_monotonic_timestamps():
    """Sanity check: monotonic-clock values pass the check."""
    import time

    start = time.monotonic_ns()
    record = PhaseRecord(
        phase=Phase.DISCOVERY,
        start_ns=start,
        end_ns=start + 1_000_000,
        attrs={},
    )
    assert verify_observability([record]) == []


def test_interval_helper_is_exported():
    # Smoke test: Interval should be importable from the timing module.
    Interval(0, 100)
