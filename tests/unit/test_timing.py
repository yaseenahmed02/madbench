"""Tests for the 8-term timing decomposition.

T_total = T_db + T_disc + T_life + T_hand + T_ser + T_parse + T_xport^excl + epsilon

Functions under test reduce a list of PhaseRecords + (T_total, T_db) into the
eight named terms, including the transport overlap correction.
"""

from __future__ import annotations

import pytest

from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseRecord
from mcp_dbs_bench.core.timing import (
    PHASE_TERMS,
    Interval,
    TimingDecomposition,
    decompose,
    exclusive_transport_time_ns,
    interval_overlap_ns,
)


def _record(phase: Phase, start_ms: float, end_ms: float) -> PhaseRecord:
    return PhaseRecord(
        phase=phase,
        start_ns=int(start_ms * 1_000_000),
        end_ns=int(end_ms * 1_000_000),
        attrs={},
    )


def test_phase_terms_lists_eight_terms():
    # T_db, T_disc, T_life, T_hand, T_ser, T_parse, T_xport_excl, epsilon
    assert PHASE_TERMS == (
        "t_db_ms",
        "t_disc_ms",
        "t_life_ms",
        "t_hand_ms",
        "t_ser_ms",
        "t_parse_ms",
        "t_xport_excl_ms",
        "epsilon_ms",
    )


def test_interval_overlap_zero_for_disjoint_intervals():
    a = Interval(0, 100)
    b = Interval(200, 300)
    assert interval_overlap_ns(a, b) == 0


def test_interval_overlap_for_overlapping_intervals():
    a = Interval(0, 100)
    b = Interval(50, 150)
    assert interval_overlap_ns(a, b) == 50


def test_interval_overlap_for_contained_interval():
    a = Interval(0, 100)
    b = Interval(25, 75)
    assert interval_overlap_ns(a, b) == 50


def test_exclusive_transport_subtracts_handover_and_parse_overlap():
    # Transport 0..100, handover 10..40 (30 overlap), parse 60..90 (30 overlap)
    xport = Interval(0, 100)
    hand = Interval(10, 40)
    parse = Interval(60, 90)
    assert exclusive_transport_time_ns(xport, hand, parse) == 100 - 30 - 30


def test_exclusive_transport_no_overlap_returns_wall_clock():
    xport = Interval(0, 100)
    hand = Interval(200, 300)
    parse = Interval(400, 500)
    assert exclusive_transport_time_ns(xport, hand, parse) == 100


def test_decompose_simple_sequential_phases():
    records = [
        _record(Phase.DISCOVERY, 0.0, 10.0),
        _record(Phase.LIFECYCLE, 10.0, 15.0),
        _record(Phase.HANDOVER, 15.0, 20.0),
        _record(Phase.SERIALIZE, 20.0, 25.0),
        _record(Phase.PARSE, 25.0, 30.0),
        _record(Phase.TRANSPORT, 30.0, 35.0),
    ]
    result = decompose(records, t_total_ms=40.0, t_db_ms=5.0)
    assert result.t_disc_ms == pytest.approx(10.0)
    assert result.t_life_ms == pytest.approx(5.0)
    assert result.t_hand_ms == pytest.approx(5.0)
    assert result.t_ser_ms == pytest.approx(5.0)
    assert result.t_parse_ms == pytest.approx(5.0)
    assert result.t_xport_excl_ms == pytest.approx(5.0)
    assert result.t_db_ms == pytest.approx(5.0)


def test_decompose_sums_multiple_lifecycle_spans():
    # IM-5: T_life sums across all lifecycle.connect spans.
    records = [
        _record(Phase.LIFECYCLE, 0.0, 2.0),
        _record(Phase.LIFECYCLE, 5.0, 8.0),
        _record(Phase.LIFECYCLE, 10.0, 11.5),
    ]
    result = decompose(records, t_total_ms=20.0, t_db_ms=0.0)
    assert result.t_life_ms == pytest.approx(2.0 + 3.0 + 1.5)


def test_decompose_corrects_transport_overlap():
    # Transport overlaps both handover and parse.
    records = [
        _record(Phase.HANDOVER, 0.0, 30.0),
        _record(Phase.TRANSPORT, 10.0, 50.0),
        _record(Phase.PARSE, 40.0, 60.0),
    ]
    # Overlap: T n H = 20ms, T n P = 10ms => exclusive transport = 40 - 30 = 10ms
    result = decompose(records, t_total_ms=70.0, t_db_ms=10.0)
    assert result.t_xport_excl_ms == pytest.approx(10.0)


def test_decompose_computes_epsilon():
    records = [
        _record(Phase.DISCOVERY, 0.0, 5.0),
        _record(Phase.LIFECYCLE, 5.0, 7.0),
        _record(Phase.HANDOVER, 7.0, 9.0),
        _record(Phase.SERIALIZE, 9.0, 12.0),
        _record(Phase.PARSE, 12.0, 14.0),
        _record(Phase.TRANSPORT, 14.0, 16.0),
    ]
    # Accounted: 5+2+2+3+2+2 + 4 (db) = 20.  T_total = 25 => epsilon = 5
    result = decompose(records, t_total_ms=25.0, t_db_ms=4.0)
    assert result.epsilon_ms == pytest.approx(5.0)
    assert result.t_total_ms == pytest.approx(25.0)


def test_decompose_zero_epsilon_when_accounted_equals_total():
    records = [
        _record(Phase.DISCOVERY, 0.0, 10.0),
    ]
    result = decompose(records, t_total_ms=12.0, t_db_ms=2.0)
    assert result.epsilon_ms == pytest.approx(0.0)


def test_decompose_handles_no_records_for_phase():
    # If TRANSPORT is absent, t_xport_excl_ms == 0.
    records = [_record(Phase.DISCOVERY, 0.0, 5.0)]
    result = decompose(records, t_total_ms=10.0, t_db_ms=3.0)
    assert result.t_xport_excl_ms == 0.0
    assert result.t_disc_ms == pytest.approx(5.0)


def test_decompose_residual_fraction_property():
    records = [_record(Phase.DISCOVERY, 0.0, 5.0)]
    result = decompose(records, t_total_ms=10.0, t_db_ms=4.0)
    # accounted=9, total=10 => epsilon=1, fraction=0.1
    assert result.residual_fraction == pytest.approx(0.1)


def test_decompose_passes_completeness_below_threshold():
    records = [_record(Phase.DISCOVERY, 0.0, 9.6)]
    result = decompose(records, t_total_ms=10.0, t_db_ms=0.0)
    # epsilon = 0.4, fraction = 0.04
    assert result.passes_completeness(threshold=0.05) is True


def test_decompose_fails_completeness_above_threshold():
    records = [_record(Phase.DISCOVERY, 0.0, 5.0)]
    result = decompose(records, t_total_ms=10.0, t_db_ms=0.0)
    # epsilon = 5, fraction = 0.5
    assert result.passes_completeness(threshold=0.05) is False


def test_decompose_rejects_negative_total():
    with pytest.raises(ValueError, match="t_total_ms"):
        decompose([], t_total_ms=-1.0, t_db_ms=0.0)


def test_decompose_rejects_negative_db():
    with pytest.raises(ValueError, match="t_db_ms"):
        decompose([], t_total_ms=10.0, t_db_ms=-1.0)


def test_timing_decomposition_to_dict_has_canonical_keys():
    result = TimingDecomposition(
        t_total_ms=10.0,
        t_db_ms=3.0,
        t_disc_ms=2.0,
        t_life_ms=1.0,
        t_hand_ms=1.0,
        t_ser_ms=1.0,
        t_parse_ms=1.0,
        t_xport_excl_ms=0.5,
        epsilon_ms=0.5,
    )
    d = result.to_dict()
    for key in PHASE_TERMS:
        assert key in d
    assert d["t_total_ms"] == 10.0
