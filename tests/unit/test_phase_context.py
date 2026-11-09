"""Tests for the PhaseContext state machine.

PhaseContext enforces Property 2 (exclusivity) at runtime: at most one phase
may be open at any instant, and every enter() must have a matching exit().
"""

from __future__ import annotations

import pytest

from mcp_dbs_bench.core.exceptions import (
    ExclusivityViolationError,
    PhaseBoundaryError,
)
from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseContextImpl, PhaseRecord


def test_enter_exit_records_single_interval():
    ctx = PhaseContextImpl()
    ctx.enter(Phase.DISCOVERY)
    ctx.exit(Phase.DISCOVERY)
    records = ctx.records
    assert len(records) == 1
    assert records[0].phase is Phase.DISCOVERY
    assert records[0].end_ns >= records[0].start_ns


def test_span_context_manager_records_interval():
    ctx = PhaseContextImpl()
    with ctx.span(Phase.HANDOVER, request_bytes=100):
        pass
    assert len(ctx.records) == 1
    assert ctx.records[0].phase is Phase.HANDOVER
    assert ctx.records[0].attrs == {"request_bytes": 100}


def test_entering_phase_while_another_open_raises_exclusivity():
    ctx = PhaseContextImpl()
    ctx.enter(Phase.DISCOVERY)
    with pytest.raises(ExclusivityViolationError):
        ctx.enter(Phase.HANDOVER)


def test_exiting_phase_not_entered_raises_phase_boundary_error():
    ctx = PhaseContextImpl()
    with pytest.raises(PhaseBoundaryError):
        ctx.exit(Phase.PARSE)


def test_exiting_wrong_phase_raises_phase_boundary_error():
    ctx = PhaseContextImpl()
    ctx.enter(Phase.DISCOVERY)
    with pytest.raises(PhaseBoundaryError, match="DISCOVERY"):
        ctx.exit(Phase.HANDOVER)


def test_two_sequential_phases_both_recorded():
    ctx = PhaseContextImpl()
    with ctx.span(Phase.DISCOVERY):
        pass
    with ctx.span(Phase.HANDOVER):
        pass
    assert len(ctx.records) == 2
    assert [r.phase for r in ctx.records] == [Phase.DISCOVERY, Phase.HANDOVER]
    # Records are non-overlapping by construction.
    assert ctx.records[1].start_ns >= ctx.records[0].end_ns


def test_transport_phase_may_overlap_handover():
    # mcp.transport is a sibling, can overlap mcp.tool_call.
    # PhaseContext models this by allowing TRANSPORT to be entered while
    # another phase is open.
    ctx = PhaseContextImpl()
    ctx.enter(Phase.HANDOVER)
    ctx.enter(Phase.TRANSPORT)  # must not raise
    ctx.exit(Phase.TRANSPORT)
    ctx.exit(Phase.HANDOVER)
    phases = [r.phase for r in ctx.records]
    assert Phase.TRANSPORT in phases
    assert Phase.HANDOVER in phases


def test_phase_attrs_passed_to_record():
    ctx = PhaseContextImpl()
    ctx.enter(Phase.DISCOVERY, attrs={"tool_count": 5})
    ctx.exit(Phase.DISCOVERY, attrs={"response_bytes": 200})
    record = ctx.records[0]
    assert record.attrs["tool_count"] == 5
    assert record.attrs["response_bytes"] == 200


def test_phase_record_duration_ns_nonnegative():
    record = PhaseRecord(
        phase=Phase.DISCOVERY,
        start_ns=1_000_000,
        end_ns=3_000_000,
        attrs={},
    )
    assert record.duration_ns == 2_000_000


def test_phase_record_rejects_end_before_start():
    with pytest.raises(ValueError, match="end_ns"):
        PhaseRecord(
            phase=Phase.DISCOVERY,
            start_ns=1_000_000,
            end_ns=500_000,
            attrs={},
        )
