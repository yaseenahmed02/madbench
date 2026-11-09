"""Tests for the Phase enum.

Phase is the closed set of named protocol-tax phases
The enum drives the PhaseContext state machine and the OTel span hierarchy.
"""

from __future__ import annotations

import pytest

from mcp_dbs_bench.core.phase import Phase


def test_phase_enum_has_exactly_six_members():
    # Six measurable protocol phases.
    assert len(list(Phase)) == 6


def test_phase_enum_has_expected_members():
    expected = {"DISCOVERY", "LIFECYCLE", "HANDOVER", "SERIALIZE", "PARSE", "TRANSPORT"}
    actual = {p.name for p in Phase}
    assert actual == expected


def test_phase_values_are_lowercase_strings():
    for phase in Phase:
        assert phase.value == phase.name.lower()


def test_phase_to_span_name_is_canonical():
    # Phases map to OTel span names.
    assert Phase.DISCOVERY.span_name == "mcp.discovery"
    assert Phase.LIFECYCLE.span_name == "mcp.lifecycle.connect"
    assert Phase.HANDOVER.span_name == "mcp.tool_call"
    assert Phase.SERIALIZE.span_name == "mcp.serialization"
    assert Phase.PARSE.span_name == "mcp.parse"
    assert Phase.TRANSPORT.span_name == "mcp.transport"


def test_phase_from_span_name_round_trips():
    for phase in Phase:
        assert Phase.from_span_name(phase.span_name) is phase


def test_phase_from_span_name_raises_on_unknown():
    with pytest.raises(KeyError):
        Phase.from_span_name("not.a.phase")
