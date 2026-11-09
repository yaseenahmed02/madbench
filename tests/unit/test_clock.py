"""Tests for the monotonic clock helpers.

`time.monotonic_ns()` is the only legal clock because
`time.time_ns()` can jump backwards under NTP adjustment.
"""

from __future__ import annotations

import pytest

from mcp_dbs_bench.core.clock import NS_PER_MS, monotonic_now_ns, ns_to_ms


def test_monotonic_now_ns_returns_int():
    assert isinstance(monotonic_now_ns(), int)


def test_monotonic_now_ns_is_strictly_increasing():
    a = monotonic_now_ns()
    b = monotonic_now_ns()
    # Two successive calls may yield equal values on coarse clocks; require >=.
    assert b >= a


def test_monotonic_now_ns_is_positive():
    assert monotonic_now_ns() > 0


def test_ns_to_ms_converts_one_million_ns_to_one_ms():
    assert ns_to_ms(1_000_000) == 1.0


def test_ns_to_ms_handles_zero():
    assert ns_to_ms(0) == 0.0


def test_ns_to_ms_returns_float_for_sub_ms_durations():
    assert ns_to_ms(500_000) == 0.5


def test_ns_to_ms_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        ns_to_ms(-1)


def test_ns_per_ms_constant():
    assert NS_PER_MS == 1_000_000
