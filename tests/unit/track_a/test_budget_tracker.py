"""Tests for the LLM budget tracker.

Default pricing: $1.00 / 1M input, $5.00 / 1M output.
Each phase has a hard cap of $10 unless overridden.
"""

from __future__ import annotations

import pytest

from mcp_dbs_bench.track_a.budget_tracker import BudgetExceeded, BudgetTracker


def test_tracker_starts_at_zero_cost() -> None:
    tracker = BudgetTracker(run_name="mcp-alchemy")

    assert tracker.cost_usd() == pytest.approx(0.0)


def test_record_accumulates_input_cost_at_one_dollar_per_million() -> None:
    tracker = BudgetTracker(run_name="mcp-alchemy")

    tracker.record(input_tokens=1_000_000, output_tokens=0)

    assert tracker.cost_usd() == pytest.approx(1.00)


def test_record_accumulates_output_cost_at_five_dollars_per_million() -> None:
    tracker = BudgetTracker(run_name="mcp-alchemy")

    tracker.record(input_tokens=0, output_tokens=1_000_000)

    assert tracker.cost_usd() == pytest.approx(5.00)


def test_record_accumulates_combined_cost() -> None:
    tracker = BudgetTracker(run_name="mcp-alchemy")

    tracker.record(input_tokens=2_000_000, output_tokens=500_000)

    assert tracker.cost_usd() == pytest.approx(2.0 + 2.5)


def test_can_afford_true_below_cap() -> None:
    tracker = BudgetTracker(run_name="mcp-alchemy", cap_usd=10.0)

    assert tracker.can_afford(projected_input=1_000_000, projected_output=100_000) is True


def test_can_afford_false_above_cap() -> None:
    tracker = BudgetTracker(run_name="mcp-alchemy", cap_usd=10.0)

    tracker.record(input_tokens=5_000_000, output_tokens=900_000)

    assert tracker.can_afford(projected_input=1_000_000, projected_output=100_000) is False


def test_record_above_cap_raises_budget_exceeded() -> None:
    tracker = BudgetTracker(run_name="mcp-alchemy", cap_usd=10.0)

    with pytest.raises(BudgetExceeded) as exc_info:
        tracker.record(input_tokens=20_000_000, output_tokens=0)

    assert "mcp-alchemy" in str(exc_info.value)


def test_simulated_thousand_traces_at_ten_thousand_input_halts_when_cap_breached() -> None:
    """Scenario: 1000 traces x (10k input + 500 output) = $12.50, halts mid-stream."""
    tracker = BudgetTracker(run_name="mcp-alchemy", cap_usd=10.0)

    records_completed = 0
    with pytest.raises(BudgetExceeded):
        for _ in range(1000):
            tracker.record(input_tokens=10_000, output_tokens=500)
            records_completed += 1

    assert records_completed < 1000
    assert tracker.halted is True
    assert tracker.cost_usd() > 10.0


def test_simulated_thousand_traces_at_twelve_dollars_halts_with_budget_exceeded() -> None:
    """If projected total exceeds cap, halt must fire."""
    tracker = BudgetTracker(run_name="mcp-alchemy", cap_usd=10.0)

    with pytest.raises(BudgetExceeded):
        for _ in range(2000):
            tracker.record(input_tokens=10_000, output_tokens=500)


def test_to_dict_serializes_state_for_budget_json() -> None:
    tracker = BudgetTracker(run_name="mcp-alchemy", cap_usd=10.0)
    tracker.record(input_tokens=1_000_000, output_tokens=200_000)

    state = tracker.to_dict()

    assert state == {
        "run": "mcp-alchemy",
        "track": "A",
        "input_tokens_used": 1_000_000,
        "output_tokens_used": 200_000,
        "llm_cost_usd": pytest.approx(2.0),
        "llm_cap_usd": 10.0,
        "halted_by_budget": False,
    }


def test_to_dict_track_b_default() -> None:
    tracker = BudgetTracker(run_name="mcp-alchemy", track="B", cap_usd=10.0)

    assert tracker.to_dict()["track"] == "B"
