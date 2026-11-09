"""Tests for derived quantities: PTR, phi, gamma, rho, S_ser, delta_CW, Gamma_T."""

from __future__ import annotations

import pytest

from mcp_dbs_bench.core.derived import (
    context_decay,
    cumulative_decay,
    payload_expansion_ratio,
    protocol_tax_ms,
    protocol_tax_ratio,
    serialization_throughput_rows_per_ms,
    token_inflation_factor,
)


def test_protocol_tax_is_total_minus_db_time():
    # Teardown and residual count as tax, so PTax = T_total - T_db.
    result = protocol_tax_ms(t_total_ms=5_120.0, t_db_ms=100.0)
    assert result == 5_020.0


def test_protocol_tax_ratio_divides_ptax_by_tdb():
    # PTR = PTax / T_db.  T_db must be positive (engine time is observable).
    ptr = protocol_tax_ratio(ptax_ms=20.0, t_db_ms=10.0)
    assert ptr == 2.0


def test_protocol_tax_ratio_rejects_zero_or_negative_tdb():
    with pytest.raises(ValueError, match="t_db_ms"):
        protocol_tax_ratio(ptax_ms=20.0, t_db_ms=0.0)
    with pytest.raises(ValueError, match="t_db_ms"):
        protocol_tax_ratio(ptax_ms=20.0, t_db_ms=-1.0)


def test_token_inflation_factor_matches_definition():
    # phi = (tau_disc + tau_result + tau_boiler) / tau_star_result
    # The function takes discovery_tokens, tool_call_response_tokens, tau_star.
    phi = token_inflation_factor(
        discovery_tokens=100,
        tool_call_response_tokens=400,
        tau_star_result=50,
    )
    assert phi == (100 + 400) / 50


def test_token_inflation_rejects_zero_tau_star():
    with pytest.raises(ValueError, match="tau_star_result"):
        token_inflation_factor(
            discovery_tokens=100,
            tool_call_response_tokens=400,
            tau_star_result=0,
        )


def test_context_decay_fraction():
    # gamma = (tau_disc + tau_envelope_boiler) / W_ctx
    gamma = context_decay(
        discovery_tokens=4000,
        envelope_boiler_tokens=1000,
        context_window_size=200_000,
    )
    assert gamma == 0.025


def test_context_decay_rejects_nonpositive_window():
    with pytest.raises(ValueError, match="context_window_size"):
        context_decay(
            discovery_tokens=100, envelope_boiler_tokens=10, context_window_size=0
        )


def test_payload_expansion_ratio():
    rho = payload_expansion_ratio(b_mcp=400, b_native=100)
    assert rho == 4.0


def test_payload_expansion_rejects_nonpositive_native():
    with pytest.raises(ValueError, match="b_native"):
        payload_expansion_ratio(b_mcp=400, b_native=0)


def test_serialization_throughput_rows_per_ms():
    s = serialization_throughput_rows_per_ms(row_count=500, t_ser_ms=10.0)
    assert s == 50.0


def test_serialization_throughput_rejects_nonpositive_t_ser():
    with pytest.raises(ValueError, match="t_ser_ms"):
        serialization_throughput_rows_per_ms(row_count=500, t_ser_ms=0.0)


def test_cumulative_decay_sums_per_step_gammas():
    # Gamma_T = sum_t gamma_t over T agent steps.
    assert cumulative_decay([0.01, 0.02, 0.03]) == pytest.approx(0.06)


def test_cumulative_decay_empty_list_returns_zero():
    assert cumulative_decay([]) == 0.0


def test_cumulative_decay_rejects_negative_gamma():
    with pytest.raises(ValueError, match="gamma"):
        cumulative_decay([0.01, -0.02])
