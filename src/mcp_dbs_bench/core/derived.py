"""Derived quantities computed from the 8-term decomposition:

    - PTax    : T_total - T_db (all phases, teardown and residual)
    - PTR     : PTax / T_db
    - rho     : B_mcp / B_native
    - phi     : (tau_disc + tau_result + tau_boiler) / tau*_result
    - S_ser   : N_rows / T_ser
    - gamma   : (tau_disc + tau_boiler) / W_ctx
    - Gamma_T : sum_t gamma_t

All inputs are pre-validated; functions are pure and side-effect free.
"""

from __future__ import annotations

from collections.abc import Sequence


def protocol_tax_ms(*, t_total_ms: float, t_db_ms: float) -> float:
    """PTax = T_total - T_db.

    Everything outside database execution counts as Protocol Tax: every
    phase, teardown, and the residual the spans do not cover.
    """
    return t_total_ms - t_db_ms


def protocol_tax_ratio(*, ptax_ms: float, t_db_ms: float) -> float:
    """PTR = PTax / T_db.

    Raises:
        ValueError: If t_db_ms <= 0. The ratio is undefined for zero engine
            time and would silently divide by epsilon if not checked.
    """
    if t_db_ms <= 0:
        raise ValueError(f"t_db_ms must be positive; got {t_db_ms}")
    return ptax_ms / t_db_ms


def token_inflation_factor(
    *,
    discovery_tokens: int,
    tool_call_response_tokens: int,
    tau_star_result: int,
) -> float:
    """phi = (tau_disc + tau_result + tau_boiler) / tau*_result.

    `tool_call_response_tokens` already includes tau_result + tau_boiler
    because both come from the same JSON-RPC envelope token count.
    `tau_star_result` is the token count of the minimal semantic content.
    """
    if tau_star_result <= 0:
        raise ValueError(
            f"tau_star_result must be positive; got {tau_star_result}. "
            "Empty result sets cannot define phi."
        )
    return (discovery_tokens + tool_call_response_tokens) / tau_star_result


def context_decay(
    *,
    discovery_tokens: int,
    envelope_boiler_tokens: int,
    context_window_size: int,
) -> float:
    """gamma = (tau_disc + tau_boiler) / W_ctx.

    Fraction of the LLM's context window consumed by non-semantic protocol
    content. Track B reports the cumulative form Gamma_T over agent steps.
    """
    if context_window_size <= 0:
        raise ValueError(f"context_window_size must be positive; got {context_window_size}")
    return (discovery_tokens + envelope_boiler_tokens) / context_window_size


def cumulative_decay(per_step_gamma: Sequence[float]) -> float:
    """Gamma_T = sum_t gamma_t."""
    for value in per_step_gamma:
        if value < 0:
            raise ValueError(f"per-step gamma must be non-negative; got {value}")
    return float(sum(per_step_gamma))


def payload_expansion_ratio(*, b_mcp: int, b_native: int) -> float:
    """rho = B_mcp / B_native (CM-2).

    B_native is recorded by the paired native-CLI baseline run for the same
    (backend, query, SF, rep) cell.
    """
    if b_native <= 0:
        raise ValueError(f"b_native must be positive; got {b_native}")
    return b_mcp / b_native


def serialization_throughput_rows_per_ms(*, row_count: int, t_ser_ms: float) -> float:
    """S_ser = N_rows / T_ser (CM-1)."""
    if t_ser_ms <= 0:
        raise ValueError(f"t_ser_ms must be positive; got {t_ser_ms}")
    return row_count / t_ser_ms
