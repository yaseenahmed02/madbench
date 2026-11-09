"""OpenRouter cost tracking for Track B.

The BudgetTracker in ``track_a.budget_tracker`` uses a flat default rate.
Track B runs on OpenRouter, where per-generation cost is returned by the
API itself. We use OpenRouter's authoritative numbers to
drive the hard budget cap.

Strategy
--------
1. After each turn we ask ``GET /generation?id=<gen_id>`` for the exact
   cost of that generation. If OpenRouter has not yet finalised the
   generation (403 during the same session), we fall back to a local
   estimate based on token counts.
2. After each attempt we ask ``GET /key`` for the account's cumulative
   usage in USD. If this exceeds the local session baseline plus the
   configured cap, the sweep aborts.
3. All numbers are logged in a ledger so the run can be checked after the
   fact.

Reference: https://openrouter.ai/docs/api-reference/get-a-generation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

_LOG = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_HTTP_TIMEOUT_S: float = 30.0
_MAX_LOOKUP_ATTEMPTS: int = 3
_LOOKUP_BACKOFF_S: float = 0.75


class BudgetExceeded(RuntimeError):
    """Raised when cumulative Track B cost passes the configured cap."""


@dataclass(slots=True)
class CostSample:
    """One line in the cost ledger."""

    cell_id: str
    rep_idx: int
    generation_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    source: str  # "openrouter" | "estimate"


@dataclass(slots=True)
class OpenRouterCostTracker:
    """Enforce a hard USD cap using OpenRouter's authoritative usage counters.

    Args:
        api_key: OpenRouter bearer token.
        budget_usd: Hard cap. ``check_or_raise`` raises when exceeded.
        session_baseline_usd: Account cumulative cost recorded at start.
            All budget checks are relative to this baseline so pre-existing
            spend does not consume this run's budget.
    """

    api_key: str
    budget_usd: float
    session_baseline_usd: float = 0.0
    samples: list[CostSample] = field(default_factory=list)
    fallback_estimate_input_per_m: float = 0.10
    fallback_estimate_output_per_m: float = 0.40

    def initialise_baseline(self) -> None:
        """Read the account's cumulative usage so we bill only new spend."""
        self.session_baseline_usd = self._fetch_key_usage_usd()
        _LOG.info(
            "openrouter baseline usage=%.6f USD, budget=+%.4f USD",
            self.session_baseline_usd, self.budget_usd,
        )

    def record_generation(
        self,
        *,
        cell_id: str,
        rep_idx: int,
        generation_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostSample:
        """Fetch the authoritative cost of one generation; log a sample."""
        cost, source = self._resolve_generation_cost(
            generation_id=generation_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        sample = CostSample(
            cell_id=cell_id, rep_idx=rep_idx, generation_id=generation_id,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost, source=source,
        )
        self.samples.append(sample)
        return sample

    def check_or_raise(self) -> float:
        """Poll the account balance; raise if the delta exceeds the cap."""
        current = self._fetch_key_usage_usd()
        delta = current - self.session_baseline_usd
        if delta > self.budget_usd:
            raise BudgetExceeded(
                f"OpenRouter cumulative delta ${delta:.4f} exceeds "
                f"cap ${self.budget_usd:.4f} — aborting sweep."
            )
        return delta

    def cumulative_local_cost(self) -> float:
        """Sum of per-sample cost fetched during this session."""
        return sum(s.cost_usd for s in self.samples)

    def _resolve_generation_cost(
        self,
        *,
        generation_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> tuple[float, str]:
        """Ask OpenRouter for cost; fall back to estimate on failure."""
        if not generation_id:
            return self._estimate(input_tokens, output_tokens), "estimate"
        for attempt in range(_MAX_LOOKUP_ATTEMPTS):
            cost = self._lookup_generation_cost(generation_id)
            if cost is not None:
                return cost, "openrouter"
            time.sleep(_LOOKUP_BACKOFF_S * (attempt + 1))
        return self._estimate(input_tokens, output_tokens), "estimate"

    def _lookup_generation_cost(self, generation_id: str) -> float | None:
        """Best-effort call to ``/generation?id=<id>``. Return None on failure."""
        url = f"{_OPENROUTER_BASE}/generation"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT_S) as client:
                resp = client.get(url, params={"id": generation_id}, headers=headers)
            if resp.status_code != 200:
                return None
            data = resp.json().get("data") or {}
            cost = data.get("total_cost") or data.get("cost")
            if cost is None:
                return None
            return float(cost)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            _LOG.debug("generation cost lookup failed: %s", exc)
            return None

    def _fetch_key_usage_usd(self) -> float:
        """Return the account's cumulative-usage number in USD."""
        url = f"{_OPENROUTER_BASE}/key"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT_S) as client:
                resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data") or {}
            return float(data.get("usage") or 0.0)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            _LOG.warning(
                "key usage lookup failed (%s); "
                "falling back to local estimate for budget check",
                exc,
            )
            return self.cumulative_local_cost() + self.session_baseline_usd

    def _estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Fallback cost estimate for DeepSeek-class pricing."""
        return (
            (input_tokens / 1e6) * self.fallback_estimate_input_per_m
            + (output_tokens / 1e6) * self.fallback_estimate_output_per_m
        )


__all__ = ["BudgetExceeded", "CostSample", "OpenRouterCostTracker"]
