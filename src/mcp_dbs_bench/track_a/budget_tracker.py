"""Hard LLM budget cap for benchmark runs.

Track A runs incur zero LLM cost; the tracker records 0 throughout. Track B
records its own per-model cost (``scripts/track_b/_cost.py``); this tracker
uses a flat default rate and halts once the cap (default $10) is exceeded.

Default pricing:
    - Input:  $1.00 per 1M tokens
    - Output: $5.00 per 1M tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INPUT_USD_PER_M: float = 1.00
OUTPUT_USD_PER_M: float = 5.00
DEFAULT_CAP_USD: float = 10.0


class BudgetExceeded(RuntimeError):
    """Raised when a run's recorded LLM cost surpasses its cap."""


@dataclass
class BudgetTracker:
    """Cumulative-cost ledger with a hard halt when the cap is exceeded.

    Attributes:
        run_name: Run identifier, e.g. the server name.
        track: ``"A"`` (no LLM) or ``"B"`` (LLM agent loop).
        cap_usd: Hard upper bound on cumulative cost. ``record`` raises
            once cumulative cost exceeds this value.
        input_tokens_used: Running sum of input tokens recorded.
        output_tokens_used: Running sum of output tokens recorded.
    """

    run_name: str
    track: str = "A"
    cap_usd: float = DEFAULT_CAP_USD
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    _halted: bool = field(default=False, init=False, repr=False)

    def cost_usd(self) -> float:
        """Return cumulative cost in USD."""
        return (
            (self.input_tokens_used / 1e6) * INPUT_USD_PER_M
            + (self.output_tokens_used / 1e6) * OUTPUT_USD_PER_M
        )

    def can_afford(self, *, projected_input: int, projected_output: int) -> bool:
        """Return True if recording the projection would stay at or below cap."""
        projected_total = (
            self.cost_usd()
            + (projected_input / 1e6) * INPUT_USD_PER_M
            + (projected_output / 1e6) * OUTPUT_USD_PER_M
        )
        return projected_total <= self.cap_usd

    def record(self, *, input_tokens: int, output_tokens: int) -> None:
        """Add tokens to the ledger. Raise BudgetExceeded if the cap is breached."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError(
                f"token counts must be non-negative; "
                f"got input={input_tokens}, output={output_tokens}"
            )
        self.input_tokens_used += input_tokens
        self.output_tokens_used += output_tokens
        if self.cost_usd() > self.cap_usd:
            self._halted = True
            raise BudgetExceeded(
                f"Run {self.run_name}: spent ${self.cost_usd():.4f}, "
                f"exceeds ${self.cap_usd:.2f} cap "
                f"(input={self.input_tokens_used}, output={self.output_tokens_used})"
            )

    @property
    def halted(self) -> bool:
        """True once the budget cap has been exceeded at least once."""
        return self._halted

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for ``budget.json``."""
        return {
            "run": self.run_name,
            "track": self.track,
            "input_tokens_used": self.input_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "llm_cost_usd": self.cost_usd(),
            "llm_cap_usd": self.cap_usd,
            "halted_by_budget": self._halted,
        }


__all__ = ["BudgetExceeded", "BudgetTracker", "DEFAULT_CAP_USD"]
