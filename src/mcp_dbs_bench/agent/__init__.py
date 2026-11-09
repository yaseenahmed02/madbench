"""Track B agent loop.

Models are reached through OpenRouter (``openrouter_llm.py``) at
`temperature=0.0`. The MCP layer uses the official `mcp` Python SDK.

Public surface:
  - :class:`AgentRunner` orchestrates one (query, paraphrase) attempt.
  - :class:`AgentOutcome` is the four-way classification.
  - :class:`Attempt` records one LLM->tool->result cycle for IM-7 / Gamma_T.
  - :class:`AgentRunResult` aggregates attempts + final answer + token usage.
"""

from __future__ import annotations

from mcp_dbs_bench.agent.loop import (
    AgentOutcome,
    AgentRunner,
    AgentRunResult,
    Attempt,
)

__all__ = [
    "AgentOutcome",
    "AgentRunResult",
    "AgentRunner",
    "Attempt",
]
