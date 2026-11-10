"""Provider-neutral agent loop core.

The loop is parameterized on a :class:`LLMClient` and an :class:`MCPClient`
protocol so it can be exercised end-to-end against stubs in unit tests, and
against a real LLM provider and the MCP Python SDK in a run.

Constraints (locked by scope):
  - max 10 model iterations per query; each iteration may request
    several tool calls
  - more than 3 retries ends the attempt (a retry is the same tool
    issued again after an error)

The harness owns correctness checking — the agent loop only knows whether a
result was submitted. ``CORRECT_UNVERIFIED`` means "submit_answer was called";
the harness verifies the answer afterward and may downgrade to incorrect.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


SUBMIT_TOOL_NAME: str = "submit_answer"


class AgentOutcome(str, Enum):
    """Four-way outcome classification"""

    CORRECT_UNVERIFIED = "correct_unverified"
    INCORRECT_UNVERIFIED = "incorrect_unverified"
    TIMEOUT = "timeout"
    CALL_LIMIT_EXCEEDED = "call_limit_exceeded"


@dataclass(frozen=True, slots=True)
class Attempt:
    """One LLM -> tool -> result cycle, suitable for an agent.retry span."""

    index: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: Any
    is_error: bool
    is_retry: bool
    input_tokens: int
    output_tokens: int


@dataclass(slots=True)
class AgentRunResult:
    """Aggregated outcome for one (query, paraphrase) attempt."""

    outcome: AgentOutcome
    final_answer: Any | None
    attempts: list[Attempt] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass(slots=True)
class _LoopState:
    """Mutable per-run state threaded through the decomposed loop helpers.

    Holds the data that every step mutates: the attempts log, retry
    counter, and token usage totals. Encapsulating this in one object
    keeps :func:`AgentRunner.run` short and the helpers' signatures clean.
    """

    attempts: list[Attempt] = field(default_factory=list)
    total_input: int = 0
    total_output: int = 0
    retries_used: int = 0


def _build_run_result(
    outcome: AgentOutcome,
    final_answer: Any | None,
    state: _LoopState,
) -> AgentRunResult:
    """Build an :class:`AgentRunResult` from accumulator state."""
    return AgentRunResult(
        outcome=outcome,
        final_answer=final_answer,
        attempts=list(state.attempts),
        total_input_tokens=state.total_input,
        total_output_tokens=state.total_output,
    )


class LLMClient(Protocol):
    """Provider-neutral LLM client interface used by :class:`AgentRunner`."""

    async def next_turn(
        self,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a dict with keys ``tool_calls`` and usage stats.

        Tool calls have the shape ``{"id", "name", "input"}``.
        """
        ...


class MCPClient(Protocol):
    """Provider-neutral MCP client interface used by :class:`AgentRunner`."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke an MCP tool. Return its result, or a dict with ``error`` key."""
        ...


def _build_assistant_message(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Anthropic message format echo of the assistant's tool-use turn."""
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
            for tc in tool_calls
        ],
    }


def _build_user_message_with_results(
    tool_calls: list[dict[str, Any]], tool_results: list[tuple[str, Any, bool]]
) -> dict[str, Any]:
    """Build the user-turn message that delivers tool_result blocks."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": json.dumps(result) if not isinstance(result, str) else result,
                "is_error": is_error,
            }
            for tc, (_, result, is_error) in zip(tool_calls, tool_results, strict=True)
        ],
    }


def _is_retry(prev: Attempt | None, current_tool: str, current_input: dict[str, Any]) -> bool:
    """Heuristic: a retry is a re-issue of the same
    tool after a recent failure."""
    if prev is None or not prev.is_error:
        return False
    return prev.tool_name == current_tool


class AgentRunner:
    """Drive a Track-B (query, paraphrase) attempt to completion.

    Args:
        llm: Provider-neutral LLM client.
        mcp: Provider-neutral MCP client.
        max_tool_calls: Hard cap on tool calls per query (default 10).
        max_retries: Per-logical-query retry budget (default 3).
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        mcp: MCPClient,
        max_tool_calls: int = 10,
        max_retries: int = 3,
    ) -> None:
        if max_tool_calls <= 0:
            raise ValueError(f"max_tool_calls must be positive; got {max_tool_calls}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative; got {max_retries}")
        self._llm = llm
        self._mcp = mcp
        self._max_tool_calls = max_tool_calls
        self._max_retries = max_retries

    async def run(
        self, *, natural_language: str, system_prompt: str,
    ) -> AgentRunResult:
        """Run the loop until success, error, or limits exhausted.

        Decomposed into ``_invoke_llm``, ``_handle_submit``,
        ``_dispatch_tool_calls`` and ``_run_one_turn`` so each step stays
        short.
        """
        tools = await self._build_tool_manifest()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": natural_language},
        ]
        state = _LoopState()
        for _ in range(self._max_tool_calls):
            terminal = await self._run_one_turn(system_prompt, tools, messages, state)
            if terminal is not None:
                return terminal
        return _build_run_result(AgentOutcome.CALL_LIMIT_EXCEEDED, None, state)

    async def _run_one_turn(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        state: _LoopState,
    ) -> AgentRunResult | None:
        """Execute one LLM + tool dispatch cycle. Returns a terminal
        :class:`AgentRunResult` when the loop should stop, or ``None`` to
        continue to the next iteration.
        """
        turn, tool_calls = await self._invoke_llm(system_prompt, tools, messages, state)
        if not tool_calls:
            return _build_run_result(AgentOutcome.INCORRECT_UNVERIFIED, None, state)
        messages.append(_build_assistant_message(tool_calls))
        submit_result = self._handle_submit(tool_calls, turn, state)
        if submit_result is not None:
            return submit_result
        await self._dispatch_tool_calls(tool_calls, turn, messages, state)
        if state.retries_used > self._max_retries:
            return _build_run_result(AgentOutcome.CALL_LIMIT_EXCEEDED, None, state)
        return None

    async def _invoke_llm(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        state: _LoopState,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run one LLM turn; track usage; return (turn, tool_calls)."""
        turn = await self._llm.next_turn(
            system_prompt=system_prompt, tools=tools, messages=messages,
        )
        state.total_input += int(turn.get("input_tokens", 0))
        state.total_output += int(turn.get("output_tokens", 0))
        return turn, turn.get("tool_calls", [])

    def _handle_submit(
        self,
        tool_calls: list[dict[str, Any]],
        turn: dict[str, Any],
        state: _LoopState,
    ) -> AgentRunResult | None:
        """If any tool call is ``submit_answer``, build the CORRECT result.

        Returns None when no submit was issued so the caller continues.
        """
        for tc in tool_calls:
            if tc["name"] != SUBMIT_TOOL_NAME:
                continue
            submit_attempt = Attempt(
                index=len(state.attempts),
                tool_name=SUBMIT_TOOL_NAME,
                tool_input=tc["input"],
                tool_result=None,
                is_error=False,
                is_retry=False,
                input_tokens=int(turn.get("input_tokens", 0)),
                output_tokens=int(turn.get("output_tokens", 0)),
            )
            state.attempts.append(submit_attempt)
            return _build_run_result(
                AgentOutcome.CORRECT_UNVERIFIED,
                tc["input"].get("result"),
                state,
            )
        return None

    async def _dispatch_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        turn: dict[str, Any],
        messages: list[dict[str, Any]],
        state: _LoopState,
    ) -> None:
        """Execute each non-submit tool call and record its Attempt."""
        tool_results: list[tuple[str, Any, bool]] = []
        for tc in tool_calls:
            prev = state.attempts[-1] if state.attempts else None
            is_retry = _is_retry(prev, tc["name"], tc["input"])
            if is_retry:
                state.retries_used += 1
            raw_result, is_error = await self._call_one_tool(tc)
            tool_results.append((tc["name"], raw_result, is_error))
            state.attempts.append(
                Attempt(
                    index=len(state.attempts),
                    tool_name=tc["name"],
                    tool_input=tc["input"],
                    tool_result=raw_result,
                    is_error=is_error,
                    is_retry=is_retry,
                    input_tokens=int(turn.get("input_tokens", 0)),
                    output_tokens=int(turn.get("output_tokens", 0)),
                )
            )
        messages.append(_build_user_message_with_results(tool_calls, tool_results))

    async def _call_one_tool(self, tc: dict[str, Any]) -> tuple[Any, bool]:
        """Invoke one MCP tool, normalising exceptions to ``{error}`` dicts."""
        try:
            raw_result = await self._mcp.call_tool(tc["name"], tc["input"])
            is_error = isinstance(raw_result, dict) and "error" in raw_result
        except Exception as exc:
            logger.warning(
                "mcp.call_tool raised %s for tool %r", type(exc).__name__, tc["name"]
            )
            raw_result = {"error": str(exc)}
            is_error = True
        return raw_result, is_error

    async def _build_tool_manifest(self) -> list[dict[str, Any]]:
        """Return the tool definitions to expose to the LLM.

        ``submit_answer`` is always included so the agent has a terminal
        action. Other tools are discovered via the MCP client's ``list_tools``
        method when implemented by the concrete client.
        """
        manifest: list[dict[str, Any]] = []
        list_method = getattr(self._mcp, "list_tools", None)
        if list_method is not None:
            discovered = await list_method()
            for tool in discovered:
                manifest.append(tool)
        manifest.append(_SUBMIT_TOOL_DEFINITION)
        return manifest


_SUBMIT_TOOL_DEFINITION: dict[str, Any] = {
    "name": SUBMIT_TOOL_NAME,
    "description": (
        "Submit the final answer to the user's question. "
        "Must be called exactly once per query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "result": {
                "description": "The query result as a list of rows.",
            }
        },
        "required": ["result"],
    },
}
