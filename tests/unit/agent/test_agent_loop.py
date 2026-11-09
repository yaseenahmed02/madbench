"""Tests for the Track B agent loop with stub MCP + LLM clients."""

from __future__ import annotations

import pytest

from mcp_dbs_bench.agent.loop import (
    AgentOutcome,
    AgentRunner,
)
from tests.unit.agent.agent_stubs import StubLLMClient, StubMCPClient


class TestAgentRunner:
    @pytest.mark.asyncio
    async def test_agent_succeeds_when_llm_submits_answer(self) -> None:
        # Stub LLM responds with submit_answer on first turn.
        llm = StubLLMClient(
            responses=[
                {
                    "tool_calls": [
                        {"id": "call_1", "name": "submit_answer",
                         "input": {"result": [[1, "Alice"]]}}
                    ],
                    "input_tokens": 100, "output_tokens": 20,
                }
            ]
        )
        mcp = StubMCPClient(
            tools=[{"name": "execute_sql", "description": "Run SQL"}],
            tool_responses={},
        )

        runner = AgentRunner(llm=llm, mcp=mcp, max_tool_calls=10, max_retries=3)
        result = await runner.run(
            natural_language="Get all users",
            system_prompt="You are an SQL agent.",
        )

        assert result.outcome == AgentOutcome.CORRECT_UNVERIFIED
        assert result.final_answer == [[1, "Alice"]]
        assert len(result.attempts) == 1

    @pytest.mark.asyncio
    async def test_agent_executes_tool_then_submits(self) -> None:
        # Two-turn conversation: LLM calls execute_sql, gets result, submits.
        llm = StubLLMClient(
            responses=[
                {
                    "tool_calls": [
                        {"id": "call_1", "name": "execute_sql",
                         "input": {"sql": "SELECT * FROM users"}}
                    ],
                    "input_tokens": 100, "output_tokens": 15,
                },
                {
                    "tool_calls": [
                        {"id": "call_2", "name": "submit_answer",
                         "input": {"result": [[1, "Alice"]]}}
                    ],
                    "input_tokens": 150, "output_tokens": 12,
                },
            ]
        )
        mcp = StubMCPClient(
            tools=[{"name": "execute_sql", "description": "Run SQL"}],
            tool_responses={"execute_sql": [[1, "Alice"]]},
        )

        runner = AgentRunner(llm=llm, mcp=mcp, max_tool_calls=10, max_retries=3)
        result = await runner.run(
            natural_language="Get all users",
            system_prompt="You are an SQL agent.",
        )

        assert result.outcome == AgentOutcome.CORRECT_UNVERIFIED
        assert len(result.attempts) == 2

    @pytest.mark.asyncio
    async def test_agent_hits_call_limit_returns_correct_outcome(self) -> None:
        # LLM never submits; should hit max_tool_calls=3.
        llm = StubLLMClient(
            responses=[
                {
                    "tool_calls": [
                        {"id": f"call_{i}", "name": "execute_sql",
                         "input": {"sql": "SELECT 1"}}
                    ],
                    "input_tokens": 100, "output_tokens": 10,
                }
                for i in range(20)
            ]
        )
        mcp = StubMCPClient(
            tools=[{"name": "execute_sql", "description": "Run SQL"}],
            tool_responses={"execute_sql": [[1]]},
        )

        runner = AgentRunner(llm=llm, mcp=mcp, max_tool_calls=3, max_retries=3)
        result = await runner.run(
            natural_language="x", system_prompt="y"
        )

        assert result.outcome == AgentOutcome.CALL_LIMIT_EXCEEDED
        assert len(result.attempts) == 3

    @pytest.mark.asyncio
    async def test_agent_records_retry_on_tool_failure(self) -> None:
        # First tool call errors; LLM retries; succeeds; submits.
        llm = StubLLMClient(
            responses=[
                {
                    "tool_calls": [
                        {"id": "c1", "name": "execute_sql",
                         "input": {"sql": "SELECT * FROM USERZ"}}  # typo
                    ],
                    "input_tokens": 100, "output_tokens": 12,
                },
                {
                    "tool_calls": [
                        {"id": "c2", "name": "execute_sql",
                         "input": {"sql": "SELECT * FROM users"}}
                    ],
                    "input_tokens": 130, "output_tokens": 12,
                },
                {
                    "tool_calls": [
                        {"id": "c3", "name": "submit_answer",
                         "input": {"result": [[1, "Alice"]]}}
                    ],
                    "input_tokens": 160, "output_tokens": 10,
                },
            ]
        )
        mcp = StubMCPClient(
            tools=[{"name": "execute_sql", "description": "Run SQL"}],
            tool_responses={
                ('execute_sql', '{"sql": "SELECT * FROM USERZ"}'):
                    {"error": "no such table USERZ"},
                ('execute_sql', '{"sql": "SELECT * FROM users"}'):
                    [[1, "Alice"]],
            },
        )

        runner = AgentRunner(llm=llm, mcp=mcp, max_tool_calls=10, max_retries=3)
        result = await runner.run(
            natural_language="Get users", system_prompt="x"
        )

        # Total: 1 error attempt + 1 successful execute + 1 submit = 3 attempts
        assert len(result.attempts) == 3
        # Exactly one retry-classified attempt (the error-then-corrected pattern)
        retry_count = sum(1 for a in result.attempts if a.is_retry)
        assert retry_count == 1
