"""Single-attempt executor for the Track B sweep.

Runs the agent loop against one (query, backend, rep_idx) and returns a
fully-populated :class:`TrackBMetrics` row. Also writes the raw
attempt log to ``trace.jsonl`` in the phase runner's canonical layout.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp_dbs_bench.agent.correctness import GoldResult, verify
from mcp_dbs_bench.agent.loop import AgentOutcome, AgentRunner, AgentRunResult
from mcp_dbs_bench.agent.openrouter_llm import OpenRouterLLMClient

from track_b._backends import BackendClient
from track_b._mcp_launch import McpLaunchSpec, build_stdio_client
from track_b._metrics import TrackBMetrics

_LOG = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "You are a SQL analyst working through an MCP database server. "
    "The user will ask questions in natural language about a TPC-H "
    "database. Use the MCP tools to answer. "
    "\n\nWhen you have the answer, call submit_answer with the result as "
    "a list of rows. Each row is itself a list of cell values. Return "
    "ONLY the data rows — DO NOT include a header row of column names. "
    "For example, a COUNT query returns [[150000]] — not "
    "[[\"count\"], [150000]]. "
    "\n\nDo not add commentary; return only the data. "
    "Use as few tool calls as possible."
)


async def run_one_attempt(
    *,
    query: dict[str, Any],
    backend: str,
    scale_factor: int,
    rep_idx: int,
    server: str,
    launch: McpLaunchSpec,
    llm_client: OpenRouterLLMClient,
    max_tool_calls: int,
    max_retries: int,
    trace_dir: Path,
    sql_executor: BackendClient | None,
) -> TrackBMetrics:
    """Drive one full agent attempt and return the row of metrics.

    Args:
        query: One entry from the workload YAML (must have ``id``,
            ``question``, ``gold_sql``, ``category``, ``target_subcat``).
        backend: Backend name (postgres / mysql / duckdb).
        scale_factor: TPC-H scale factor (1 or 3 in the published runs).
        rep_idx: Repetition index (0-based).
        server: MCP server name (e.g. ``mcp-alchemy``).
        launch: Precomputed subprocess launch spec.
        llm_client: OpenRouter adapter instance.
        max_tool_calls: Per-attempt tool-call cap.
        max_retries: Per-attempt retry budget.
        trace_dir: Directory for this cell's ``trace.jsonl``.
        sql_executor: Optional executor used only when the agent submits
            SQL instead of rows.
    """
    trace_path = trace_dir / f"rep_{rep_idx:03d}.jsonl"
    trace_dir.mkdir(parents=True, exist_ok=True)
    started_ms = time.monotonic() * 1_000
    run_result, agent_error = await _run_agent_loop(
        launch=launch, llm_client=llm_client,
        question=query["question"],
        max_tool_calls=max_tool_calls, max_retries=max_retries,
    )
    wall_ms = time.monotonic() * 1_000 - started_ms
    _write_trace_file(
        path=trace_path, query=query, backend=backend, rep_idx=rep_idx,
        run_result=run_result, wall_ms=wall_ms, agent_error=agent_error,
    )
    verdict, diagnostic = _verify(
        query=query, backend=backend, scale_factor=scale_factor,
        run_result=run_result, agent_error=agent_error,
        sql_executor=sql_executor,
    )
    return _build_metrics(
        query=query, backend=backend, scale_factor=scale_factor,
        rep_idx=rep_idx, server=server, run_result=run_result,
        wall_ms=wall_ms, verdict=verdict, diagnostic=diagnostic,
        trace_path=str(trace_path), agent_error=agent_error,
    )


async def _run_agent_loop(
    *,
    launch: McpLaunchSpec,
    llm_client: OpenRouterLLMClient,
    question: str,
    max_tool_calls: int,
    max_retries: int,
) -> tuple[AgentRunResult | None, str | None]:
    """Spawn mcp-alchemy, run the agent to completion, return the result."""
    try:
        async with build_stdio_client(launch) as mcp:
            runner = AgentRunner(
                llm=llm_client, mcp=mcp,
                max_tool_calls=max_tool_calls,
                max_retries=max_retries,
            )
            return await runner.run(
                natural_language=question, system_prompt=SYSTEM_PROMPT,
            ), None
    except Exception as exc:
        _LOG.exception("agent loop crashed for question=%r", question)
        return None, f"{type(exc).__name__}: {exc}"


def _verify(
    *,
    query: dict[str, Any],
    backend: str,
    scale_factor: int,
    run_result: AgentRunResult | None,
    agent_error: str | None,
    sql_executor: BackendClient | None,
) -> tuple[str, str]:
    """Route the agent's final answer through the correctness verifier."""
    if agent_error is not None:
        return "agent_error", agent_error
    if run_result is None:
        return "agent_error", "runner returned None"
    if run_result.outcome == AgentOutcome.CALL_LIMIT_EXCEEDED:
        return "call_limit_exceeded", "max tool calls reached"
    if run_result.outcome == AgentOutcome.TIMEOUT:
        return "timeout", "attempt timed out"
    gold_data = _load_gold_data(query["id"], backend, scale_factor)
    gold = GoldResult(
        query_id=query["id"], backend=backend, gold_sql=query["gold_sql"],
        columns=gold_data["columns"], rows=gold_data["rows"],
    )
    verdict, diagnostic = verify(
        run_result.final_answer, gold,
        sql_executor=_sql_executor_adapter(sql_executor),
    )
    return verdict, diagnostic


def _load_gold_data(
    query_id: str, backend: str, scale_factor: int,
) -> dict[str, Any]:
    """Read the captured gold JSON for ``(query_id, backend, sf)``.

    Gold values scale with SF (aggregates like ``SUM(o_totalprice)`` differ),
    so gold results are namespaced by ``gold_results_sf{sf}/``.
    """
    from track_b._mcp_launch import load_gold
    from track_b._paths import gold_results_dir
    return load_gold(
        str(gold_results_dir(sf=scale_factor)), query_id, backend,
    )


def _sql_executor_adapter(client: BackendClient | None):
    """Adapt BackendClient.execute to the verifier's expected signature."""
    if client is None:
        return None
    return client.execute


def _write_trace_file(
    *,
    path: Path,
    query: dict[str, Any],
    backend: str,
    rep_idx: int,
    run_result: AgentRunResult | None,
    wall_ms: float,
    agent_error: str | None,
) -> None:
    """Append one JSON line per attempt event. Line-buffered for crash safety."""
    with path.open("w") as fh:
        header = {
            "type": "attempt_meta", "query_id": query["id"],
            "backend": backend, "rep_idx": rep_idx,
            "question": query["question"], "gold_sql": query["gold_sql"],
            "wall_clock_ms": wall_ms,
            "agent_error": agent_error,
        }
        fh.write(json.dumps(header, default=str) + "\n")
        fh.flush()
        if run_result is None:
            return
        fh.write(json.dumps({
            "type": "outcome", "outcome": run_result.outcome.value,
            "final_answer": run_result.final_answer,
            "total_input_tokens": run_result.total_input_tokens,
            "total_output_tokens": run_result.total_output_tokens,
        }, default=str) + "\n")
        for attempt in run_result.attempts:
            fh.write(json.dumps({
                "type": "attempt", **asdict(attempt),
            }, default=str) + "\n")
            fh.flush()


def _build_metrics(
    *,
    query: dict[str, Any],
    backend: str,
    scale_factor: int,
    rep_idx: int,
    server: str,
    run_result: AgentRunResult | None,
    wall_ms: float,
    verdict: str,
    diagnostic: str,
    trace_path: str,
    agent_error: str | None,
) -> TrackBMetrics:
    """Assemble one row of :class:`TrackBMetrics` from the run result."""
    cell = f"{server}_{backend}_{query['id']}_sf{scale_factor}"
    if run_result is None:
        return TrackBMetrics(
            cell_id=cell, server=server, backend=backend, query=query["id"],
            category=query["category"], target_subcat=query["target_subcat"],
            scale_factor=scale_factor, rep_idx=rep_idx,
            outcome=verdict, verifier_diagnostic=diagnostic,
            wall_clock_ms=wall_ms, trace_path=trace_path,
        )
    tool_names = [a.tool_name for a in run_result.attempts]
    n_tool_calls = sum(1 for a in run_result.attempts if a.tool_name != "submit_answer")
    retry_count = sum(1 for a in run_result.attempts if a.is_retry)
    phi = _safe_ratio(
        run_result.total_input_tokens, run_result.total_output_tokens,
    )
    kind = _classify_answer(run_result.final_answer)
    return TrackBMetrics(
        cell_id=cell, server=server, backend=backend, query=query["id"],
        category=query["category"], target_subcat=query["target_subcat"],
        scale_factor=scale_factor, rep_idx=rep_idx,
        outcome=verdict, verifier_diagnostic=diagnostic,
        attempts=len(run_result.attempts),
        retry_count=retry_count,
        gamma_cumulative_input_tokens=run_result.total_input_tokens,
        phi_token_inflation=phi,
        n_turns=len(run_result.attempts),
        n_tool_calls=n_tool_calls,
        total_input_tokens=run_result.total_input_tokens,
        total_output_tokens=run_result.total_output_tokens,
        cost_usd=0.0,  # filled in by the sweep runner from cost tracker
        wall_clock_ms=wall_ms,
        trace_path=trace_path,
        final_answer_kind=kind,
        tool_names=tool_names,
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """Guarded division; returns None on zero denominator."""
    if denominator <= 0:
        return None
    return numerator / denominator


def _classify_answer(answer: Any) -> str:
    """Coarse classification for the ``final_answer_kind`` column."""
    if answer is None:
        return "none"
    if isinstance(answer, str):
        return "sql_or_string"
    if isinstance(answer, (int, float, bool)):
        return "scalar"
    if isinstance(answer, dict):
        return "dict"
    if isinstance(answer, list):
        return "list"
    return type(answer).__name__


__all__ = ["SYSTEM_PROMPT", "run_one_attempt"]
