"""Unit tests for ``MCPStdioClient``.

These unit tests cover the construction contract,
the unguarded-session error path, and the text-extraction helper without
any subprocess I/O.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mcp_dbs_bench.agent.mcp_client import MCPStdioClient, _extract_text


def test_constructor_stores_command_args_and_env() -> None:
    """Constructor must accept command/args/env and not eagerly spawn."""
    client = MCPStdioClient(
        command="mcp-alchemy",
        args=["--db", "duckdb:///t.duckdb"],
        env={"FOO": "bar"},
    )
    # Internals are private; verify by checking the session is not yet open.
    assert client._session is None


def test_constructor_defaults_args_to_empty_list() -> None:
    """``args=None`` must default to an empty list (StdioServerParameters
    rejects None for the args field)."""
    client = MCPStdioClient(command="mcp-alchemy")
    assert client._params.args == []


@pytest.mark.asyncio
async def test_list_tools_raises_when_not_entered() -> None:
    """Calling list_tools without ``async with`` must raise a clear error."""
    client = MCPStdioClient(command="mcp-alchemy")
    with pytest.raises(RuntimeError, match="not entered"):
        await client.list_tools()


@pytest.mark.asyncio
async def test_call_tool_raises_when_not_entered() -> None:
    """Calling call_tool without entering must raise a clear error."""
    client = MCPStdioClient(command="mcp-alchemy")
    with pytest.raises(RuntimeError, match="not entered"):
        await client.call_tool("execute_sql", {"sql": "SELECT 1"})


def test_extract_text_concatenates_text_blocks() -> None:
    """Multiple text blocks are joined by newline."""
    blocks = [
        SimpleNamespace(text="line one"),
        SimpleNamespace(text="line two"),
    ]
    assert _extract_text(blocks) == "line one\nline two"


def test_extract_text_handles_non_text_blocks() -> None:
    """Non-text blocks fall back to ``str(block)`` and are still joined."""
    class _NoText:
        def __str__(self) -> str:
            return "[image]"

    blocks = [
        SimpleNamespace(text="text"),
        _NoText(),
        SimpleNamespace(text=None),  # explicit None text path
    ]
    out = _extract_text(blocks)
    assert "text" in out
    assert "[image]" in out


@pytest.mark.asyncio
async def test_list_tools_translates_mcp_tool_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate ``mcp.Tool`` objects into Anthropic Messages API tool dicts."""

    class _StubSession:
        async def initialize(self) -> None:
            pass

        async def list_tools(self) -> Any:
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="execute_sql",
                        description="Run a SQL statement",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ],
            )

    client = MCPStdioClient(command="mcp-alchemy")
    client._session = _StubSession()
    tools = await client.list_tools()
    assert tools == [
        {
            "name": "execute_sql",
            "description": "Run a SQL statement",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


@pytest.mark.asyncio
async def test_call_tool_returns_error_dict_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raised exception inside the MCP SDK is normalised to an
    ``{"error": "..."}`` dict so the agent loop can classify it.
    """

    class _StubSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            raise RuntimeError("boom")

    client = MCPStdioClient(command="mcp-alchemy")
    client._session = _StubSession()
    result = await client.call_tool("execute_sql", {"sql": "x"})
    assert isinstance(result, dict)
    assert "error" in result
    assert "RuntimeError" in result["error"]


@pytest.mark.asyncio
async def test_call_tool_returns_error_dict_when_isError_true() -> None:
    """An ``isError=True`` MCP response is converted to ``{"error": text}``."""

    class _StubSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            return SimpleNamespace(
                isError=True,
                content=[SimpleNamespace(text="SQL parse error")],
            )

    client = MCPStdioClient(command="mcp-alchemy")
    client._session = _StubSession()
    result = await client.call_tool("execute_sql", {"sql": "x"})
    assert isinstance(result, dict)
    assert result["error"] == "SQL parse error"


@pytest.mark.asyncio
async def test_call_tool_extracts_text_on_success() -> None:
    """An ``isError=False`` response returns the concatenated text content."""

    class _StubSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text="row1\nrow2")],
            )

    client = MCPStdioClient(command="mcp-alchemy")
    client._session = _StubSession()
    result = await client.call_tool("execute_sql", {"sql": "x"})
    assert result == "row1\nrow2"
