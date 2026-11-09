"""MCP Python SDK adapter over stdio.

Implements the :class:`MCPClient` protocol against the official
``mcp`` Python SDK (no LangChain, no LiteLLM). Spawns the server as a child
process and communicates over stdio.

"""

from __future__ import annotations

import logging
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any, Self

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPStdioClient(AbstractAsyncContextManager["MCPStdioClient"]):
    """Connect to an MCP server via stdio and forward tool calls.

    Use as an async context manager so the subprocess and session are
    deterministically torn down on exit::

        async with MCPStdioClient(command="mcp-alchemy", args=["--db", "duckdb:///t.duckdb"]) as client:
            tools = await client.list_tools()
            result = await client.call_tool("execute_sql", {"sql": "SELECT 1"})

    Implementation note: an ``AsyncExitStack`` pops both
    inner context managers in the same task that entered them, which
    anyio's cancel-scope ownership invariant requires.

    Args:
        command: Executable launching the MCP server.
        args: Arguments passed to the executable.
        env: Optional environment vars merged with the parent process env.
    """

    def __init__(
        self,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._params = StdioServerParameters(command=command, args=args or [], env=env)
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> Self:
        self._exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(self._params)
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            return self
        except BaseException:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's tool manifest in Anthropic Messages API shape."""
        if self._session is None:
            raise RuntimeError("MCPStdioClient not entered; use 'async with'")
        result = await self._session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
            }
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool and unwrap its content."""
        if self._session is None:
            raise RuntimeError("MCPStdioClient not entered; use 'async with'")
        try:
            result = await self._session.call_tool(name=name, arguments=arguments)
        except Exception as exc:
            logger.exception("MCP tool call failed for %s", name)
            return {"error": f"{type(exc).__name__}: {exc}"}

        if result.isError:
            return {"error": _extract_text(result.content)}
        return _extract_text(result.content)


def _extract_text(content: list[Any]) -> str:
    """Concatenate the textual portion of an MCP tool result.

    MCP tool results are a list of content blocks (text / image / resource).
    Track B only reads text content; non-text blocks are stringified.
    """
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(str(block))
    return "\n".join(parts)
