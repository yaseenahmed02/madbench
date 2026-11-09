"""In-memory stubs for unit-testing the agent loop without an LLM or MCP server."""

from __future__ import annotations

import json
from typing import Any


class StubLLMClient:
    """Yield pre-canned responses in order. Raises if exhausted."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def next_turn(
        self,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._index >= len(self._responses):
            raise IndexError(
                f"StubLLMClient exhausted after {self._index} turns; "
                f"add more responses to the fixture."
            )
        response = self._responses[self._index]
        self._index += 1
        return response


class StubMCPClient:
    """Return canned tool responses keyed by tool name or (name, json-input)."""

    def __init__(
        self,
        tools: list[dict[str, Any]],
        tool_responses: dict[Any, Any],
    ) -> None:
        self._tools = list(tools)
        self._responses = dict(tool_responses)

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        key_specific = (name, json.dumps(arguments, sort_keys=False))
        if key_specific in self._responses:
            return self._responses[key_specific]
        if name in self._responses:
            return self._responses[name]
        return {"error": f"No stub response for tool '{name}'"}
