"""OpenRouter (OpenAI-compatible) adapter for Track B.

Lets Track B run against any OpenAI-tool-calling-compatible model exposed
through https://openrouter.ai — DeepSeek, GPT, Mistral, Llama, etc — using
one agent loop. A single API key reaches every model under test.

Format bridging
---------------
The agent loop (``loop.py``) builds messages in Anthropic format (tool_use
content blocks + tool_result content blocks). OpenAI Chat Completions uses
a different shape (assistant.tool_calls + a separate tool-role message per
result). This adapter translates Anthropic-format messages into OpenAI
format on the way out, and OpenAI-format responses back into the
``{tool_calls, input_tokens, output_tokens}`` shape the loop expects.

Environment variables
---------------------
- ``OPENROUTER_API_KEY``: bearer token for OpenRouter.
- ``OPENROUTER_LLM_MODEL`` (optional): default model id, e.g.
  ``deepseek/deepseek-v4-flash`` or ``openai/gpt-4o-mini``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_LOGGED_TEXT_CHARS = 300


OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
TRACK_B_TEMPERATURE: float = 0.0
TRACK_B_MAX_OUTPUT_TOKENS: int = 4096


class OpenRouterLLMClient:
    """Adapter implementing the :class:`LLMClient` protocol via OpenRouter.

    Args:
        api_key: OpenRouter API key. Defaults to ``OPENROUTER_API_KEY``.
        model: OpenRouter model id. Defaults to ``OPENROUTER_LLM_MODEL`` env
            var, falling back to ``deepseek/deepseek-v4-flash``.
        temperature: Locked to 0.0 for reproducibility.
        max_output_tokens: Output cap per turn.
        timeout_s: HTTP timeout for each request.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = TRACK_B_TEMPERATURE,
        max_output_tokens: int = TRACK_B_MAX_OUTPUT_TOKENS,
        timeout_s: float = 120.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenRouterLLMClient requires an API key. Pass api_key= or set "
                "OPENROUTER_API_KEY in the environment."
            )
        self._model = (
            model
            or os.environ.get("OPENROUTER_LLM_MODEL")
            or "deepseek/deepseek-v4-flash"
        )
        if temperature != 0.0:
            logger.warning(
                "Track B convention locks temperature=0.0; running with %s",
                temperature,
            )
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s

    async def next_turn(
        self,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Translate, call OpenRouter, translate back."""
        openai_messages = _anthropic_to_openai_messages(system_prompt, messages)
        openai_tools = [_anthropic_tool_to_openai(t) for t in tools]
        body = {
            "model": self._model,
            "messages": openai_messages,
            "tools": openai_tools,
            "tool_choice": "auto",
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Title": "madbench-track-b",
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions", json=body, headers=headers,
            )
        if resp.is_error:
            logger.warning(
                "OpenRouter returned HTTP %d for model %s: %s",
                resp.status_code, self._model, resp.text[:_LOGGED_TEXT_CHARS],
            )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"]) if fn.get("arguments") else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc["id"], "name": fn["name"], "input": args})
        if not tool_calls:
            logger.warning(
                "Model %s via provider %s returned no tool call "
                "(finish_reason=%s): %r",
                self._model, data.get("provider"),
                data["choices"][0].get("finish_reason"),
                (msg.get("content") or "")[:_LOGGED_TEXT_CHARS],
            )
        return {
            "tool_calls": tool_calls,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        }


def _anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic-format tool (name/description/input_schema) to OpenAI."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description") or "",
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _anthropic_to_openai_messages(
    system_prompt: str, messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate Anthropic message list into OpenAI Chat Completions shape.

    The loop builds ``messages`` with three shapes:
      - ``{"role": "user", "content": "..."}``                       — initial question
      - ``{"role": "assistant", "content": [tool_use blocks...]}``    — model turn
      - ``{"role": "user", "content": [tool_result blocks...]}``      — tool results
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            # Anthropic tool_use list → OpenAI assistant.tool_calls
            tool_calls = []
            text_parts = []
            for block in content:
                btype = block.get("type")
                if btype == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    })
                elif btype == "text":
                    text_parts.append(block.get("text", ""))
            out.append({
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls or None,
            })
        elif role == "user":
            # Anthropic tool_result blocks → one OpenAI tool-role message per result.
            for block in content:
                if block.get("type") == "tool_result":
                    body = block.get("content", "")
                    if not isinstance(body, str):
                        body = json.dumps(body)
                    out.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": body,
                    })
                else:
                    out.append({"role": "user", "content": str(block)})
        else:
            out.append({"role": role, "content": str(content)})
    return out
