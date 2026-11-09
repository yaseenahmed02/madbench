"""Subprocess MCP client: JSON-RPC 2.0 over LF-framed stdio.

The upstream MCP server adapters share this transport. Each upstream
implements its own framing detail (banner on stderr, varying tool names,
content-shape) but the JSON-RPC envelope and the LF framing are identical
across mcp-alchemy, dbhub, and mcp-server-postgres.

This module owns the subprocess lifecycle (spawn, drain stderr to a log,
terminate on close) and the I/O accounting (bytes sent/received) used to
compute IM-2 (B_mcp) and CM-2 (rho). It does NOT know about phases, OTel,
or per-server tool schemas — those live in each adapter.

Property 2 (exclusivity) is preserved because the adapter wraps individual
client method calls inside ``phase.span(...)`` blocks; this module exposes
fine-grained `initialize / tools_list / call_tool` entry points so the
adapter can place each in its own phase scope.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_dbs_bench.core.exceptions import MCPDBSBenchError

_LOG = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"
# Bumped from 300.0 to 900.0: MySQL Q9 SF10 (6-table join on 60M rows) on
# vanilla MySQL 8 without FK indexes can exceed 300s for a single rep.
# Observed failure mode: SubprocessProtocolError abandons the cell with
# only the warmup file written. Allow ~15 min for the worst single-rep case.
_READ_TIMEOUT_S = 900.0
_TERMINATE_TIMEOUT_S = 5.0


class SubprocessClosedError(MCPDBSBenchError):
    """Operation attempted on a client that was never started or already closed."""


class SubprocessProtocolError(MCPDBSBenchError):
    """The child process returned a malformed JSON-RPC frame."""


class SubprocessToolError(MCPDBSBenchError):
    """The server returned ``isError: true`` on a ``tools/call`` response."""


@dataclass(frozen=True, slots=True)
class _ClientInfo:
    """Minimal JSON-RPC client identification."""

    name: str = "madbench"
    version: str = "0.1.0"


class SubprocessMCPClient:
    """Drives one stdio MCP child process.

    Lifecycle: ``start()`` spawns the process; ``initialize()`` performs the
    handshake; ``tools_list()`` and ``call_tool()`` issue requests; ``close()``
    terminates the child and stops the stderr drain task. ``close()`` is
    idempotent. The harness must call ``close()`` once per ``start()``.
    """

    def __init__(
        self,
        binary: str,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        stderr_log_path: Path | None = None,
        client_info: _ClientInfo | None = None,
    ) -> None:
        self._binary = binary
        self._args = list(args)
        self._env = dict(env) if env else {}
        self._stderr_log_path = stderr_log_path
        self._client_info = client_info or _ClientInfo()
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_buf: list[bytes] = []
        self._request_id = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self._closed = False

    @property
    def is_started(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def child_pid(self) -> int | None:
        """OS pid of the spawned child, or None before ``start()`` / after exit."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return None
        return proc.pid

    async def start(self) -> None:
        """Spawn the child process and begin draining stderr."""
        if self._proc is not None:
            raise RuntimeError("SubprocessMCPClient.start() called twice")
        proc = await asyncio.create_subprocess_exec(
            self._binary,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self._env},
        )
        self._proc = proc
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def initialize(self) -> dict[str, Any]:
        """Send the JSON-RPC ``initialize`` + ``notifications/initialized`` pair."""
        proc = self._require_started()
        init_resp = await self._request(
            proc,
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": self._client_info.name,
                    "version": self._client_info.version,
                },
            },
        )
        await self._notify(proc, "notifications/initialized", {})
        return init_resp.get("result", {})

    async def tools_list(self) -> tuple[list[dict[str, Any]], bytes]:
        """Issue ``tools/list``. Returns (tools, raw_response_bytes)."""
        proc = self._require_started()
        envelope, raw = await self._request_with_raw(proc, "tools/list", None)
        result = envelope.get("result", {})
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            raise SubprocessProtocolError(
                f"tools/list returned non-list 'tools'; got {type(tools).__name__}"
            )
        return tools, raw

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, bytes, bool]:
        """Issue ``tools/call``. Returns (concatenated_text, raw_bytes, is_error)."""
        proc = self._require_started()
        envelope, raw = await self._request_with_raw(
            proc, "tools/call", {"name": name, "arguments": arguments}
        )
        result = envelope.get("result")
        if not isinstance(result, dict):
            error = envelope.get("error", {})
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise SubprocessProtocolError(
                f"tools/call returned no 'result' for tool {name!r}: {message}"
            )
        is_error = bool(result.get("isError", False))
        text = _join_text_content(result.get("content", []))
        return text, raw, is_error

    async def close(self) -> None:
        """Terminate the child and drain stderr. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None and proc.stdin is not None and not proc.stdin.is_closing():
            try:
                proc.stdin.close()
            except (ConnectionResetError, BrokenPipeError):
                pass
        if proc is not None and proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_TIMEOUT_S)
            except asyncio.TimeoutError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_TIMEOUT_S)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        self._write_stderr_log()

    def _require_started(self) -> asyncio.subprocess.Process:
        if self._proc is None or self._closed:
            raise SubprocessClosedError(
                "SubprocessMCPClient is not running; call start() first."
            )
        return self._proc

    async def _request(
        self,
        proc: asyncio.subprocess.Process,
        method: str,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        envelope, _ = await self._request_with_raw(proc, method, params)
        return envelope

    async def _request_with_raw(
        self,
        proc: asyncio.subprocess.Process,
        method: str,
        params: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], bytes]:
        self._request_id += 1
        request = {"jsonrpc": "2.0", "id": self._request_id, "method": method}
        if params is not None:
            request["params"] = params
        await self._send(proc, request)
        raw = await self._read_line(proc)
        envelope = _parse_jsonrpc(raw)
        return envelope, raw

    async def _notify(
        self,
        proc: asyncio.subprocess.Process,
        method: str,
        params: dict[str, Any],
    ) -> None:
        await self._send(proc, {"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(
        self,
        proc: asyncio.subprocess.Process,
        msg: dict[str, Any],
    ) -> None:
        if proc.stdin is None:
            raise SubprocessClosedError("child process has no stdin")
        frame = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        proc.stdin.write(frame)
        await proc.stdin.drain()
        self.bytes_sent += len(frame)

    async def _read_line(self, proc: asyncio.subprocess.Process) -> bytes:
        if proc.stdout is None:
            raise SubprocessClosedError("child process has no stdout")
        try:
            raw = await asyncio.wait_for(
                proc.stdout.readline(), timeout=_READ_TIMEOUT_S
            )
        except asyncio.TimeoutError as exc:
            raise SubprocessProtocolError(
                f"timed out waiting for response after {_READ_TIMEOUT_S}s"
            ) from exc
        if not raw:
            raise SubprocessProtocolError("child closed stdout before responding")
        self.bytes_received += len(raw)
        return raw

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            try:
                line = await proc.stderr.readline()
            except asyncio.CancelledError:
                return
            if not line:
                return
            self._stderr_buf.append(line)

    def _write_stderr_log(self) -> None:
        if not self._stderr_log_path or not self._stderr_buf:
            return
        try:
            self._stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._stderr_log_path.open("ab") as fh:
                for line in self._stderr_buf:
                    fh.write(line)
        except OSError as exc:
            _LOG.warning("failed to write stderr log %s: %s", self._stderr_log_path, exc)


def _parse_jsonrpc(raw: bytes) -> dict[str, Any]:
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        snippet = raw[:200].decode("utf-8", errors="replace")
        raise SubprocessProtocolError(
            f"malformed JSON-RPC frame from child: {snippet!r}"
        ) from exc
    if not isinstance(envelope, dict):
        raise SubprocessProtocolError(
            f"JSON-RPC frame must be an object; got {type(envelope).__name__}"
        )
    return envelope


def _join_text_content(content: Any) -> str:
    """Concatenate all ``text`` blocks in a tools/call result.

    MCP responses can include multiple content blocks; the convention across
    the four servers we wrap is a single text block, but we tolerate multiple.
    """
    if not isinstance(content, list):
        return ""
    pieces: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                pieces.append(text)
    return "".join(pieces)


__all__ = [
    "SubprocessClosedError",
    "SubprocessMCPClient",
    "SubprocessProtocolError",
    "SubprocessToolError",
]
