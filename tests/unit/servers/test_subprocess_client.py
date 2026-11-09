"""Unit tests for the subprocess MCP client.

The client speaks JSON-RPC 2.0 over LF-framed stdio. These tests use a
fake stdio server implemented as a tiny Python script piped through
``asyncio.create_subprocess_exec``. No upstream binary is required.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from mcp_dbs_bench.servers._subprocess import (
    SubprocessClosedError,
    SubprocessMCPClient,
    SubprocessProtocolError,
)


_ECHO_SERVER_SOURCE = textwrap.dedent(
    """
    import json
    import sys

    def write(obj):
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method = msg.get("method")
        if "id" not in msg:
            continue  # notification, no reply
        if method == "initialize":
            write({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo", "version": "0.0"},
            }})
        elif method == "tools/list":
            write({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "tools": [{"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}}]
            }})
        elif method == "tools/call":
            args = msg.get("params", {}).get("arguments", {})
            write({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text", "text": json.dumps(args)}],
                "isError": False,
            }})
    """
)


@pytest.fixture
def echo_server_script(tmp_path: Path) -> Path:
    p = tmp_path / "echo_server.py"
    p.write_text(_ECHO_SERVER_SOURCE)
    return p


async def test_client_initialize_handshake_succeeds(echo_server_script: Path):
    client = SubprocessMCPClient(sys.executable, [str(echo_server_script)])
    await client.start()
    try:
        info = await client.initialize()
        assert info["serverInfo"]["name"] == "echo"
        assert client.bytes_sent > 0
        assert client.bytes_received > 0
    finally:
        await client.close()


async def test_client_tools_list_returns_raw_bytes(echo_server_script: Path):
    client = SubprocessMCPClient(sys.executable, [str(echo_server_script)])
    await client.start()
    try:
        await client.initialize()
        tools, raw = await client.tools_list()
        assert tools[0]["name"] == "echo"
        assert json.loads(raw)["result"]["tools"][0]["name"] == "echo"
    finally:
        await client.close()


async def test_client_call_tool_returns_text_and_bytes(echo_server_script: Path):
    client = SubprocessMCPClient(sys.executable, [str(echo_server_script)])
    await client.start()
    try:
        await client.initialize()
        text, raw, is_error = await client.call_tool("echo", {"hello": "world"})
        assert json.loads(text) == {"hello": "world"}
        assert is_error is False
        assert len(raw) > 0
    finally:
        await client.close()


async def test_client_records_byte_counts_per_call(echo_server_script: Path):
    client = SubprocessMCPClient(sys.executable, [str(echo_server_script)])
    await client.start()
    try:
        bytes_before = client.bytes_sent
        await client.initialize()
        bytes_after_init = client.bytes_sent
        await client.call_tool("echo", {"x": 1})
        bytes_after_call = client.bytes_sent
        assert bytes_after_init > bytes_before
        assert bytes_after_call > bytes_after_init
    finally:
        await client.close()


async def test_client_close_is_idempotent(echo_server_script: Path):
    client = SubprocessMCPClient(sys.executable, [str(echo_server_script)])
    await client.start()
    await client.close()
    await client.close()  # must not raise


async def test_client_raises_when_used_before_start(echo_server_script: Path):
    client = SubprocessMCPClient(sys.executable, [str(echo_server_script)])
    with pytest.raises(SubprocessClosedError):
        await client.initialize()


async def test_client_raises_on_malformed_json(tmp_path: Path):
    bad_server = tmp_path / "bad_server.py"
    bad_server.write_text(
        "import sys\nfor line in sys.stdin:\n    sys.stdout.write('not json\\n')\n    sys.stdout.flush()\n"
    )
    client = SubprocessMCPClient(sys.executable, [str(bad_server)])
    await client.start()
    try:
        with pytest.raises(SubprocessProtocolError, match="malformed"):
            await client.initialize()
    finally:
        await client.close()
