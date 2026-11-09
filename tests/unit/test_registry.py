"""Tests for the entry-point plugin registry."""

from __future__ import annotations

import pytest

from mcp_dbs_bench.core.exceptions import PluginRegistrationError
from mcp_dbs_bench.core.registry import (
    available_backends,
    available_servers,
    load_backend,
    load_server,
)


def test_load_unknown_server_raises():
    with pytest.raises(PluginRegistrationError, match="not_a_real_server"):
        load_server("not_a_real_server")


def test_load_unknown_backend_raises():
    with pytest.raises(PluginRegistrationError, match="not_a_real_backend"):
        load_backend("not_a_real_backend")


def test_available_servers_returns_list_of_strings():
    # Order-independent; may contain built-ins if entry points are registered.
    result = available_servers()
    assert isinstance(result, list)
    for name in result:
        assert isinstance(name, str)


def test_available_backends_returns_list_of_strings():
    result = available_backends()
    assert isinstance(result, list)
    for name in result:
        assert isinstance(name, str)


def test_register_and_load_server_via_in_memory_table():
    # The registry supports an explicit override table for tests and CLI plumbing.
    from mcp_dbs_bench.core.registry import register_server

    class _Dummy:
        name = "dummy"
        version = "0.0.1"

        def capabilities(self):
            raise NotImplementedError

        def connect(self, backend, phase):
            raise NotImplementedError

        async def discover(self, conn, phase):
            raise NotImplementedError

        async def call_tool(self, conn, request, phase):
            raise NotImplementedError

    register_server("dummy", _Dummy)
    plugin = load_server("dummy")
    assert plugin.__class__ is _Dummy
    assert plugin.name == "dummy"
