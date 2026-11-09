"""Built-in MCP server adapters.

Each adapter conforms to ``core.plugin.MCPServerPlugin`` and is registered
through the ``mcp_dbs_bench.servers`` entry-point group — the same channel
third-party plugins use.

Adapters:
    - ``anthropic_pg`` : Anthropic's reference PostgreSQL MCP server (archived).
    - ``mcp_alchemy``  : SQLAlchemy-backed; covers PG + MySQL + DuckDB.
    - ``dbhub``        : Bytebase's gateway server.
    - ``direct``       : no MCP server; in-process database reference.

Each MCP adapter spawns the upstream server process via stdio and translates
the JSON-RPC traffic into PhaseContext events. Upstream versions are pinned
in ``scripts/setup/03_install_mcp_servers.sh``.
"""

from __future__ import annotations
