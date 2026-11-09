"""Launch MCP server subprocesses for Track B.

Supports three servers:

  * ``mcp-alchemy`` (Python, uv-installed) — DSN goes via ``DB_URL`` in
    the child env. DuckDB uses the local ``mcp_alchemy_shim.py`` to work
    around a duckdb_engine/duckdb 1.0 incompatibility.
  * ``anthropic-pg`` (Node, ``@modelcontextprotocol/server-postgres``) —
    PostgreSQL only. DSN is the sole positional argv.
  * ``dbhub`` (Node, ``@bytebase/dbhub``) — PostgreSQL + MySQL. DSN goes
    via ``--dsn=<url>`` and stdio transport is selected by
    ``--transport=stdio``.

Downstream (``_attempt.py``, ``run_track_b_sweep.py``) is server-agnostic:
each accepts an :class:`McpLaunchSpec` and only ever calls
:func:`build_stdio_client` on it. Each server's DSN shape, binary
discovery path, and env requirements live entirely inside this module.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp_dbs_bench.agent.mcp_client import MCPStdioClient

_LOG = logging.getLogger(__name__)

SUPPORTED_SERVERS: tuple[str, ...] = ("mcp-alchemy", "anthropic-pg", "dbhub")

_MCP_ALCHEMY_BINARY_NAME: str = "mcp-alchemy"
_ANTHROPIC_PG_BINARY_NAME: str = "mcp-server-postgres"
_DBHUB_BINARY_NAME: str = "dbhub"

_PG_HOST: str = "localhost"
_PG_PORT: int = 55432
_PG_USER: str = "bench"
_PG_PASSWORD: str = "bench"
_PG_DATABASE: str = "tpch"
_DBHUB_SEARCH_PATH_ENV: str = "DBHUB_SEARCH_PATH_CONFIG"
_MYSQL_HOST: str = "localhost"
_MYSQL_PORT: int = 53306
_MYSQL_USER: str = "bench"
_MYSQL_PASSWORD: str = "bench"


@dataclass(frozen=True, slots=True)
class McpLaunchSpec:
    """Everything the launcher needs to spawn one MCP server child.

    Attributes:
        binary_path: Executable to spawn.
        extra_args: Extra positional argv passed after the executable.
        env_extra: Env vars to inject into the child (mcp-alchemy uses
            ``DB_URL`` here; Node servers get an empty dict since DSN
            travels via argv).
        path_prepend: Directory prepended to child ``PATH`` before the
            inherited PATH. Required for Node servers whose shebang is
            ``#!/usr/bin/env node`` — without ``node`` on the child's
            PATH, ``env`` will fail with "node: No such file".
        db_url: Retained for backward compatibility with callers that
            read ``spec.db_url`` for logging. Not used at spawn time.
    """

    binary_path: str
    extra_args: tuple[str, ...] = ()
    env_extra: dict[str, str] = field(default_factory=dict)
    path_prepend: str = ""
    db_url: str = ""


def build_stdio_client(spec: McpLaunchSpec) -> MCPStdioClient:
    """Wrap a launch spec in an :class:`MCPStdioClient` (async ctx manager)."""
    env = _child_env(env_extra=spec.env_extra, path_prepend=spec.path_prepend)
    return MCPStdioClient(
        command=spec.binary_path, args=list(spec.extra_args), env=env,
    )


def _child_env(
    *, env_extra: dict[str, str], path_prepend: str,
) -> dict[str, str]:
    """Build the child process env.

    Preserves only a minimal parent env (PATH, HOME, locale, TZ) then
    applies ``path_prepend`` and ``env_extra`` on top.
    """
    keep_keys = {"PATH", "HOME", "LANG", "LC_ALL", "TZ"}
    env: dict[str, str] = {k: v for k, v in os.environ.items() if k in keep_keys}
    if path_prepend:
        parent_path = env.get("PATH", "")
        env["PATH"] = f"{path_prepend}:{parent_path}" if parent_path else path_prepend
    env.update(env_extra)
    return env


def duckdb_url(file_path: str) -> str:
    """Build the SQLAlchemy URL for a DuckDB file (absolute path).

    SQLAlchemy convention: absolute paths use four slashes
    (``scheme:////abs/path``) so ``/tmp/...`` is not treated as relative
    to the process CWD.
    """
    abs_path = file_path.lstrip("/")
    return f"duckdb:////{abs_path}"


def postgres_sqlalchemy_url(*, scale_factor: int) -> str:
    """Build the psycopg SQLAlchemy URL used by mcp-alchemy."""
    schema = f"sf{scale_factor}"
    return (
        f"postgresql+psycopg://{_PG_USER}:{_PG_PASSWORD}"
        f"@{_PG_HOST}:{_PG_PORT}/{_PG_DATABASE}"
        f"?options=-c%20search_path%3D{schema}"
    )


def mysql_sqlalchemy_url(*, scale_factor: int) -> str:
    """Build the pymysql SQLAlchemy URL used by mcp-alchemy."""
    database = f"tpch_sf{scale_factor}"
    return f"mysql+pymysql://{_MYSQL_USER}:{_MYSQL_PASSWORD}@{_MYSQL_HOST}:{_MYSQL_PORT}/{database}"


def postgres_native_url(*, scale_factor: int) -> str:
    """Build the libpq-style PG URL consumed by anthropic-pg / dbhub.

    Includes ``search_path`` in the query-string options so per-SF schema
    routing works uniformly. Parallel-worker knobs are disabled to match
    the Track A DSN produced by :func:`track_a.phase_runner._apply_namespace`
    (Docker /dev/shm is only 64M — PG parallel workers fail to allocate DSM).
    """
    schema = f"sf{scale_factor}"
    opts = (
        f"-csearch_path%3D{schema}"
        "%20-cmax_parallel_workers_per_gather%3D0"
        "%20-cmax_parallel_workers%3D0"
    )
    return (
        f"postgresql://{_PG_USER}:{_PG_PASSWORD}"
        f"@{_PG_HOST}:{_PG_PORT}/{_PG_DATABASE}"
        f"?options={opts}"
    )


def mysql_native_url(*, scale_factor: int) -> str:
    """Build the native MySQL URL consumed by dbhub."""
    database = f"tpch_sf{scale_factor}"
    return f"mysql://{_MYSQL_USER}:{_MYSQL_PASSWORD}@{_MYSQL_HOST}:{_MYSQL_PORT}/{database}"


def _resolve_mcp_alchemy_binary() -> str:
    """Return the absolute path to ``mcp-alchemy`` or raise a clear error."""
    path = shutil.which(_MCP_ALCHEMY_BINARY_NAME)
    if path:
        return path
    home = os.environ.get("HOME", "")
    candidates = [
        f"{home}/.local/share/uv/tools/mcp-alchemy/bin/mcp-alchemy",
        f"{home}/.local/bin/mcp-alchemy",
        f"{home}/.pipx/venvs/mcp-alchemy/bin/mcp-alchemy",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "mcp-alchemy binary not found on PATH or in common install locations. "
        f"Probed: PATH, {candidates}. Install with: pip install mcp-alchemy"
    )


# Public alias for callers outside this module.
resolve_binary_path = _resolve_mcp_alchemy_binary


def _resolve_node_binary(name: str) -> str:
    """Locate a Node-based MCP server binary via nvm-aware discovery.

    Mirrors :func:`mcp_dbs_bench.servers._binary_discovery.find_binary` so
    the two subsystems agree on which binary is picked up.
    """
    path = shutil.which(name)
    if path:
        return path
    home = Path(os.path.expanduser("~"))
    candidates: list[Path] = []
    nvm_root = home / ".nvm" / "versions" / "node"
    if nvm_root.is_dir():
        for node_ver_dir in sorted(nvm_root.iterdir(), reverse=True):
            candidates.append(node_ver_dir / "bin" / name)
    candidates.extend([
        home / ".local" / "bin" / name,
        Path("/usr/local/bin") / name,
        Path("/opt/homebrew/bin") / name,
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(
        f"{name} binary not found on PATH or in common install locations. "
        f"Probed: PATH, {[str(c) for c in candidates]}. "
        f"Install with: npm i -g <upstream-package> (or use uv tool)."
    )


def _resolve_shim_command() -> list[str]:
    """Return the argv that runs the DuckDB-compat shim inside our .venv."""
    repo = Path(__file__).resolve().parents[2]
    interp = repo / ".venv" / "bin" / "python"
    shim = repo / "scripts" / "track_b" / "mcp_alchemy_shim.py"
    if not interp.exists() or not shim.exists():
        return []
    return [str(interp), str(shim)]


def _build_alchemy_launch(*, backend: str, scale_factor: int) -> McpLaunchSpec:
    """Assemble the launch spec for the ``mcp-alchemy`` upstream."""
    if backend == "duckdb":
        shim = _resolve_shim_command()
        if shim:
            binary, extra = shim[0], tuple(shim[1:])
        else:
            binary, extra = _resolve_mcp_alchemy_binary(), ()
        url = duckdb_url(f"/tmp/madbench_duckdb_sf{scale_factor}.duckdb")
    elif backend == "postgres":
        binary, extra = _resolve_mcp_alchemy_binary(), ()
        url = postgres_sqlalchemy_url(scale_factor=scale_factor)
    elif backend == "mysql":
        binary, extra = _resolve_mcp_alchemy_binary(), ()
        url = mysql_sqlalchemy_url(scale_factor=scale_factor)
    else:
        raise ValueError(f"unsupported backend for mcp-alchemy: {backend!r}")
    return McpLaunchSpec(
        binary_path=binary, extra_args=extra,
        env_extra={"DB_URL": url}, db_url=url,
    )


def _build_anthropic_pg_launch(
    *, backend: str, scale_factor: int,
) -> McpLaunchSpec:
    """Assemble the launch spec for the ``mcp-server-postgres`` upstream."""
    if backend != "postgres":
        raise ValueError(
            f"anthropic-pg only supports the postgres backend; got {backend!r}"
        )
    binary = _resolve_node_binary(_ANTHROPIC_PG_BINARY_NAME)
    url = postgres_native_url(scale_factor=scale_factor)
    return McpLaunchSpec(
        binary_path=binary, extra_args=(url,),
        env_extra={}, path_prepend=str(Path(binary).parent), db_url=url,
    )


def write_dbhub_postgres_config(*, scale_factor: int, directory: Path) -> Path:
    """Write a dbhub TOML source that pins the per-scale-factor schema.

    dbhub drops the ``options`` query parameter of a PostgreSQL DSN, so the
    ``search_path`` in :func:`postgres_native_url` never reaches the session
    and the agent sees every ``sf*`` schema. Its ``search_path`` source setting
    is honoured, but that only sets the default: ``search_objects`` still lists
    every schema and agents then name ``sf1`` in their SQL. So dbhub also logs
    in as ``bench_sf<N>``, which can read only its own schema. Create those
    logins with ``scripts/track_b/dbhub_scoped_roles.sql``.

    Args:
        scale_factor: TPC-H scale factor selecting the ``sf<N>`` schema.
        directory: Existing directory to write the config into.

    Returns:
        Path of the written TOML file.
    """
    config_path = directory / f"dbhub_postgres_sf{scale_factor}.toml"
    dsn = (
        f"postgresql://{_PG_USER}_sf{scale_factor}:{_PG_PASSWORD}"
        f"@{_PG_HOST}:{_PG_PORT}/{_PG_DATABASE}?sslmode=disable"
    )
    config_path.write_text(
        "[[sources]]\n"
        'id = "default"\n'
        f'dsn = "{dsn}"\n'
        f'search_path = "sf{scale_factor}"\n'
    )
    return config_path


def _build_dbhub_launch(*, backend: str, scale_factor: int) -> McpLaunchSpec:
    """Assemble the launch spec for the ``dbhub`` upstream."""
    if backend == "duckdb":
        raise ValueError("dbhub does not support the duckdb backend")
    if backend == "postgres":
        url = postgres_native_url(scale_factor=scale_factor)
    elif backend == "mysql":
        url = mysql_native_url(scale_factor=scale_factor)
    else:
        raise ValueError(f"unsupported backend for dbhub: {backend!r}")
    binary = _resolve_node_binary(_DBHUB_BINARY_NAME)
    source_arg = f"--dsn={url}"
    if backend == "postgres" and os.environ.get(_DBHUB_SEARCH_PATH_ENV) == "1":
        config_path = write_dbhub_postgres_config(
            scale_factor=scale_factor, directory=Path(tempfile.gettempdir()),
        )
        source_arg = f"--config={config_path}"
    return McpLaunchSpec(
        binary_path=binary,
        extra_args=("--transport=stdio", source_arg),
        env_extra={}, path_prepend=str(Path(binary).parent), db_url=url,
    )


def build_launch_spec_for(
    *, server: str, backend: str, scale_factor: int,
) -> McpLaunchSpec:
    """Return an :class:`McpLaunchSpec` for one (server, backend, sf) cell.

    Args:
        server: Upstream MCP server name. One of :data:`SUPPORTED_SERVERS`.
        backend: Database backend: ``postgres`` / ``mysql`` / ``duckdb``.
            Server-specific restrictions apply (anthropic-pg is PG-only;
            dbhub does not support DuckDB).
        scale_factor: TPC-H scale factor (routes to the right schema /
            database / DuckDB file).

    Raises:
        ValueError: On unsupported server, unsupported (server, backend)
            pair, or missing DuckDB shim.
        FileNotFoundError: When the upstream binary is not present.
    """
    if server == "mcp-alchemy":
        return _build_alchemy_launch(backend=backend, scale_factor=scale_factor)
    if server == "anthropic-pg":
        return _build_anthropic_pg_launch(
            backend=backend, scale_factor=scale_factor,
        )
    if server == "dbhub":
        return _build_dbhub_launch(backend=backend, scale_factor=scale_factor)
    raise ValueError(
        f"unsupported server: {server!r}; expected one of {SUPPORTED_SERVERS!r}"
    )


def load_gold(gold_dir: str, query_id: str, backend: str) -> dict[str, Any]:
    """Load a captured gold result for ``(query_id, backend)`` from disk."""
    import json
    path = Path(gold_dir) / f"{query_id}__{backend}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"gold result missing at {path} — run capture_gold_results.py first"
        )
    with path.open() as fh:
        return json.load(fh)


__all__ = [
    "McpLaunchSpec", "SUPPORTED_SERVERS", "build_launch_spec_for",
    "build_stdio_client", "duckdb_url", "load_gold",
    "mysql_native_url", "mysql_sqlalchemy_url",
    "postgres_native_url", "postgres_sqlalchemy_url",
    "resolve_binary_path",
]
