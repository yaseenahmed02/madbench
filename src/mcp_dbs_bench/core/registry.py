"""Plugin discovery via Python entry-points.

Built-in MCP server adapters and DBMS backends register through the same
entry-point groups (`mcp_dbs_bench.servers`, `mcp_dbs_bench.backends`) that
third-party plugins use. This forces the SDK to be expressive enough to
implement every shipped adapter.

The registry also exposes an in-process override table so that the CLI and
tests can register plugins without an install step. Overrides take
precedence over entry-points if both exist.
"""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from mcp_dbs_bench.core.exceptions import PluginRegistrationError
from mcp_dbs_bench.core.plugin import MCPServerPlugin

_LOG = logging.getLogger(__name__)

_SERVER_GROUP = "mcp_dbs_bench.servers"
_BACKEND_GROUP = "mcp_dbs_bench.backends"

_server_overrides: dict[str, type[Any]] = {}
_backend_overrides: dict[str, type[Any]] = {}


def register_server(name: str, plugin_cls: type[Any]) -> None:
    """Register an MCP server plugin under ``name``, bypassing entry-points.

    Used by tests, by the CLI's ``--plugin-module`` flag, and by built-in
    adapters during process startup. Overrides take precedence over
    entry-points.
    """
    _server_overrides[name] = plugin_cls


def register_backend(name: str, backend_cls: type[Any]) -> None:
    """Register a DBMS backend under ``name`` (see ``register_server``)."""
    _backend_overrides[name] = backend_cls


def available_servers() -> list[str]:
    """Return all known server plugin names (overrides + entry-points)."""
    discovered = {ep.name for ep in _iter_entry_points(_SERVER_GROUP)}
    return sorted(discovered | set(_server_overrides))


def available_backends() -> list[str]:
    """Return all known backend names (overrides + entry-points)."""
    discovered = {ep.name for ep in _iter_entry_points(_BACKEND_GROUP)}
    return sorted(discovered | set(_backend_overrides))


# Narrow exception sets at each registry boundary. Construction may fail
# on missing drivers / config; entry-point load may fail on import errors;
# entry_points(group=...) may raise on packaging-metadata corruption.
_INSTANTIATION_ERRORS: tuple[type[BaseException], ...] = (
    TypeError, ValueError, RuntimeError, ImportError,
    ModuleNotFoundError, AttributeError, OSError,
)
_ENTRY_POINT_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    ImportError, ModuleNotFoundError, AttributeError, ValueError,
)
_METADATA_QUERY_ERRORS: tuple[type[BaseException], ...] = (
    ValueError, OSError, ImportError, ModuleNotFoundError,
)


def load_server(name: str) -> MCPServerPlugin:
    """Instantiate the named MCP server plugin.

    Raises:
        PluginRegistrationError: If the name is unknown, the entry point
            fails to load, or the loaded object does not implement
            ``MCPServerPlugin``.
    """
    cls = _resolve(name, _server_overrides, _SERVER_GROUP, "server")
    try:
        instance = cls()
    except _INSTANTIATION_ERRORS as exc:
        raise PluginRegistrationError(
            f"server plugin '{name}' raised {type(exc).__name__} on instantiation: {exc}"
        ) from exc
    if not isinstance(instance, MCPServerPlugin):
        raise PluginRegistrationError(
            f"server plugin '{name}' does not implement MCPServerPlugin "
            "(missing attribute or method). See core.plugin.MCPServerPlugin."
        )
    return instance


def load_backend(name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate the named DBMS backend, forwarding constructor arguments.

    Raises:
        PluginRegistrationError: If the name is unknown or the entry point
            fails to load.
    """
    cls = _resolve(name, _backend_overrides, _BACKEND_GROUP, "backend")
    try:
        return cls(*args, **kwargs)
    except _INSTANTIATION_ERRORS as exc:
        raise PluginRegistrationError(
            f"backend '{name}' raised {type(exc).__name__} on instantiation: {exc}"
        ) from exc


def _resolve(
    name: str, overrides: dict[str, type[Any]], group: str, kind: str
) -> type[Any]:
    if name in overrides:
        return overrides[name]
    for ep in _iter_entry_points(group):
        if ep.name == name:
            try:
                return ep.load()
            except _ENTRY_POINT_LOAD_ERRORS as exc:
                raise PluginRegistrationError(
                    f"{kind} plugin '{name}' failed to load: {type(exc).__name__}: {exc}"
                ) from exc
    available = sorted({ep.name for ep in _iter_entry_points(group)} | set(overrides))
    raise PluginRegistrationError(
        f"unknown {kind} '{name}'. Available: {available}"
    )


def _iter_entry_points(group: str) -> list[EntryPoint]:
    """Return entry points for ``group``. Wrapped so tests can monkeypatch."""
    try:
        return list(entry_points(group=group))
    except _METADATA_QUERY_ERRORS as exc:
        _LOG.warning("entry_points(group=%s) raised %s", group, exc)
        return []
