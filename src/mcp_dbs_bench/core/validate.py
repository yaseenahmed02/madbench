"""Plugin validator: conformance check for MCP server plugins.

Third-party plugins get a documented validate path from the SDK. ``validate_plugin`` runs four checks against a
plugin class:

    1. Protocol conformance      - ``isinstance(plugin, MCPServerPlugin)``
    2. Capability self-consistency - ``capabilities()`` returns a usable
                                     :class:`ServerCapabilities`
    3. Required attributes        - ``name`` and ``version`` are non-empty
    4. Construction does not raise

The result is a :class:`ValidationReport` with ``ok: bool`` and a list of
human-readable issue strings. A clean report (``ok=True``, ``issues=[]``)
is what third-party plugins are expected to produce; any failure becomes
a ``PluginRegistrationError`` when the harness later attempts to load it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mcp_dbs_bench.core.plugin import MCPServerPlugin
from mcp_dbs_bench.core.types import ServerCapabilities


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of :func:`validate_plugin`.

    Attributes:
        ok: True iff all checks passed.
        plugin_name: The plugin's ``name`` attribute, or empty string if
            the plugin failed to construct.
        issues: Human-readable diagnostic strings. Empty when ``ok``.
    """

    ok: bool
    plugin_name: str
    issues: list[str] = field(default_factory=list)


def validate_plugin(plugin_cls: type) -> ValidationReport:
    """Run all conformance checks against ``plugin_cls`` and return a report.

    The validator never raises; failures are returned as ``issues`` in
    the report so the caller can decide whether to abort or merely warn.

    Args:
        plugin_cls: A class that purports to implement ``MCPServerPlugin``.

    Returns:
        :class:`ValidationReport` summarising every check's outcome.
    """
    issues: list[str] = []
    instance = _instantiate_or_record(plugin_cls, issues)
    name = _validate_name_and_version(instance, issues) if instance is not None else ""

    if instance is not None:
        _validate_protocol_conformance(instance, issues)
        _validate_capabilities(instance, issues)

    return ValidationReport(ok=not issues, plugin_name=name, issues=issues)


def _instantiate_or_record(plugin_cls: type, issues: list[str]) -> object | None:
    """Try to construct an instance; record the failure as an issue."""
    try:
        return plugin_cls()
    except (TypeError, ValueError, RuntimeError, ImportError) as exc:
        issues.append(
            f"construction failed: {type(exc).__name__}: {exc}. "
            f"Plugins must support zero-arg construction."
        )
        return None


def _validate_name_and_version(instance: object, issues: list[str]) -> str:
    """Verify ``name`` and ``version`` are non-empty strings."""
    name = getattr(instance, "name", "")
    version = getattr(instance, "version", "")
    if not isinstance(name, str) or not name.strip():
        issues.append("attribute 'name' must be a non-empty string")
        name = ""
    if not isinstance(version, str) or not version.strip():
        issues.append("attribute 'version' must be a non-empty string")
    return name


def _validate_protocol_conformance(instance: object, issues: list[str]) -> None:
    """Run ``isinstance(instance, MCPServerPlugin)`` and record the result."""
    if not isinstance(instance, MCPServerPlugin):
        issues.append(
            "instance does not implement MCPServerPlugin "
            "(missing one of: name, version, capabilities, connect, "
            "discover, call_tool)"
        )


def _validate_capabilities(instance: object, issues: list[str]) -> None:
    """Call ``capabilities()`` and verify a valid object is returned."""
    capabilities_fn = getattr(instance, "capabilities", None)
    if capabilities_fn is None:
        issues.append("missing capabilities() method")
        return
    try:
        caps = capabilities_fn()
    except (TypeError, ValueError, RuntimeError) as exc:
        issues.append(
            f"capabilities() raised {type(exc).__name__}: {exc}"
        )
        return
    if not isinstance(caps, ServerCapabilities):
        issues.append(
            f"capabilities() must return ServerCapabilities; got {type(caps).__name__}"
        )
