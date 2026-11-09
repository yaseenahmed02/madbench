#!/usr/bin/env python
"""Runtime shim to launch mcp-alchemy with a duckdb_engine compat patch.

Why this exists
---------------
The mcp-alchemy binary bundled through uv-tool (and the copy in our .venv)
imports ``duckdb_engine`` 0.17, whose ``ConnectionWrapper.__getattr__``
forwards ``set_isolation_level`` to the underlying ``duckdb.DuckDBPyConnection``.
Recent duckdb (>=1.0) removed that method, so SQLAlchemy's default connection
setup raises::

    AttributeError: '_duckdb.DuckDBPyConnection' object has no attribute
    'set_isolation_level'

This shim monkey-patches ``ConnectionWrapper`` to accept ``set_isolation_level``
as a no-op before importing ``mcp_alchemy``. It then hands control to
mcp-alchemy's ``main`` so the subprocess-stdio contract is preserved.

Use as a drop-in replacement for the mcp-alchemy binary:
    DB_URL=duckdb:////tmp/foo.duckdb python scripts/track_b/mcp_alchemy_shim.py
"""

from __future__ import annotations



def _apply_duckdb_engine_compat() -> None:
    """Add a no-op ``set_isolation_level`` to duckdb_engine's ConnectionWrapper.

    Safe to call multiple times; idempotent.
    """
    try:
        import duckdb_engine
    except ImportError:
        # No duckdb_engine present — nothing to patch.
        return

    wrapper_cls = getattr(duckdb_engine, "ConnectionWrapper", None)
    if wrapper_cls is None:
        return
    if getattr(wrapper_cls, "_track_b_isolation_noop_installed", False):
        return

    def _noop_set_isolation_level(self, level):  # type: ignore[no-untyped-def]
        """DuckDB does not support SQLAlchemy isolation levels; noop for compat."""
        return None

    wrapper_cls.set_isolation_level = _noop_set_isolation_level
    wrapper_cls._track_b_isolation_noop_installed = True


def main() -> int:
    """Apply the compat patch, then delegate to mcp_alchemy's own entry point."""
    _apply_duckdb_engine_compat()
    from mcp_alchemy.server import main as _mcp_alchemy_main
    _mcp_alchemy_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
