"""Cross-environment binary discovery for MCP server subprocesses.

shutil.which() only searches PATH. Two MCP servers (dbhub, mcp-server-postgres)
are typically installed via nvm-managed npm. When the harness is launched from
a shell that hasn't sourced nvm (e.g., a vanilla nohup or systemd unit), nvm's
shell function isn't loaded and its bin dir isn't on PATH — so shutil.which()
returns None and subprocess.Popen() fails with FileNotFoundError.

This module adds known install-location probing so plugins can find their
binary regardless of how the parent shell was set up. Search order:

  1. Explicit ``binary`` override (caller-supplied path).
  2. ``shutil.which(name)`` (current PATH).
  3. nvm-managed Node bin dirs under ``$HOME/.nvm/versions/node/*/bin/``.
  4. ``$HOME/.local/bin/``.
  5. ``/usr/local/bin/``.
  6. Bare name (lets subprocess.Popen do its own resolution; will fail loudly
     if absent, which is the desired behavior).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def find_binary(name: str, *, explicit: str | None = None) -> str:
    """Resolve a binary name to an executable path.

    Args:
        name: Bare command name (e.g., ``"mcp-server-postgres"``).
        explicit: Optional caller-supplied path; bypasses search when set.

    Returns:
        Absolute path to the executable, or the bare ``name`` if not found
        (so subprocess.Popen raises a clear FileNotFoundError downstream).
    """
    if explicit:
        return explicit

    # Prefer the copy installed in the running environment, so the pinned
    # server version wins over an unrelated global install on PATH.
    in_env = Path(sys.executable).parent / name
    if in_env.is_file() and os.access(in_env, os.X_OK):
        return str(in_env)

    via_path = shutil.which(name)
    if via_path:
        return via_path

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
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)

    return name
