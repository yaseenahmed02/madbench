"""Paired-cache Postgres restart for the sweep runner.

For cold-start reps we restart the PG container so its buffer cache is
empty before each attempt. MySQL is not restarted (a MySQL container
restart takes 30-60 s). DuckDB is process-owned by
the mcp-alchemy child, so no external restart is needed.
"""

from __future__ import annotations

import logging
import subprocess
import time

_LOG = logging.getLogger(__name__)

_PG_CONTAINER: str = "madbench-pg"
_HEALTHCHECK_TIMEOUT_S: float = 60.0
_HEALTHCHECK_POLL_S: float = 1.0
_DOCKER_TIMEOUT_S: float = 120.0


def restart_postgres_wait_healthy() -> None:
    """Restart the PG container and block until it reports healthy.

    Raises:
        RuntimeError: If Docker refuses the restart or the container does
            not become healthy inside the timeout window.
    """
    _LOG.info("restarting %s (paired cache)", _PG_CONTAINER)
    try:
        subprocess.run(
            ["docker", "restart", _PG_CONTAINER],
            check=True, timeout=_DOCKER_TIMEOUT_S, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"docker restart failed: {exc.stderr.decode(errors='replace')}"
        ) from exc
    _wait_until_healthy()


def _wait_until_healthy() -> None:
    """Poll ``docker inspect`` until the health status is ``healthy``."""
    deadline = time.monotonic() + _HEALTHCHECK_TIMEOUT_S
    last_status = "unknown"
    while time.monotonic() < deadline:
        status = _health_status()
        last_status = status
        if status == "healthy":
            _LOG.info("%s healthy", _PG_CONTAINER)
            return
        time.sleep(_HEALTHCHECK_POLL_S)
    raise RuntimeError(
        f"{_PG_CONTAINER} did not become healthy in "
        f"{_HEALTHCHECK_TIMEOUT_S}s (last status={last_status!r})"
    )


def _health_status() -> str:
    """Return the container's health status per ``docker inspect``."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Health.Status}}", _PG_CONTAINER],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return f"error:{result.stderr.strip()}"
    return result.stdout.strip() or "no-status"


__all__ = ["restart_postgres_wait_healthy"]
