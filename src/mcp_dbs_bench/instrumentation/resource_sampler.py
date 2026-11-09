"""Background RSS + CPU% sampler for MCP child subprocesses.

CM-3 (M_peak) and CM-4 (sigma*_CPU) require per-interaction memory and CPU
samples from the upstream MCP server's process. The harness can't infer
either from JSON-RPC frames; the sampler attaches a daemon thread that
polls ``psutil`` while a call is in flight, then exposes the collected
samples for storage on a span.

The sampler is intentionally lazy about psutil: if the dependency or the
target process is unavailable, the context manager degrades to a no-op
that yields empty sample lists. The downstream loader treats missing
samples as "data unavailable" rather than fabricating zeros, so a
psutil-less run still produces valid (but reduced) CSVs.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

try:  # psutil is optional; absence degrades to no-op sampling.
    import psutil  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only on hosts w/o psutil
    psutil = None  # type: ignore[assignment]

_DEFAULT_INTERVAL_S = 0.05
_BYTES_PER_KB = 1024


@dataclass(slots=True)
class ResourceSamples:
    """Collected samples from one sampling window."""

    rss_samples_kb: list[int] = field(default_factory=list)
    cpu_percent_samples: list[float] = field(default_factory=list)


class ResourceSampler:
    """Daemon-thread sampler driven as a context manager.

    Usage::

        with ResourceSampler(child_pid) as samples:
            await call_upstream(...)
        span_attrs["rss_samples_kb"] = samples.rss_samples_kb
        span_attrs["cpu_percent_samples"] = samples.cpu_percent_samples

    The sampler never raises out of ``__enter__``/``__exit__``: any
    psutil failure (process gone, permission error) silently leaves the
    sample list empty so the harness keeps running.
    """

    def __init__(
        self,
        pid: int | None,
        *,
        interval_s: float = _DEFAULT_INTERVAL_S,
    ) -> None:
        self._pid = pid
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = ResourceSamples()

    def __enter__(self) -> ResourceSamples:
        if self._pid is None or psutil is None:
            return self._samples
        try:
            proc = psutil.Process(self._pid)
            proc.cpu_percent(interval=None)  # prime the rolling delta
        except psutil.Error:
            return self._samples
        self._thread = threading.Thread(
            target=self._run, args=(proc,), daemon=True,
        )
        self._thread.start()
        return self._samples

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self, proc) -> None:
        while not self._stop.is_set():
            try:
                rss_bytes = proc.memory_info().rss
                cpu_pct = proc.cpu_percent(interval=None)
            except psutil.Error:
                return
            self._samples.rss_samples_kb.append(rss_bytes // _BYTES_PER_KB)
            self._samples.cpu_percent_samples.append(float(cpu_pct))
            if self._stop.wait(self._interval_s):
                return


def iter_resource_intervals(
    pid: int | None, *, interval_s: float = _DEFAULT_INTERVAL_S,
) -> Iterator[ResourceSampler]:
    """One-shot iterator returning a single sampler for ``pid``.

    Provided so callers can write ``for sampler in iter_resource_intervals(pid):``
    without committing to the context-manager idiom; useful in async
    code paths where ``async with`` wrapping is awkward.
    """
    yield ResourceSampler(pid, interval_s=interval_s)


__all__ = ["ResourceSampler", "ResourceSamples", "iter_resource_intervals"]
