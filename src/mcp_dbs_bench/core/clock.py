"""Monotonic-clock helpers.

The benchmark uses `time.monotonic_ns()` exclusively.
`time.time_ns()` is wall-clock, can jump backwards during NTP sync, and would
silently corrupt span-duration measurements; using it anywhere is a bug.
"""

from __future__ import annotations

import time

NS_PER_MS: int = 1_000_000
NS_PER_S: int = 1_000_000_000


def monotonic_now_ns() -> int:
    """Return the current monotonic time in nanoseconds.

    Returns:
        Nanoseconds since an unspecified but monotonic reference point.
        Suitable for measuring elapsed durations only; not a wall-clock value.
    """
    return time.monotonic_ns()


def ns_to_ms(duration_ns: int) -> float:
    """Convert a non-negative nanosecond duration to milliseconds.

    Args:
        duration_ns: Elapsed time in nanoseconds. Must be >= 0.

    Returns:
        Equivalent duration in milliseconds as a float.

    Raises:
        ValueError: If ``duration_ns`` is negative.
    """
    if duration_ns < 0:
        raise ValueError(f"duration_ns must be non-negative; got {duration_ns}")
    return duration_ns / NS_PER_MS
