"""CM-3: Peak Memory M_peak.

Peak RSS across the mcp.interaction. Sampling happens in a sidecar; this
function just summarizes the samples.
"""

from __future__ import annotations

from typing import Iterable

_KB_PER_MB = 1024.0


def cm3_peak_memory_mb(rss_samples_kb: Iterable[int]) -> float:
    """Return the peak RSS observed during the interaction, in megabytes.

    Args:
        rss_samples_kb: RSS samples (kilobytes) collected at 10 ms intervals
            during the mcp.interaction. Empty iterable returns 0.

    Returns:
        Peak memory in MB (kilobytes / 1024).
    """
    samples = list(rss_samples_kb)
    if not samples:
        return 0.0
    return max(samples) / _KB_PER_MB
