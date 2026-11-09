"""CM-4: CPU Saturation Threshold sigma*_CPU.

The smallest scale factor sigma at which serialization CPU% reaches a
saturation threshold (default 90% of one core).
"""

from __future__ import annotations


def cm4_cpu_saturation_threshold(
    sf_to_serialization_cpu_pct: dict[int, float],
    threshold_pct: float = 90.0,
) -> int | None:
    """Return the smallest sigma at which serialization CPU saturates.

    Args:
        sf_to_serialization_cpu_pct: Map from TPC-H scale factor to observed
            serialization CPU% during T_ser.
        threshold_pct: Saturation threshold in CPU% of one core.

    Returns:
        The smallest sigma whose CPU% exceeds the threshold, or None if no
        measured sigma saturates. The metric does not extrapolate beyond
        measured sigmas.
    """
    for sf in sorted(sf_to_serialization_cpu_pct.keys()):
        if sf_to_serialization_cpu_pct[sf] >= threshold_pct:
            return sf
    return None
