"""CM-6: Back-Pressure Classification beta.

Classifies how a server behaves under large result sets: OOM, chunked stream,
truncation, or error overflow. The conservative default for unknown
stream_mode is BUFFER_OOM.
"""

from __future__ import annotations

from enum import Enum

_MEMORY_PRESSURE_FRACTION = 0.95


class BackpressurePattern(str, Enum):
    """The four documented back-pressure behaviors."""

    BUFFER_OOM = "buffer_oom"
    CHUNKED_STREAM = "chunked_stream"
    TRUNCATE = "truncate"
    ERROR_OVERFLOW = "error_overflow"


def cm6_backpressure_classification(
    stream_mode: bool | None,
    requested_rows: int,
    returned_rows: int,
    interaction_status: str,
    peak_memory_mb: float,
    memory_limit_mb: float,
) -> BackpressurePattern:
    """Classify a server's back-pressure behavior under load.

    Args:
        stream_mode: True if the response was chunked-streamed; False if
            buffered; None if the server did not declare.
        requested_rows: Rows requested (or upper bound).
        returned_rows: Rows actually returned to the caller.
        interaction_status: "OK" | "OOM" | "ERROR".
        peak_memory_mb: Observed peak RSS (CM-3).
        memory_limit_mb: Container memory limit at the time of measurement.

    Returns:
        The matching :class:`BackpressurePattern`.
    """
    if interaction_status == "OOM" or peak_memory_mb >= memory_limit_mb * _MEMORY_PRESSURE_FRACTION:
        return BackpressurePattern.BUFFER_OOM
    if interaction_status == "ERROR" and requested_rows > returned_rows:
        return BackpressurePattern.ERROR_OVERFLOW
    if returned_rows < requested_rows:
        return BackpressurePattern.TRUNCATE
    if stream_mode is True:
        return BackpressurePattern.CHUNKED_STREAM
    return BackpressurePattern.BUFFER_OOM
