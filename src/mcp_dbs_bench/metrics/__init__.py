"""Metric computation: project validated spans into the benchmark metrics.

Each metric implementation is a pure function from one or more SpanRecord
objects (and possibly external scalars) to a numeric or classification result.
No I/O, no time, no random — entirely deterministic.

Sub-packages:
  coupling/        Coupling metrics CM-3, CM-4, CM-6
"""

from __future__ import annotations

from mcp_dbs_bench.metrics.span_record import SpanRecord

__all__ = ["SpanRecord"]
