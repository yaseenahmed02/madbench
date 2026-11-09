"""Instrumentation layer: OTel tracer and property verification.

The harness configures a tracer once at startup and passes it down to the
PhaseContext implementation. The tracer is :class:`OTelTracer`: the OTel SDK with a JSON Lines exporter.
"""

from __future__ import annotations
