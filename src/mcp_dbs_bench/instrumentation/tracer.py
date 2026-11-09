"""OTel tracer wiring with the JSON Lines exporter.

JSONL is the default exporter because it is:
  - inspectable: one self-contained JSON record per line, jq/grep friendly
  - forward-compatible: each line carries a `schema_version` field
  - safe under partial writes: a truncated tail line is discarded by jq

OTel SDK 1.27+ records span timestamps from `time.monotonic_ns()` internally
on Python, so all spans use the monotonic clock without patching.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

_TRACER_NAME = "madbench"
_SCHEMA_VERSION = "0.1.0"


class JsonlSpanExporter(SpanExporter):
    """Default exporter: one JSON object per line.

    The file is opened in TRUNCATE mode (``"w"``). The harness invokes one
    exporter per cell, so each cell's trace.jsonl is owned by exactly one
    process for its lifetime. Truncate mode makes retries idempotent: in
    append mode a re-attempted cell would concatenate its spans onto the
    partial trace and inflate the ``mcp.interaction`` count.
    """

    def __init__(self, output_path: Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", buffering=1)  # line-buffered, truncate
        self._closed = False

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        if self._closed:
            return SpanExportResult.FAILURE
        for span in spans:
            self._fh.write(json.dumps(_serialize_span(span), sort_keys=True))
            self._fh.write("\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        if self._closed:
            return
        self._fh.flush()
        self._fh.close()
        self._closed = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        if not self._closed:
            self._fh.flush()
        return True


def _serialize_span(span: ReadableSpan) -> dict[str, Any]:
    """Convert a ReadableSpan to the JSONL record shape."""
    ctx = span.get_span_context()
    parent_ctx = span.parent
    return {
        "schema_version": _SCHEMA_VERSION,
        "name": span.name,
        "trace_id": _hex(ctx.trace_id, 32),
        "span_id": _hex(ctx.span_id, 16),
        "parent_span_id": (
            _hex(parent_ctx.span_id, 16) if parent_ctx is not None else None
        ),
        "start_ns": int(span.start_time or 0),
        "end_ns": int(span.end_time or 0),
        "attributes": dict(span.attributes or {}),
        "status": span.status.status_code.name if span.status else "UNSET",
    }


def _hex(value: int, width: int) -> str:
    return format(value, f"0{width}x")


class OTelTracer:
    """Thin wrapper that owns a TracerProvider + exporter lifecycle.

    The wrapper is process-scoped — one instance per Track A run
    invocation. The OTel global TracerProvider is intentionally not used,
    so multiple tracers can coexist in the same process (calibration runs
    a traced and untraced arm in alternation).
    """

    def __init__(self, exporter: SpanExporter) -> None:
        resource = Resource.create(
            {"service.name": _TRACER_NAME, "service.version": _SCHEMA_VERSION}
        )
        self._provider = TracerProvider(resource=resource)
        self._exporter = exporter
        # BatchSpanProcessor batches span exports off the hot path, keeping
        # the observer effect under the 5% Property 1 threshold for ms-scale
        # interactions.
        self._provider.add_span_processor(BatchSpanProcessor(exporter))
        self._tracer = self._provider.get_tracer(_TRACER_NAME, _SCHEMA_VERSION)
        self._closed = False

    @classmethod
    def with_jsonl_exporter(cls, output_path: Path) -> OTelTracer:
        """Construct a tracer that exports to a JSON Lines file."""
        return cls(JsonlSpanExporter(output_path))

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Iterator[Any]:
        """Same API as OTel Tracer.start_as_current_span."""
        with self._tracer.start_as_current_span(name, **kwargs) as span:
            yield span

    def start_span(self, name: str, **kwargs: Any) -> Any:
        return self._tracer.start_span(name, **kwargs)

    def shutdown(self) -> None:
        """Flush pending spans and close the exporter. Idempotent."""
        if self._closed:
            return
        self._provider.shutdown()
        self._closed = True
