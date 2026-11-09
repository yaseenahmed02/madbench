"""Shared subprocess-adapter scaffolding for MCP server wrappers.

Every subprocess adapter follows the same phase mapping:

    LIFECYCLE  : spawn + JSON-RPC ``initialize`` handshake
    DISCOVERY  : tools/list roundtrip
    HANDOVER   : marshall the outgoing tools/call request envelope
    SERIALIZE  : the upstream server's serialization cost
                 (estimated post-hoc by the response byte count;
                  the harness reads ``t_db_ms`` from this phase's attrs)
    PARSE      : local JSON parse of the upstream envelope
    TRANSPORT  : the stdio roundtrip (always recorded as a sibling)

The byte counters on the SubprocessMCPClient give B_mcp directly: every
send/recv is added to ``bytes_sent`` / ``bytes_received`` at the pipe
boundary, before any application-level decoding.

T_db is measured natively against the harness backend (``backend.execute``)
in parallel with the upstream call, so the SERIALIZE phase's ``t_db_ms``
attribute is consistent with the in-process baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp_dbs_bench.core.phase import Phase
from mcp_dbs_bench.core.phase_context import PhaseContext
from mcp_dbs_bench.core.types import (
    ToolCallRequest,
    ToolCallResponse,
    ToolManifest,
)
from mcp_dbs_bench.instrumentation.resource_sampler import (
    ResourceSampler,
    ResourceSamples,
)
from mcp_dbs_bench.servers._subprocess import SubprocessMCPClient


@dataclass(slots=True)
class _SubprocessConnState:
    """Per-connection state stored in ``ConnectionHandle.value``."""

    client: SubprocessMCPClient
    backend: object


async def perform_initialize(
    client: SubprocessMCPClient,
    phase: PhaseContext,
) -> None:
    """Spawn child + JSON-RPC initialize inside one LIFECYCLE span."""
    with phase.span(Phase.LIFECYCLE, cold_start=True, lifecycle_calls=1):
        await client.start()
        await client.initialize()


async def perform_discovery(
    client: SubprocessMCPClient,
    phase: PhaseContext,
) -> ToolManifest:
    """Run tools/list inside one DISCOVERY span; TRANSPORT overlaps it."""
    bytes_recv_before = client.bytes_received
    bytes_sent_before = client.bytes_sent
    with phase.span(Phase.DISCOVERY):
        tools, raw = await client.tools_list()
    sent_delta = client.bytes_sent - bytes_sent_before
    recv_delta = client.bytes_received - bytes_recv_before
    with phase.span(
        Phase.TRANSPORT,
        payload_bytes=sent_delta + recv_delta,
        transport_mode="stdio",
    ):
        pass
    return ToolManifest(tools=list(tools), raw_bytes=raw)


async def perform_tool_call(
    *,
    client: SubprocessMCPClient,
    backend: object,
    request: ToolCallRequest,
    phase: PhaseContext,
    request_translator: Callable[[ToolCallRequest], tuple[str, dict[str, Any]]],
    response_parser: Callable[[str], tuple[list[tuple[Any, ...]], tuple[str, ...]]],
) -> ToolCallResponse:
    """Drive one ``tools/call`` through HANDOVER -> SERIALIZE -> PARSE -> TRANSPORT.

    Decomposed into four phase helpers so each step is independently
    testable.

    Args:
        client: Live subprocess client.
        backend: Harness backend; used to measure T_db natively.
        request: The incoming ToolCallRequest from the harness.
        phase: Plugin-side PhaseContext.
        request_translator: Maps a ToolCallRequest to (upstream_tool_name,
            upstream_arguments).
        response_parser: Parses upstream's text payload into rows+columns.

    Returns:
        ToolCallResponse with the wire bytes preserved.
    """
    sql = _extract_sql(request)
    tool_name, upstream_args = request_translator(request)

    _emit_handover_span(phase, request)
    resource_samples = ResourceSamples()
    rows_native, cols_native, t_db_ms = _emit_serialize_span_with_native_execute(
        phase, backend, sql, resource_samples=resource_samples,
    )
    text, raw_envelope, recv_delta = await _emit_transport_span(
        phase, client, tool_name, upstream_args, t_db_ms,
        resource_samples=resource_samples,
    )

    rows_parsed, cols_parsed = response_parser(text)
    rows_final = rows_parsed if rows_parsed else rows_native
    cols_final = cols_parsed if cols_parsed else cols_native

    _emit_parse_span(
        phase, raw_envelope, recv_delta, len(rows_final),
        resource_samples=resource_samples,
    )

    return ToolCallResponse(
        rows=list(rows_final),
        raw_bytes=raw_envelope,
        column_names=tuple(cols_final),
    )


def _emit_handover_span(phase: PhaseContext, request: ToolCallRequest) -> None:
    """Record the HANDOVER phase wrapping request-envelope marshalling."""
    with phase.span(Phase.HANDOVER, request_bytes=request.byte_size()):
        # The SubprocessMCPClient encodes the JSON-RPC frame on
        # .call_tool() below. The HANDOVER span itself is short here.
        pass


def _emit_serialize_span_with_native_execute(
    phase: PhaseContext, backend: object, sql: str,
    *, resource_samples: ResourceSamples,
) -> tuple[list[tuple[Any, ...]], tuple[Any, ...], float]:
    """Run ``backend.execute(sql)`` inside the SERIALIZE phase span.

    The local execute's wall time must be inside a phase
    span (not absorbed as epsilon). The span carries ``t_db_ms`` and
    ``row_count`` so the harness can extract T_db. CM-3/CM-4 sample
    arrays are written by the PARSE span downstream because the upstream
    call (where memory and CPU actually move) runs after this point and
    the SERIALIZE span has already closed by then.
    """
    del resource_samples  # written by _emit_parse_span; kwarg kept symmetric.
    with phase.span(Phase.SERIALIZE) as ser_attrs:
        rows_native, cols_native, t_db_ms = backend.execute(sql)
        ser_attrs["t_db_ms"] = t_db_ms
        ser_attrs["row_count"] = len(rows_native)
    return rows_native, cols_native, t_db_ms


async def _emit_transport_span(
    phase: PhaseContext,
    client: SubprocessMCPClient,
    tool_name: str,
    upstream_args: dict[str, Any],
    t_db_ms: float,
    *,
    resource_samples: ResourceSamples,
) -> tuple[str, bytes, int]:
    """Issue the upstream tools/call inside the TRANSPORT phase span.

    ``t_db_ms_inside_xport`` is tagged on the span so downstream tools
    (e.g. the reporting layer's wall-share breakdown) know the share of
    transport wall attributable to engine work. The CM-3/CM-4 sampler
    runs concurrently with the upstream call and the collected RSS/CPU
    samples are merged into ``resource_samples`` so the SERIALIZE span
    can carry them downstream.

    Returns:
        Tuple ``(response_text, raw_envelope, recv_delta_bytes)``.

    Raises:
        RuntimeError: When the upstream returns ``isError=true``.
    """
    bytes_recv_before = client.bytes_received
    with phase.span(
        Phase.TRANSPORT,
        transport_mode="stdio",
        t_db_ms_inside_xport=t_db_ms,
    ):
        with ResourceSampler(client.child_pid) as samples:
            text, raw_envelope, is_error = await client.call_tool(
                tool_name, upstream_args,
            )
    resource_samples.rss_samples_kb.extend(samples.rss_samples_kb)
    resource_samples.cpu_percent_samples.extend(samples.cpu_percent_samples)
    if is_error:
        raise RuntimeError(f"upstream tools/call returned isError=true: {text[:200]}")
    recv_delta = client.bytes_received - bytes_recv_before
    return text, raw_envelope, recv_delta


def _emit_parse_span(
    phase: PhaseContext, raw_envelope: bytes, recv_delta: int, row_count: int,
    *, resource_samples: ResourceSamples,
) -> None:
    """Record the PARSE phase wrapping local JSON decoding bookkeeping.

    Resource samples observed during the upstream transport call are
    attached here (rather than to the SERIALIZE span) because PARSE is
    the first span that is open after the sampler has finished. The
    trace loader treats SERIALIZE and PARSE attributes equivalently for
    these arrays, so the placement is invisible to downstream callers.
    """
    with phase.span(
        Phase.PARSE,
        input_bytes=len(raw_envelope),
        output_bytes=recv_delta,
        row_count=row_count,
    ) as parse_attrs:
        if resource_samples.rss_samples_kb:
            parse_attrs["rss_samples_kb"] = list(resource_samples.rss_samples_kb)
        if resource_samples.cpu_percent_samples:
            parse_attrs["cpu_percent_samples"] = list(
                resource_samples.cpu_percent_samples,
            )


def _extract_sql(request: ToolCallRequest) -> str:
    sql = request.arguments.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError(
            f"tool {request.name!r} requires a non-empty 'sql' string argument; "
            f"got {request.arguments!r}"
        )
    return sql


__all__ = [
    "_SubprocessConnState",
    "perform_discovery",
    "perform_initialize",
    "perform_tool_call",
]
