"""Load JSONL traces and compute the 16 IM/CM metrics per rep.

Each ``trace.jsonl`` file holds one or more ``mcp.interaction`` traces.
For every ``mcp.interaction`` root span we project the descendant spans
into a :class:`TraceMetrics` record carrying all nine timing terms,
derived ratios, byte/token aggregates, and identification keys.

The ninth timing term ``T_teardown`` measures the interval between the
final ``mcp.parse`` span end and the parent ``mcp.interaction`` span end.
Prior to its introduction, that interval was folded into the residual
``epsilon_ms``, which inflated the Property 1 shortfall on servers whose
teardown path (subprocess terminate, buffered pipe flush) contributes a
non-negligible slice of ``T_total``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from mcp_dbs_bench.core.clock import ns_to_ms
from mcp_dbs_bench.core.derived import (
    protocol_tax_ratio,
    serialization_throughput_rows_per_ms,
)
from mcp_dbs_bench.metrics.coupling.peak_memory import cm3_peak_memory_mb
from mcp_dbs_bench.track_a.organize import CellSpec, cell_id

_INTERACTION_NAME = "mcp.interaction"
_PHASE_SPAN_NAMES = {
    "discovery": "mcp.discovery",
    "lifecycle": "mcp.lifecycle.connect",
    "handover": "mcp.tool_call",
    "serialize": "mcp.serialization",
    "parse": "mcp.parse",
    "transport": "mcp.transport",
}
_RETRY_SPAN_NAME = "mcp.retry"
_CONNECTION_ACQUIRE_SPAN_NAME = "connection.acquire"
_CONNECTION_RELEASE_SPAN_NAME = "connection.release"


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    """One per rep. 16 IM/CM fields plus identification keys.

    Attributes:
        cell_id: Canonical cell identifier.
        server / backend / query / scale_factor / rep_idx / cold_warm: keys.
        t_total_ms..t_xport_excl_ms: 8 timing terms (IM-1..IM-7 + epsilon).
        t_teardown_ms: 9th phase capturing the interval from ``mcp.parse``
            end to ``mcp.interaction`` end. 0.0 when no parse span exists.
        epsilon_ms: residual (T_total - sum-of-eight-phases - T_db).
        ptax_ms: protocol-tax in ms.
        ptr: protocol-tax ratio (None if T_db <= 0).
        b_mcp: bytes received over MCP wire (from serialize span).
        tau_disc: discovery token count.
        phi: token inflation factor (None if tau_star_result missing).
        gamma: context decay (None if context_window not provided).
        s_ser_rows_per_ms: serialization throughput.
        rho: payload expansion ratio (None if no B_native baseline).
        n_life: lifecycle handshake count.
        r_retry_ms: IM-7 retry-loop cost (0.0 when no retries).
        t_pool_ms: CM-5 connection handover (None when no handover spans).
        m_peak_mb: CM-3 peak RSS observed during the interaction
            (None when no ``rss_samples_kb`` attribute was recorded).
        cpu_percent_samples: raw CPU% samples observed during the serialize
            phase (empty list when no ``cpu_percent_samples`` attribute was
            recorded). Per-cell σ*_CPU is derived from these at the
            cell-summary level.
        trace_path: source ``trace.jsonl`` path for back-reference.
    """

    cell_id: str
    server: str
    backend: str
    query: str
    scale_factor: int
    rep_idx: int
    cold_warm: str
    # Eight timing terms (ms)
    t_total_ms: float
    t_db_ms: float
    t_disc_ms: float
    t_life_ms: float
    t_hand_ms: float
    t_ser_ms: float
    t_parse_ms: float
    t_xport_excl_ms: float
    t_teardown_ms: float
    epsilon_ms: float
    # Derived
    ptax_ms: float
    ptr: float | None
    # IM/CM counts and rates
    b_mcp: int
    tau_disc: int
    phi: float | None
    gamma: float | None
    s_ser_rows_per_ms: float | None
    rho: float | None
    n_life: int
    # Additional IM/CM (6 of 16 wired here; δ_CW + σ*_CPU live at the
    # cell-summary level since they require cross-cell or cross-SF inputs).
    r_retry_ms: float
    t_pool_ms: float | None
    m_peak_mb: float | None
    cpu_percent_samples: tuple[float, ...]
    # Provenance
    trace_path: str


def load_trace_metrics(
    trace_path: Path,
    *,
    server: str,
    backend: str,
    query: str,
    scale_factor: int,
    cold_warm: str,
    rep_idx_start: int = 0,
) -> Iterator[TraceMetrics]:
    """Yield one :class:`TraceMetrics` per ``mcp.interaction`` span in the file.

    Raises:
        FileNotFoundError: If ``trace_path`` does not exist.
    """
    path = Path(trace_path)
    if not path.is_file():
        raise FileNotFoundError(f"trace file not found: {path}")
    spans = _parse_jsonl(path)
    interactions = [s for s in spans if s["name"] == _INTERACTION_NAME]
    spec = CellSpec(
        server=server, backend=backend, query=query,
        scale_factor=scale_factor, cold_warm=cold_warm,
    )
    cid = cell_id(spec)
    for offset, root in enumerate(interactions):
        # Filter descendant spans by trace_id and time-window membership.
        children = _children_of(spans, root)
        yield _project_to_metrics(
            root=root, children=children,
            cid=cid, spec=spec,
            rep_idx=rep_idx_start + offset,
            trace_path=str(path),
        )


# Allow rep_idx as kwarg too for clarity.
def _project_to_metrics(
    *, root: dict, children: list[dict],
    cid: str, spec: CellSpec, rep_idx: int, trace_path: str,
) -> TraceMetrics:
    t_total_ms = ns_to_ms(root["end_ns"] - root["start_ns"])
    by_phase: dict[str, list[dict]] = {p: [] for p in _PHASE_SPAN_NAMES}
    retry_spans: list[dict] = []
    pool_spans: list[dict] = []
    for span in children:
        name = span["name"]
        for phase_key, span_name in _PHASE_SPAN_NAMES.items():
            if name == span_name:
                by_phase[phase_key].append(span)
        if name == _RETRY_SPAN_NAME:
            retry_spans.append(span)
        elif name in (_CONNECTION_ACQUIRE_SPAN_NAME, _CONNECTION_RELEASE_SPAN_NAME):
            pool_spans.append(span)

    t_disc_ms = _sum_durations_ms(by_phase["discovery"])
    t_life_ms = _sum_durations_ms(by_phase["lifecycle"])
    t_hand_ms = _sum_durations_ms(by_phase["handover"])
    t_ser_wall_ms = _sum_durations_ms(by_phase["serialize"])
    t_parse_ms = _sum_durations_ms(by_phase["parse"])
    # Exclusive transport = wall - overlap(handover) - overlap(parse).
    t_xport_excl_ms = _exclusive_transport_ms(
        transport_spans=by_phase["transport"],
        handover_spans=by_phase["handover"],
        parse_spans=by_phase["parse"],
    )
    # The plugin records its backend.execute call inside the SERIALIZE span,
    # so wall(SERIALIZE) includes t_db. Subtract to avoid double-counting
    # (matches core/timing.py decompose()).
    t_db_ms = _extract_t_db_ms(by_phase["serialize"])
    t_ser_ms = max(0.0, t_ser_wall_ms - t_db_ms)
    # The HANDOVER span wraps the SERIALIZE + PARSE + TRANSPORT subtree;
    # exclusive HANDOVER time is wall(HANDOVER) - (ser + parse + xport_excl + t_db)
    # so we don't double-count nested durations.
    t_hand_excl_ms = max(
        0.0,
        t_hand_ms - (t_ser_ms + t_parse_ms + t_xport_excl_ms + t_db_ms),
    )
    t_teardown_ms = _teardown_ms(root=root, parse_spans=by_phase["parse"])
    epsilon_ms = t_total_ms - (
        t_db_ms + t_disc_ms + t_life_ms + t_hand_excl_ms
        + t_ser_ms + t_parse_ms + t_xport_excl_ms + t_teardown_ms
    )
    t_hand_ms = t_hand_excl_ms
    # Protocol Tax is everything outside database execution, so teardown and
    # the residual count as tax: PTax = T_total - T_db.
    ptax = t_total_ms - t_db_ms
    ptr_value = (
        protocol_tax_ratio(ptax_ms=ptax, t_db_ms=t_db_ms) if t_db_ms > 0 else None
    )
    ser_attrs = _merge_attrs(by_phase["serialize"])
    disc_attrs = _merge_attrs(by_phase["discovery"])
    b_mcp = int(ser_attrs.get("B_mcp", 0))
    tau_disc = int(disc_attrs.get("tokens", disc_attrs.get("tau_disc", 0)))
    rows = int(ser_attrs.get("rows", 0))
    phi = _safe_phi(ser_attrs, disc_attrs)
    gamma = _safe_gamma(ser_attrs, disc_attrs)
    s_ser = (
        serialization_throughput_rows_per_ms(row_count=rows, t_ser_ms=t_ser_ms)
        if rows > 0 and t_ser_ms > 0 else None
    )
    b_native = int(ser_attrs.get("B_native", 0))
    rho = (b_mcp / b_native) if b_native > 0 else None
    r_retry_ms = _sum_durations_ms(retry_spans)
    t_pool_ms = _sum_durations_ms(pool_spans) if pool_spans else None
    # Resource samples may be attached to SERIALIZE (for inprocess adapters
    # where the sampler runs during execute) or to PARSE (for subprocess
    # adapters where the sampler closes after SERIALIZE has ended). The
    # interaction-root attribute path covers servers that sample globally.
    sample_carriers = by_phase["serialize"] + by_phase["parse"]
    m_peak_mb = _peak_memory_from_spans(sample_carriers, root)
    cpu_samples = _cpu_samples_from_spans(sample_carriers)
    return TraceMetrics(
        cell_id=cid, server=spec.server, backend=spec.backend,
        query=spec.query, scale_factor=spec.scale_factor,
        rep_idx=rep_idx, cold_warm=spec.cold_warm,
        t_total_ms=t_total_ms, t_db_ms=t_db_ms,
        t_disc_ms=t_disc_ms, t_life_ms=t_life_ms, t_hand_ms=t_hand_ms,
        t_ser_ms=t_ser_ms, t_parse_ms=t_parse_ms, t_xport_excl_ms=t_xport_excl_ms,
        t_teardown_ms=t_teardown_ms,
        epsilon_ms=epsilon_ms, ptax_ms=ptax, ptr=ptr_value,
        b_mcp=b_mcp, tau_disc=tau_disc, phi=phi, gamma=gamma,
        s_ser_rows_per_ms=s_ser, rho=rho,
        n_life=len(by_phase["lifecycle"]),
        r_retry_ms=r_retry_ms,
        t_pool_ms=t_pool_ms,
        m_peak_mb=m_peak_mb,
        cpu_percent_samples=cpu_samples,
        trace_path=trace_path,
    )


def _parse_jsonl(path: Path) -> list[dict]:
    spans: list[dict] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            spans.append(json.loads(line))
    return spans


def _children_of(spans: list[dict], root: dict) -> list[dict]:
    """Return spans inside the root's time window (excluding the root itself).

    We use a time-window filter rather than parent_span_id traversal because
    nested span IDs can be omitted in some serializers; the time-window
    approach matches the harness's existing decompose() logic.
    """
    root_start, root_end = root["start_ns"], root["end_ns"]
    return [
        s for s in spans
        if s is not root
        and s["start_ns"] >= root_start
        and s["end_ns"] <= root_end
        and s["name"] != _INTERACTION_NAME
    ]


def _sum_durations_ms(spans: list[dict]) -> float:
    return sum(ns_to_ms(s["end_ns"] - s["start_ns"]) for s in spans)


def _exclusive_transport_ms(
    *, transport_spans: list[dict], handover_spans: list[dict],
    parse_spans: list[dict],
) -> float:
    total = 0.0
    for x in transport_spans:
        wall_ns = x["end_ns"] - x["start_ns"]
        overlap_h_ns = sum(_overlap_ns(x, h) for h in handover_spans)
        overlap_p_ns = sum(_overlap_ns(x, p) for p in parse_spans)
        total += ns_to_ms(max(0, wall_ns - overlap_h_ns - overlap_p_ns))
    return total


def _teardown_ms(*, root: dict, parse_spans: list[dict]) -> float:
    """Ninth timing term: root end minus the latest parse span end.

    Captures the interval between the framework's final ``mcp.parse`` end
    and the ``mcp.interaction`` end. Returns 0.0 when no parse span exists,
    which keeps epsilon_ms unchanged for interactions that never emit a
    parse span (e.g., pre-teardown-span traces or interactions that fail
    before parsing).
    """
    if not parse_spans:
        return 0.0
    last_parse_end = max(span["end_ns"] for span in parse_spans)
    return max(0.0, ns_to_ms(root["end_ns"] - last_parse_end))


def _overlap_ns(a: dict, b: dict) -> int:
    start = max(a["start_ns"], b["start_ns"])
    end = min(a["end_ns"], b["end_ns"])
    return max(0, end - start)


def _extract_t_db_ms(serialize_spans: list[dict]) -> float:
    for span in serialize_spans:
        attrs = span.get("attributes", {})
        if "t_db_ms" in attrs:
            return float(attrs["t_db_ms"])
    return 0.0


def _merge_attrs(spans: list[dict]) -> dict:
    merged: dict = {}
    for span in spans:
        merged.update(span.get("attributes", {}))
    return merged


def _peak_memory_from_spans(
    sample_carriers: list[dict], root: dict,
) -> float | None:
    """Return CM-3 M_peak in MiB.

    Samples may be attached to any of (SERIALIZE, PARSE, root); the
    fallback chain reflects placement differences between inprocess and
    subprocess adapters. When no ``rss_samples_kb`` attribute is present
    anywhere, return None so the CSV emits an empty cell rather than
    fabricating 0.0.
    """
    samples = _first_attribute(sample_carriers, "rss_samples_kb")
    if samples is None:
        samples = root.get("attributes", {}).get("rss_samples_kb")
    if samples is None:
        return None
    return cm3_peak_memory_mb(int(s) for s in samples)


def _cpu_samples_from_spans(sample_carriers: list[dict]) -> tuple[float, ...]:
    samples = _first_attribute(sample_carriers, "cpu_percent_samples")
    if samples is None:
        return ()
    return tuple(float(s) for s in samples)


def _first_attribute(spans: list[dict], key: str):
    for span in spans:
        attrs = span.get("attributes", {})
        if key in attrs:
            return attrs[key]
    return None


def _safe_phi(ser_attrs: dict, disc_attrs: dict) -> float | None:
    tau_star = ser_attrs.get("tau_star_result")
    tcr_tokens = ser_attrs.get("tool_call_response_tokens")
    disc_tokens = disc_attrs.get("tokens", disc_attrs.get("tau_disc"))
    if (
        tau_star is None or tcr_tokens is None or disc_tokens is None
        or float(tau_star) <= 0
    ):
        return None
    return (float(disc_tokens) + float(tcr_tokens)) / float(tau_star)


def _safe_gamma(ser_attrs: dict, disc_attrs: dict) -> float | None:
    boiler = ser_attrs.get("envelope_boiler_tokens")
    disc_tokens = disc_attrs.get("tokens", disc_attrs.get("tau_disc"))
    ctx_window = ser_attrs.get("context_window_size", disc_attrs.get("context_window_size"))
    if disc_tokens is None or boiler is None or ctx_window is None:
        return None
    if float(ctx_window) <= 0:
        return None
    return (float(disc_tokens) + float(boiler)) / float(ctx_window)


__all__ = ["TraceMetrics", "load_trace_metrics"]
