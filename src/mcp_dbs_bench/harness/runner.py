"""Track A runner: one ``mcp.interaction`` end-to-end.

Lifecycle:
    1. capabilities() — pure
    2. async with plugin.connect(backend, phase)    -> LIFECYCLE phase
    3. plugin.discover(conn, phase)                  -> DISCOVERY phase
    4. plugin.call_tool(conn, request, phase)         -> HANDOVER / SERIALIZE / PARSE
       (TRANSPORT is recorded as a sibling)
    5. context exit                                   -> LIFECYCLE close
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from itertools import product

from mcp_dbs_bench.core.clock import monotonic_now_ns, ns_to_ms
from mcp_dbs_bench.core.derived import (
    protocol_tax_ms,
    protocol_tax_ratio,
)
from mcp_dbs_bench.core.phase_context import PhaseRecord
from mcp_dbs_bench.core.plugin import MCPServerPlugin
from mcp_dbs_bench.core.timing import TimingDecomposition, decompose
from mcp_dbs_bench.core.types import ToolCallRequest, ToolCallResponse
from mcp_dbs_bench.instrumentation.phase_context_otel import OTelPhaseContext
from mcp_dbs_bench.instrumentation.properties import (
    CompletenessResult,
    ExclusivityViolation,
    verify_completeness,
    verify_exclusivity,
    verify_observability,
)
from mcp_dbs_bench.instrumentation.tracer import OTelTracer

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InteractionResult:
    """All artifacts of one mcp.interaction."""

    server: str
    backend: str
    query_id: str
    scale_factor: int
    iteration: int
    cold_start: bool
    decomposition: TimingDecomposition
    completeness: CompletenessResult
    exclusivity_violations: list[ExclusivityViolation]
    observability_errors: list[str]
    response: ToolCallResponse
    records: list[PhaseRecord]

    @property
    def passes_all_properties(self) -> bool:
        return (
            self.completeness.passes
            and not self.exclusivity_violations
            and not self.observability_errors
        )

    @property
    def ptax_ms(self) -> float:
        d = self.decomposition
        return protocol_tax_ms(t_total_ms=d.t_total_ms, t_db_ms=d.t_db_ms)

    @property
    def ptr(self) -> float | None:
        if self.decomposition.t_db_ms <= 0:
            return None
        return protocol_tax_ratio(ptax_ms=self.ptax_ms, t_db_ms=self.decomposition.t_db_ms)


class TrackARunner:
    """Drives one (server, backend, query, scale_factor) cell.

    The runner owns the tracer and instantiates one OTelPhaseContext per
    interaction. The plugin sees only the PhaseContext — it cannot reach
    the tracer directly.
    """

    def __init__(
        self,
        *,
        plugin: MCPServerPlugin,
        backend: object,
        tracer: OTelTracer,
        backend_name: str,
        completeness_threshold: float = 0.05,
    ) -> None:
        self._plugin = plugin
        self._backend = backend
        self._tracer = tracer
        self._backend_name = backend_name
        self._completeness_threshold = completeness_threshold

    async def run_interaction(
        self,
        *,
        query_id: str,
        scale_factor: int,
        sql: str,
        iteration: int,
        cold_start: bool,
        tool_name: str = "query",
    ) -> InteractionResult:
        """Execute one full mcp.interaction and return its decomposition."""
        phase = OTelPhaseContext(tracer=self._tracer)
        request = ToolCallRequest(name=tool_name, arguments={"sql": sql})

        t_total_start_ns = monotonic_now_ns()
        with self._tracer.start_as_current_span("mcp.interaction") as root:
            root.set_attribute("server.name", self._plugin.name)
            root.set_attribute("dbms.backend", self._backend_name)
            root.set_attribute("track", "A")
            root.set_attribute("tpch.scale_factor", scale_factor)
            root.set_attribute("tpch.query_id", query_id)

            async with self._plugin.connect(self._backend, phase) as conn:
                await self._plugin.discover(conn, phase)
                response = await self._plugin.call_tool(conn, request, phase)
        t_total_end_ns = monotonic_now_ns()
        t_total_ms = ns_to_ms(t_total_end_ns - t_total_start_ns)

        t_db_ms = self._extract_t_db_ms(phase.records)
        decomposition = decompose(phase.records, t_total_ms=t_total_ms, t_db_ms=t_db_ms)

        completeness = verify_completeness(
            t_total_ms=decomposition.t_total_ms,
            t_db_ms=decomposition.t_db_ms,
            t_disc_ms=decomposition.t_disc_ms,
            t_life_ms=decomposition.t_life_ms,
            t_hand_ms=decomposition.t_hand_ms,
            t_ser_ms=decomposition.t_ser_ms,
            t_parse_ms=decomposition.t_parse_ms,
            t_xport_excl_ms=decomposition.t_xport_excl_ms,
            threshold=self._completeness_threshold,
        )
        exclusivity = verify_exclusivity(phase.records)
        observability = verify_observability(phase.records)

        return InteractionResult(
            server=self._plugin.name,
            backend=self._backend_name,
            query_id=query_id,
            scale_factor=scale_factor,
            iteration=iteration,
            cold_start=cold_start,
            decomposition=decomposition,
            completeness=completeness,
            exclusivity_violations=exclusivity,
            observability_errors=observability,
            response=response,
            records=phase.records,
        )

    def _extract_t_db_ms(self, records: list[PhaseRecord]) -> float:
        """T_db is recorded by the plugin as an attribute on the SERIALIZE phase.

        The contract: the plugin places the backend-reported t_db_ms into
        the SERIALIZE phase's attrs under the key ``t_db_ms``. This is the
        cleanest split: the plugin owns the backend handoff, the harness
        owns the decomposition.

        Raises:
            KeyError: If no PhaseRecord carries ``t_db_ms``. The harness
                cannot construct PTR or verify Property 1 without this
                value; silently defaulting to 0.0 would produce PTR=None
                and a misleading "completeness passes" trace. The plugin
                must set the attribute on the SERIALIZE
                phase span.
        """
        return _extract_t_db_ms_or_raise(records)


def _extract_t_db_ms_or_raise(records: list[PhaseRecord]) -> float:
    """Extract ``t_db_ms`` from the SERIALIZE phase's attrs.

    Plugin contract: every plugin must
    expose the backend-reported DB execution time as a ``t_db_ms`` attr
    on its SERIALIZE phase span. Without this value the harness cannot
    compute PTR or verify Property 1.

    Raises:
        KeyError: If no record carries ``t_db_ms``.
    """
    for record in records:
        if "t_db_ms" in record.attrs:
            return float(record.attrs["t_db_ms"])
    raise KeyError(
        "plugin contract violation: no PhaseRecord carries 't_db_ms'. "
        "The plugin must set the attribute on its SERIALIZE phase span "
        "(see the core.plugin.MCPServerPlugin docstring)."
    )


def randomized_run_order(
    *,
    servers: list[str],
    backends: list[str],
    queries: list[str],
    scale_factors: list[int],
    seed: int,
) -> list[tuple[str, str, str, int]]:
    """Block-randomized cell order.

    Every (server, backend, query, sf) tuple appears exactly once per pass.
    Seeded so a re-run reproduces the order.
    """
    rng = random.Random(seed)
    cells = list(product(servers, backends, queries, scale_factors))
    rng.shuffle(cells)
    return cells
