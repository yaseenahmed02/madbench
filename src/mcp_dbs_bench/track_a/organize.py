"""Canonical filesystem layout for Track A runs.

Every server run mirrors this structure under ``results/track_a/``:

    results/track_a/
    +-- <server>/
        +-- raw_traces/<server>_<backend>_<query>_sf<sf>_<cold_or_warm>/
        |   +-- manifest.json
        |   +-- trace.jsonl
        +-- csv/
        +-- figures/{png,pdf,source.py}
        +-- manifest.json
        +-- summary.md
        +-- budget.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ColdWarm = Literal["cold", "warm"]


@dataclass(frozen=True, slots=True)
class CellSpec:
    """Identifies a single (server, backend, query, SF, cold/warm) cell.

    A "cell" aggregates ``reps`` measured traces under identical conditions.

    Raises:
        ValueError: If ``cold_warm`` is not ``"cold"`` or ``"warm"``, or if
            ``scale_factor`` is not strictly positive.
    """

    server: str
    backend: str
    query: str
    scale_factor: int
    cold_warm: str

    def __post_init__(self) -> None:
        if self.cold_warm not in ("cold", "warm"):
            raise ValueError(
                f"cold_warm must be 'cold' or 'warm'; got {self.cold_warm!r}"
            )
        if self.scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be positive; got {self.scale_factor}"
            )


def cell_id(spec: CellSpec) -> str:
    """Canonical cell identifier: ``<server>_<backend>_<query>_sf<sf>_<cold|warm>``."""
    return (
        f"{spec.server}_{spec.backend}_{spec.query}"
        f"_sf{spec.scale_factor}_{spec.cold_warm}"
    )


def phase_root(output_root: Path, *, server: str) -> Path:
    """Return the server run directory under ``output_root``."""
    return output_root / server


def trace_dir(output_root: Path, *, spec: CellSpec) -> Path:
    """Return the per-cell trace directory."""
    return (
        phase_root(output_root, server=spec.server)
        / "raw_traces"
        / cell_id(spec)
    )


__all__ = ["CellSpec", "ColdWarm", "cell_id", "phase_root", "trace_dir"]
