"""Track B filesystem layout constants.

Centralised so all Track B scripts agree on paths.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return _REPO_ROOT


def query_file() -> Path:
    return _REPO_ROOT / "workloads" / "track_b" / "questions.yaml"


def gold_results_dir(sf: int = 1) -> Path:
    """Gold-results directory namespaced by TPC-H scale factor.

    Convention: ``workloads/track_b/gold_results_sf{sf}/`` so that gold
    values captured at different SFs (which differ because aggregate values
    scale with data size) do not collide.
    """
    return _REPO_ROOT / "workloads" / "track_b" / f"gold_results_sf{sf}"


def default_output_root() -> Path:
    return _REPO_ROOT / "results" / "track_b"


def phase_dir(output_root: Path, *, server: str) -> Path:
    """Directory for one server's Track B run."""
    return Path(output_root) / server


def raw_traces_dir(output_root: Path, *, server: str) -> Path:
    return phase_dir(output_root, server=server) / "raw_traces"


def cell_trace_dir(
    output_root: Path, *, server: str, cell_id: str,
) -> Path:
    return raw_traces_dir(output_root, server=server) / cell_id


def per_trace_csv(output_root: Path, *, server: str) -> Path:
    return phase_dir(output_root, server=server) / "per_trace_metrics.csv"


def per_cell_csv(output_root: Path, *, server: str) -> Path:
    return phase_dir(output_root, server=server) / "per_cell_summary.csv"


def budget_json(output_root: Path, *, server: str) -> Path:
    return phase_dir(output_root, server=server) / "budget.json"


def events_log(output_root: Path, *, server: str) -> Path:
    return phase_dir(output_root, server=server) / "run_events.log"


def manifest_json(output_root: Path, *, server: str) -> Path:
    return phase_dir(output_root, server=server) / "manifest.json"


__all__ = [
    "events_log", "budget_json", "cell_trace_dir", "default_output_root",
    "gold_results_dir", "manifest_json", "per_cell_csv", "per_trace_csv",
    "phase_dir", "query_file", "raw_traces_dir", "repo_root",
]
