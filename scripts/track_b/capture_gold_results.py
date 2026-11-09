#!/usr/bin/env python
"""Capture canonical Track B gold result sets.

For each (query, backend) pair in the workload YAML, execute the pre-
registered gold SQL against the actual TPC-H database and persist the
canonical result set as JSON. These files are the ground truth against
which the correctness verifier compares agent answers during the sweep.

The gold JSON is idempotent, driver-independent, and diff-stable:
  - column names are lowercased + stripped
  - floats round to 4 decimals; Decimal coerced to float
  - rows are sorted deterministically unless the gold SQL has ORDER BY

Usage
-----
    python scripts/track_b/capture_gold_results.py \\
        --query-file workloads/track_b/questions.yaml \\
        --backends postgres,mysql,duckdb \\
        --sf 1 \\
        --output-dir workloads/track_b/gold_results

Add ``--force`` to overwrite existing gold files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from track_b._backends import SUPPORTED_BACKENDS, connect  # noqa: E402
from track_b._normalise import normalise_columns, normalise_rows  # noqa: E402
from track_b._workload import WorkloadError, load_workload  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
_LOG = logging.getLogger("track_b.gold")

DEFAULT_QUERY_FILE = _REPO_ROOT / "workloads" / "track_b" / "questions.yaml"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "workloads" / "track_b" / "gold_results"


def main() -> int:
    args = _parse_args()
    queries = _load_queries(args.query_file)
    backends = _parse_csv(args.backends, valid=set(SUPPORTED_BACKENDS))
    # SF-namespace the gold dir so SF1 and SF3 gold don't collide.
    # Gold values scale with SF (aggregates like SUM(o_totalprice) differ),
    # so caching by (query_id, backend) alone is wrong across SFs.
    output_dir = Path(args.output_dir)
    if output_dir.name in ("gold_results", "gold_results_sf1", "gold_results_sf3", "gold_results_sf10"):
        # If the user passed the base gold_results dir (or an SF-suffixed
        # one), route to the SF-namespaced sub-path for the requested SF.
        base = output_dir.parent / "gold_results" if output_dir.name.startswith("gold_results") else output_dir.parent
        output_dir = base.parent / f"gold_results_sf{args.sf}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info(
        "capturing gold: %d queries × %d backends (sf=%d) -> %s",
        len(queries), len(backends), args.sf, output_dir,
    )
    summary = _capture_all(
        queries=queries, backends=backends, sf=args.sf,
        output_dir=output_dir, force=args.force,
    )
    _write_summary(output_dir=output_dir, summary=summary)
    _log_summary(summary)
    return 0 if summary["failed"] == 0 else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file", nargs="+", default=[str(DEFAULT_QUERY_FILE)],
        help="one or more workload YAMLs (e.g. the c2/c5 and c4/c6 files)",
    )
    parser.add_argument("--backends", default="postgres,mysql,duckdb")
    parser.add_argument("--sf", type=int, default=1)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing gold JSON files.",
    )
    return parser.parse_args()


def _parse_csv(spec: str, *, valid: set[str]) -> list[str]:
    items = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [x for x in items if x not in valid]
    if unknown:
        raise SystemExit(f"unknown backend(s): {unknown!r}; expected any of {valid!r}")
    return items


def _load_queries(paths: list[str]) -> list[dict[str, Any]]:
    """Load one or more workload YAMLs into a flat ``[{id, question, gold_sql}]``."""
    try:
        return load_workload([Path(p) for p in paths])
    except WorkloadError as exc:
        raise SystemExit(str(exc)) from exc


def _capture_all(
    *,
    queries: list[dict[str, Any]],
    backends: list[str],
    sf: int,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    """Iterate the (backend, query) grid. Returns a summary of outcomes."""
    counts = {"captured": 0, "skipped": 0, "failed": 0}
    failures: list[dict[str, Any]] = []
    for backend in backends:
        with _open_backend(backend, sf) as client:
            if client is None:
                counts["failed"] += len(queries)
                failures.append({"backend": backend, "error": "connect failed"})
                continue
            for query in queries:
                outcome = _capture_one(
                    query=query, backend=backend, sf=sf,
                    client=client, output_dir=output_dir, force=force,
                )
                counts[outcome["status"]] += 1
                if outcome["status"] == "failed":
                    failures.append(outcome)
    return {**counts, "failures": failures}


class _OptionalCM:
    """Yield ``None`` when the underlying connect() raises."""

    def __init__(self, backend: str, sf: int) -> None:
        self._backend = backend
        self._sf = sf
        self._cm: Any = None

    def __enter__(self) -> Any:
        try:
            self._cm = connect(self._backend, self._sf)
            return self._cm.__enter__()
        except Exception as exc:
            _LOG.exception("connect(%s, sf=%d) failed: %s", self._backend, self._sf, exc)
            return None

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._cm is not None:
            self._cm.__exit__(exc_type, exc, tb)


def _open_backend(backend: str, sf: int) -> _OptionalCM:
    return _OptionalCM(backend, sf)


def _capture_one(
    *,
    query: dict[str, Any],
    backend: str,
    sf: int,
    client: Any,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    """Run one gold SQL, normalise, write JSON; return an outcome dict."""
    query_id = query["id"]
    target = output_dir / f"{query_id}__{backend}.json"
    if target.exists() and not force:
        _LOG.info("skip %s (%s) — already captured", query_id, backend)
        return {"status": "skipped", "path": str(target)}
    try:
        cols, rows = client.execute(query["gold_sql"])
    except Exception as exc:
        _LOG.exception("gold SQL failed for %s on %s", query_id, backend)
        return {
            "status": "failed", "query_id": query_id, "backend": backend,
            "error": f"{type(exc).__name__}: {exc}",
        }
    canonical_cols = normalise_columns(cols)
    canonical_rows = normalise_rows(rows, gold_sql=query["gold_sql"])
    payload = {
        "query_id": query_id,
        "category": query["category"],
        "target_subcat": query["target_subcat"],
        "question": query["question"],
        "backend": backend,
        "sf": sf,
        "gold_sql": query["gold_sql"],
        "columns": canonical_cols,
        "rows": canonical_rows,
        "row_count": len(canonical_rows),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(target, payload)
    _LOG.info(
        "wrote %s (%s): %d rows × %d cols",
        query_id, backend, len(canonical_rows), len(canonical_cols),
    )
    return {"status": "captured", "path": str(target)}


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    """Write JSON via temp+rename so a crash never leaves half-written files."""
    tmp = target.parent / (target.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    tmp.replace(target)


def _write_summary(*, output_dir: Path, summary: dict[str, Any]) -> None:
    path = output_dir / "_capture_summary.json"
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    path.write_text(json.dumps(payload, indent=2))


def _log_summary(summary: dict[str, Any]) -> None:
    _LOG.info(
        "done: captured=%d skipped=%d failed=%d",
        summary["captured"], summary["skipped"], summary["failed"],
    )
    for f in summary["failures"][:10]:
        _LOG.warning("failure: %s", f)


if __name__ == "__main__":
    raise SystemExit(main())
