#!/usr/bin/env python
"""Load TPC-H data into Postgres / MySQL / DuckDB at SF 1, 3, 10.

Idempotent: drops + recreates each target. Verifies lineitem counts.
Writes a JSON summary to stdout for the orchestrator to consume.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from mcp_dbs_bench.track_a.tpch_loaders import (
    EXPECTED_LINEITEM_COUNTS,
    load_duckdb_scale_factor,
    load_mysql_scale_factor,
    load_postgres_scale_factor,
    verify_lineitem_count,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DUCKDB_PATH_TEMPLATE = "/tmp/madbench_duckdb_sf{sf}.duckdb"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
_LOG = logging.getLogger("load_tpch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", default="postgres,mysql,duckdb",
        help="Comma-separated list of backends",
    )
    parser.add_argument(
        "--sfs", default="1,3,10",
        help="Comma-separated list of scale factors",
    )
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=55432)
    parser.add_argument("--pg-user", default="bench")
    parser.add_argument("--pg-password", default="bench")
    parser.add_argument("--pg-database", default="tpch")
    parser.add_argument("--mysql-host", default="localhost")
    parser.add_argument("--mysql-port", type=int, default=53306)
    parser.add_argument("--mysql-user", default="bench")
    parser.add_argument("--mysql-password", default="bench")
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    sfs = [int(x.strip()) for x in args.sfs.split(",") if x.strip()]

    report: dict = {"backends": {}, "expected_lineitem": EXPECTED_LINEITEM_COUNTS}
    total_started = time.time()
    for backend in backends:
        report["backends"][backend] = {}
        for sf in sfs:
            t0 = time.time()
            _LOG.info("loading %s SF%d", backend, sf)
            try:
                if backend == "postgres":
                    counts = load_postgres_scale_factor(
                        host=args.pg_host, port=args.pg_port,
                        user=args.pg_user, password=args.pg_password,
                        database=args.pg_database,
                        scale_factor=sf, repo_root=_REPO_ROOT,
                    )
                elif backend == "mysql":
                    counts = load_mysql_scale_factor(
                        host=args.mysql_host, port=args.mysql_port,
                        user=args.mysql_user, password=args.mysql_password,
                        scale_factor=sf, repo_root=_REPO_ROOT,
                    )
                elif backend == "duckdb":
                    counts = load_duckdb_scale_factor(
                        db_path=Path(_DUCKDB_PATH_TEMPLATE.format(sf=sf)),
                        scale_factor=sf, repo_root=_REPO_ROOT,
                    )
                else:
                    raise ValueError(f"unknown backend {backend!r}")
                verify_lineitem_count(counts, sf)
                duration = time.time() - t0
                report["backends"][backend][f"sf{sf}"] = {
                    "ok": True, "counts": counts, "duration_s": duration,
                }
                _LOG.info(
                    "%s SF%d OK lineitem=%s duration=%.1fs",
                    backend, sf, counts.get("lineitem"), duration,
                )
            except Exception as exc:
                _LOG.exception("%s SF%d FAILED", backend, sf)
                report["backends"][backend][f"sf{sf}"] = {
                    "ok": False, "error": str(exc),
                    "duration_s": time.time() - t0,
                }
    report["total_duration_s"] = time.time() - total_started
    print(json.dumps(report, indent=2))
    failed = [
        f"{backend} {sf_key}"
        for backend, by_sf in report["backends"].items()
        for sf_key, result in by_sf.items()
        if not result["ok"]
    ]
    if failed:
        _LOG.error("Load failed for: %s. See the tracebacks above.", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
