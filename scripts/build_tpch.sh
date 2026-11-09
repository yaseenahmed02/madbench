#!/usr/bin/env bash
# Fetch and build TPC-H dbgen, then generate data at the requested scale factors.
#
# Usage:
#   scripts/build_tpch.sh [scale_factor ...]
#
# Defaults to scale factors 1 3 10 if none are given. Output goes to
# .cache/tpch-data/sf<N>/ as |-delimited .tbl files for COPY / LOAD DATA INFILE.
#
# Requirements: git, make, a C compiler. dbgen is not redistributed by this
# project; the user is responsible for the TPC research-use license terms.
#
# Honored env vars:
#   DBGEN_MACHINE  Override the dbgen MACHINE value (LINUX | MACOS).
#                  Auto-detected from uname when unset.
#   DATA_DIR       Override output dir for the .tbl files
#                  (default: <repo>/.cache/tpch-data).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DBGEN_DIR="${REPO_ROOT}/.cache/tpch-kit/dbgen"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/.cache/tpch-data}"

# Auto-detect dbgen MACHINE value if not overridden.
if [[ -z "${DBGEN_MACHINE:-}" ]]; then
  case "$(uname -s)" in
    Linux*)  DBGEN_MACHINE="LINUX" ;;
    Darwin*) DBGEN_MACHINE="MACOS" ;;
    *) echo "[build_tpch] unknown OS $(uname -s); set DBGEN_MACHINE explicitly" >&2; exit 1 ;;
  esac
fi
echo "[build_tpch] dbgen MACHINE=${DBGEN_MACHINE}"

SCALE_FACTORS=("$@")
if [[ ${#SCALE_FACTORS[@]} -eq 0 ]]; then
  SCALE_FACTORS=(1 3 10)
fi

echo "[build_tpch] cloning tpch-kit if needed..."
if [[ ! -d "${REPO_ROOT}/.cache/tpch-kit/.git" ]]; then
  mkdir -p "${REPO_ROOT}/.cache"
  # gregrahn/tpch-kit is the modern-compiler-friendly fork; electrum/tpch-dbgen
  # has stale typedefs that fail on clang 14+ and gcc 12+. Users in academic
  # settings should review TPC research-use terms (EULA.txt in repo).
  git clone --depth 1 https://github.com/gregrahn/tpch-kit.git "${REPO_ROOT}/.cache/tpch-kit"
fi

echo "[build_tpch] building dbgen (MACHINE=${DBGEN_MACHINE})..."
pushd "${DBGEN_DIR}" >/dev/null
make -s CC="${CC:-cc}" MACHINE="${DBGEN_MACHINE}" DATABASE=POSTGRESQL WORKLOAD=TPCH 2>&1 | tail -3
popd >/dev/null

for SF in "${SCALE_FACTORS[@]}"; do
  OUT="${DATA_DIR}/sf${SF}"
  if [[ -f "${OUT}/lineitem.tbl" ]]; then
    echo "[build_tpch] sf${SF} already generated at ${OUT}; skipping"
    continue
  fi
  echo "[build_tpch] generating sf${SF} into ${OUT}..."
  mkdir -p "${OUT}"
  pushd "${DBGEN_DIR}" >/dev/null
  ./dbgen -vf -s "${SF}"
  mv ./*.tbl "${OUT}/"
  popd >/dev/null
done

echo "[build_tpch] done. Data layout:"
ls -lh "${DATA_DIR}"/sf*/lineitem.tbl 2>/dev/null || true
