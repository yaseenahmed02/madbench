#!/usr/bin/env bash
# Step 04: Generate TPC-H data at SF1, SF3, and SF10.
#
# Wraps scripts/build_tpch.sh and places output under $DATA_DIR (default:
# scratch_dir on Linux, repo data/ on macOS — see _lib.sh).
#
# SF10 takes ~5-15 minutes depending on CPU; SF1+SF3 are sub-minute.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="04-tpch-data"
step_banner "Step 04: Build TPC-H dbgen and generate SF1/SF3/SF10 data"

cd "${REPO_ROOT}"
require_network

# Pick a data dir with enough room (~15 GB for SF1+SF3+SF10 combined)
DATA_DIR="${DATA_DIR:-$(scratch_dir)/data/tpch}"
mkdir -p "${DATA_DIR}"
log "DATA_DIR=${DATA_DIR}"

# Surface to build_tpch.sh
export DATA_DIR

# Scale factors to generate; override with SF env var (space-sep list)
SF_LIST="${SF:-1 3 10}"
log "Generating scale factors: ${SF_LIST}"

# Hand off to the canonical builder (auto-detects MACHINE=LINUX|MACOS)
bash "${REPO_ROOT}/scripts/build_tpch.sh" ${SF_LIST}

# Sanity check
for sf in ${SF_LIST}; do
  if [[ ! -f "${DATA_DIR}/sf${sf}/lineitem.tbl" ]]; then
    err "Expected file missing: ${DATA_DIR}/sf${sf}/lineitem.tbl"
  fi
  ROWS="$(wc -l < "${DATA_DIR}/sf${sf}/lineitem.tbl")"
  log "sf${sf}: lineitem.tbl has ${ROWS} rows"
done

ok "Step 04 complete: TPC-H data at ${DATA_DIR}"
