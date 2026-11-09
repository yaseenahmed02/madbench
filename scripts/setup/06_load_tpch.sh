#!/usr/bin/env bash
# Step 06: Load TPC-H data (SF1, SF3, SF10) into PostgreSQL, MySQL, DuckDB.
#
# Postgres : per-SF schemas    (sf1, sf3, sf10)
# MySQL    : per-SF databases  (tpch_sf1, tpch_sf3, tpch_sf10)
# DuckDB   : per-SF file       (/tmp/madbench_duckdb_sf<N>.duckdb)
#
# Idempotent: each schema/db is dropped and recreated.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="06-load-tpch"
step_banner "Step 06: Bulk-load TPC-H data into all 3 backends"

cd "${REPO_ROOT}"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  err ".venv missing. Run scripts/setup/02_install_repo.sh first."
fi

# Tell the loader where the .tbl files live (matches step 04)
DATA_DIR_BASE="${DATA_DIR_BASE:-$(scratch_dir)/data/tpch}"
if [[ ! -d "${DATA_DIR_BASE}" ]]; then
  err "TPC-H data dir not found: ${DATA_DIR_BASE}. Run scripts/setup/04_build_tpch_data.sh first."
fi
export TPCH_DATA_DIR="${DATA_DIR_BASE}"
log "TPCH_DATA_DIR=${TPCH_DATA_DIR}"

BACKENDS="${BACKENDS:-postgres,mysql,duckdb}"
SFS="${SFS:-1,3,10}"
log "Loading backends=${BACKENDS} sfs=${SFS}"

mkdir -p "${REPO_ROOT}/.cache"
"${REPO_ROOT}/.venv/bin/python" scripts/load_tpch.py \
  --backends "${BACKENDS}" \
  --sfs "${SFS}" \
  --pg-host localhost --pg-port 55432 --pg-user bench --pg-password bench --pg-database tpch \
  --mysql-host localhost --mysql-port 53306 --mysql-user bench --mysql-password bench \
  2>&1 | tee "${REPO_ROOT}/.cache/load_tpch.log" | tail -40

ok "Step 06 complete: TPC-H loaded into all 3 backends. See .cache/load_tpch.log"
