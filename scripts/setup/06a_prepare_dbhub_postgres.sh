#!/usr/bin/env bash
# Step 06a: PostgreSQL layouts used by dbhub.
#
# dbhub drops the libpq ``options`` parameter of a DSN, so it cannot select
# the per-SF schema of the shared ``tpch`` database through search_path.
#   - Track A points dbhub at per-SF databases tpch_sf<N> with the tables in
#     ``public``. This step builds them by loading schema sf<N> into a fresh
#     database and renaming it to ``public``.
#   - Track B logs dbhub in as bench_sf<N>, a role that can read only schema
#     sf<N> of ``tpch`` (scripts/track_b/dbhub_scoped_roles.sql).
#
# Idempotent: each tpch_sf<N> database is dropped and recreated.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="06a-dbhub-postgres"
step_banner "Step 06a: Prepare PostgreSQL databases and roles for dbhub"

cd "${REPO_ROOT}"
[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || err ".venv missing. Run scripts/setup/02_install_repo.sh first."

DATA_DIR_BASE="${DATA_DIR_BASE:-$(scratch_dir)/data/tpch}"
[[ -d "${DATA_DIR_BASE}" ]] || err "TPC-H data dir not found: ${DATA_DIR_BASE}. Run scripts/setup/04_build_tpch_data.sh first."
export TPCH_DATA_DIR="${DATA_DIR_BASE}"

SFS="${SFS:-1,3,10}"
psql_in() { docker exec -i "${PG_CONTAINER}" psql -v ON_ERROR_STOP=1 -U bench -d "$1"; }

for sf in ${SFS//,/ }; do
  db="tpch_sf${sf}"
  log "Building ${db} (tables in public)..."
  echo "DROP DATABASE IF EXISTS ${db}; CREATE DATABASE ${db};" | psql_in tpch >/dev/null
  "${REPO_ROOT}/.venv/bin/python" scripts/load_tpch.py --backends postgres --sfs "${sf}" \
    --pg-database "${db}" > /dev/null
  echo "DROP SCHEMA public CASCADE; ALTER SCHEMA sf${sf} RENAME TO public;" | psql_in "${db}" >/dev/null
  ok "${db} ready"
done

if [[ ",${SFS}," == *",1,"* && ",${SFS}," == *",3,"* ]]; then
  log "Creating the scoped bench_sf1 / bench_sf3 roles for Track B dbhub..."
  psql_in tpch < scripts/track_b/dbhub_scoped_roles.sql >/dev/null
else
  warn "SFS=${SFS} lacks SF1 or SF3; skipping the Track B dbhub roles (they need schemas sf1 and sf3)."
fi
ok "Step 06a complete."
