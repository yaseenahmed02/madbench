#!/usr/bin/env bash
# Step 06b: Apply TPC-H PK + FK constraints + standard FK/date indexes.
#
# The TPC-H kit ships dss.ddl (tables) and dss.ri (PK + FK). Step 06 applies
# dss.ddl; this step applies the equivalent of dss.ri (adapted from DB2 to
# PG/MySQL) plus the standard FK index set used in published TPC-H work.
#
# Without PK/FK indexes, join queries (Q3/Q9/Q21) run via full table scans
# and hash joins instead of index seeks, which inflates T_db and deflates PTR.
#
# References:
#   - Leis et al. PVLDB 2015 ("How Good Are Query Optimizers, Really?")
#   - Boncz, Neumann, Erling TPCTC 2013 ("TPC-H Analyzed")
#   - TPC-H Spec v3.0.1 sections 1.5.6-1.5.7 (permitted index DDL)

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="06b-apply-constraints"
step_banner "Step 06b: Apply TPC-H PK + FK + indexes (canonical TPC-H config)"

cd "${REPO_ROOT}"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  err ".venv missing. Run scripts/setup/02_install_repo.sh first."
fi

BACKENDS="${BACKENDS:-postgres,mysql}"
SFS="${SFS:-1,3,10}"
log "Applying constraints to backends=${BACKENDS} sfs=${SFS}"

mkdir -p "${REPO_ROOT}/.cache"
"${REPO_ROOT}/.venv/bin/python" scripts/apply_tpch_constraints.py \
  --backends "${BACKENDS}" \
  --sfs "${SFS}" \
  --pg-host localhost --pg-port 55432 --pg-user bench --pg-password bench --pg-database tpch \
  --pg-dbhub-databases "$(printf 'tpch_sf%s,' ${SFS//,/ } | sed 's/,$//')" \
  --mysql-host localhost --mysql-port 53306 --mysql-user bench --mysql-password bench \
  2>&1 | tee "${REPO_ROOT}/.cache/apply_constraints.log"

ok "Step 06b complete: TPC-H PK+FK+indexes applied. See .cache/apply_constraints.log"
