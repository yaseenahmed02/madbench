#!/usr/bin/env bash
# Orchestrator: run all setup steps in order, bailing out on the first failure.
#
# Usage:
#   bash scripts/setup/setup_all.sh                # full setup
#   bash scripts/setup/setup_all.sh --skip-data    # everything except TPC-H gen+load (e.g., dry run)
#   bash scripts/setup/setup_all.sh --from 04      # resume from step N
#
# Honors the same env vars as the individual steps:
#   PYTHON_VERSION, NODE_VERSION, MCP_ALCHEMY_VERSION, DBHUB_VERSION,
#   SERVER_POSTGRES_VERSION
#   SCRATCH_DIR, DATA_DIR_BASE, TPCH_DATA_DIR
#   BACKENDS, SFS  (for the load step)

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="setup_all"

SKIP_DATA=0
FROM_STEP=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-data)  SKIP_DATA=1; shift ;;
    --from)       FROM_STEP="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,13p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) err "Unknown arg: $1" ;;
  esac
done

STEPS=(
  "01_install_uv_python.sh"
  "02_install_repo.sh"
  "03_install_mcp_servers.sh"
  "04_build_tpch_data.sh"
  "05_start_databases.sh"
  "06_load_tpch.sh"
  "06a_prepare_dbhub_postgres.sh"
  "06b_apply_constraints.sh"
)

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

OVERALL_START="$(date +%s)"
for i in "${!STEPS[@]}"; do
  STEP_NUM=$((i + 1))
  if (( STEP_NUM < FROM_STEP )); then
    log "skipping step ${STEP_NUM} (before --from)"
    continue
  fi
  STEP_FILE="${STEPS[$i]}"
  if (( SKIP_DATA == 1 )) && [[ "${STEP_FILE}" == "04_build_tpch_data.sh" || "${STEP_FILE}" == "06_load_tpch.sh" || "${STEP_FILE}" == "06a_prepare_dbhub_postgres.sh" ]]; then
    log "--skip-data: skipping ${STEP_FILE}"
    continue
  fi
  T0="$(date +%s)"
  bash "${SCRIPT_DIR}/${STEP_FILE}"
  ELAPSED=$(( $(date +%s) - T0 ))
  ok "completed ${STEP_FILE} in ${ELAPSED}s"
done

TOTAL=$(( $(date +%s) - OVERALL_START ))
ok "Full setup complete in ${TOTAL}s."
