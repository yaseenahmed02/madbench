#!/usr/bin/env bash
# Run the Track B matrix: every model x every coupling x every scale factor,
# with all questions in workloads/track_b/questions.yaml.
#
# Streams are split by DBMS: PostgreSQL attempts restart the container, so
# PostgreSQL couplings must not overlap each other, and mcp-alchemy locks the
# DuckDB file. The three DBMSs do not share state, so one stream each is safe.
#
# A finished (model, coupling, scale factor) leaves a .done marker, so rerunning
# this script resumes without repeating or re-billing completed work.
#
# Requires OPENROUTER_API_KEY in the environment.
# Overrides: MODELS, SFS, REPS, QUERY_IDS, QUERY_FILES, OUT_ROOT, BUDGET_USD.
# ONLY_COUPLING=server:backend runs that one coupling instead of all six.
# DBHUB_SEARCH_PATH_CONFIG=1 (default) makes dbhub on PostgreSQL log in as the
# per-scale-factor bench_sf<N> role (create it with dbhub_scoped_roles.sql).
#
# A cell whose log shows lost API access (key limit, no account credit, bad key)
# is never marked done: the sweep exits 0 with every attempt recorded as
# agent_error, which would otherwise look like a finished cell. The stream stops
# there and the script exits non-zero.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

if [ -s "${HOME}/.nvm/nvm.sh" ]; then
  # shellcheck source=/dev/null
  . "${HOME}/.nvm/nvm.sh" >/dev/null
fi
export PATH="${HOME}/.local/bin:${PATH}"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "FATAL: OPENROUTER_API_KEY is not set; export it before running Track B" >&2
  exit 1
fi

MODELS="${MODELS:-deepseek/deepseek-v4-flash z-ai/glm-5.3-flash openai/gpt-6-luna anthropic/claude-haiku-5.5}"
SFS="${SFS:-1 3}"
REPS="${REPS:-3}"
QUERY_IDS="${QUERY_IDS:-}"
QUERY_FILES="${QUERY_FILES:-workloads/track_b/questions.yaml}"
OUT_ROOT="${OUT_ROOT:-results/track_b}"
BUDGET_USD="${BUDGET_USD:-6}"
ONLY_COUPLING="${ONLY_COUPLING:-}"
export DBHUB_SEARCH_PATH_CONFIG="${DBHUB_SEARCH_PATH_CONFIG:-1}"

LOST_API_ACCESS_PATTERN="Key limit exceeded|OpenRouter returned HTTP 40[12] "

run_cell() {
  local model="$1" server="$2" backend="$3" sf="$4"
  local cell_dir="${OUT_ROOT}/${model##*/}/${server}-${backend}/sf${sf}"
  local started_at exit_code
  if [ -f "${cell_dir}/.done" ]; then
    echo "CELL_SKIP model=${model##*/} coupling=${server}:${backend} sf=${sf}"
    return 0
  fi
  mkdir -p "${cell_dir}"
  started_at="$(date +%s)"
  local query_id_args=()
  if [ -n "${QUERY_IDS}" ]; then
    query_id_args=(--query-ids "${QUERY_IDS}")
  fi
  # shellcheck disable=SC2086
  .venv/bin/python scripts/track_b/run_track_b_sweep.py \
    --server "${server}" --backends "${backend}" \
    --query-file ${QUERY_FILES} "${query_id_args[@]}" \
    --reps "${REPS}" --sf "${sf}" --paired-cache \
    --budget-usd "${BUDGET_USD}" --model "${model}" \
    --output-root "${cell_dir}" --log-level INFO \
    > "${cell_dir}/sweep.log" 2>&1
  exit_code=$?
  if grep -Eq "${LOST_API_ACCESS_PATTERN}" "${cell_dir}/sweep.log"; then
    echo "FATAL: API access lost in ${cell_dir}; cell left unfinished" >&2
    return 1
  fi
  if [ "${exit_code}" -eq 0 ]; then
    touch "${cell_dir}/.done"
  fi
  echo "CELL_END model=${model##*/} coupling=${server}:${backend} sf=${sf} exit=${exit_code} secs=$(( $(date +%s) - started_at ))"
}

run_stream() {
  local coupling server backend model sf
  for sf in ${SFS}; do
    for model in ${MODELS}; do
      for coupling in "$@"; do
        server="${coupling%%:*}"
        backend="${coupling##*:}"
        run_cell "${model}" "${server}" "${backend}" "${sf}" || return 1
      done
    done
  done
}

if [ -n "${ONLY_COUPLING}" ]; then
  run_stream "${ONLY_COUPLING}" || exit 1
  echo MATRIX_DONE
  exit 0
fi

run_stream mcp-alchemy:postgres dbhub:postgres anthropic-pg:postgres &
postgres_stream=$!
run_stream mcp-alchemy:mysql dbhub:mysql &
mysql_stream=$!
run_stream mcp-alchemy:duckdb &
duckdb_stream=$!

streams_failed=0
for stream in "${postgres_stream}" "${mysql_stream}" "${duckdb_stream}"; do
  wait "${stream}" || streams_failed=1
done
if [ "${streams_failed}" -ne 0 ]; then
  exit 1
fi
echo MATRIX_DONE
