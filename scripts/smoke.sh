#!/usr/bin/env bash
# Quick end-to-end check of a finished setup.
#
#   Track A: mcp-alchemy x PostgreSQL x Q6 x SF1, warm, 2 measured reps.
#   Track B: one question (tb_c2_01) x deepseek/deepseek-v4-flash x
#            mcp-alchemy on PostgreSQL x SF1, 1 rep (a few cents at most).
#
# Every prerequisite is checked first; the script stops with a message naming
# the missing piece and the setup step that provides it.
#
# Usage: bash scripts/smoke.sh            (both tracks)
#        SKIP_TRACK_B=1 bash scripts/smoke.sh   (Track A only, no API key needed)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PY="${REPO_ROOT}/.venv/bin/python"
OUT="${REPO_ROOT}/results/smoke"
SMOKE_MODEL="${SMOKE_MODEL:-deepseek/deepseek-v4-flash}"

fail() { printf "\033[1;31mSMOKE FAILED:\033[0m %s\n" "$*" >&2; exit 1; }
pass() { printf "\033[1;32mok\033[0m %s\n" "$*"; }
count_rows() { "${PY}" -c "import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))" "$1"; }

[[ -s "${HOME}/.nvm/nvm.sh" ]] && { export NVM_DIR="${HOME}/.nvm"; source "${NVM_DIR}/nvm.sh" >/dev/null; }
export PATH="${HOME}/.local/bin:${PATH}"

[[ -x "${PY}" ]] || fail ".venv is missing. Run: bash scripts/setup/02_install_repo.sh"
pass "python $(${PY} -c 'import sys; print(sys.version.split()[0])')"

command -v node >/dev/null || fail "node is not on PATH. Run: bash scripts/setup/03_install_mcp_servers.sh"
pass "node $(node --version)"
[[ "$(node --version)" == v22.* ]] || printf "warning: the published results used Node.js 22; found %s\n" "$(node --version)"
for bin in mcp-alchemy dbhub mcp-server-postgres; do
  command -v "${bin}" >/dev/null || fail "${bin} is not on PATH. Run: bash scripts/setup/03_install_mcp_servers.sh"
done
pass "MCP servers on PATH"

"${PY}" - <<'PYEOF' || fail "PostgreSQL on localhost:55432 is unreachable or schema sf1 is not loaded. Run: bash scripts/setup/05_start_databases.sh, then 06_load_tpch.sh"
import psycopg
with psycopg.connect("postgresql://bench:bench@localhost:55432/tpch", connect_timeout=10) as conn:
    count = conn.execute("SELECT count(*) FROM sf1.lineitem").fetchone()[0]
assert count == 6_001_215, f"sf1.lineitem has {count} rows, expected 6001215"
PYEOF
pass "PostgreSQL sf1 loaded"

rm -rf "${OUT}" && mkdir -p "${OUT}"
"${PY}" scripts/run_track_a_mcp_alchemy.py \
  --backends postgres --queries Q6 --sfs 1 --reps 2 --warmup 1 --cold-warm warm \
  --output-root "${OUT}/track_a" >"${OUT}/track_a.log" 2>&1 \
  || fail "Track A run failed; see ${OUT}/track_a.log"
TRACES="${OUT}/track_a/mcp-alchemy/csv/per_trace_metrics.csv"
[[ -f "${TRACES}" && "$(count_rows "${TRACES}")" -eq 2 ]] || fail "expected 2 Track A traces in ${TRACES}"
pass "Track A: 2 traces in ${TRACES}"

if [[ "${SKIP_TRACK_B:-0}" == "1" ]]; then
  echo "SMOKE PASSED (Track B skipped)"
  exit 0
fi

[[ -n "${OPENROUTER_API_KEY:-}" ]] || fail "OPENROUTER_API_KEY is not set. Export it, or run with SKIP_TRACK_B=1"
"${PY}" scripts/track_b/capture_gold_results.py --backends postgres --sf 1 >"${OUT}/gold.log" 2>&1 \
  || fail "gold capture failed; see ${OUT}/gold.log"
"${PY}" scripts/track_b/run_track_b_sweep.py \
  --server mcp-alchemy --backends postgres --query-ids tb_c2_01 --reps 1 --sf 1 \
  --model "${SMOKE_MODEL}" --budget-usd 0.5 --output-root "${OUT}/track_b" \
  >"${OUT}/track_b.log" 2>&1 || fail "Track B run failed; see ${OUT}/track_b.log"
ATTEMPTS="${OUT}/track_b/mcp-alchemy/per_trace_metrics.csv"
[[ -f "${ATTEMPTS}" && "$(count_rows "${ATTEMPTS}")" -eq 1 ]] || fail "expected 1 Track B attempt in ${ATTEMPTS}"
OUTCOME="$("${PY}" -c "import csv,sys; print(next(csv.DictReader(open(sys.argv[1])))['outcome'])" "${ATTEMPTS}")"
if [[ "${OUTCOME}" == "agent_error" ]]; then
  grep -q "401 Unauthorized" "${OUT}/track_b.log" && fail "OpenRouter rejected OPENROUTER_API_KEY (HTTP 401)"
  fail "Track B attempt ended in agent_error; see ${OUT}/track_b.log"
fi
pass "Track B: 1 attempt, outcome=${OUTCOME}"
echo "SMOKE PASSED"
