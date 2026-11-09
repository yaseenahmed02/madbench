#!/usr/bin/env bash
# Shared helpers for scripts/setup/*.sh.
# Source this file from each setup step:
#   source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- Pinned versions (the versions used for the published results; Track A dbhub on PostgreSQL used dbhub 0.21.2) ----
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
UV_VERSION="${UV_VERSION:-0.5.10}"
NODE_VERSION="${NODE_VERSION:-22}"
NVM_VERSION="${NVM_VERSION:-v0.40.1}"
MCP_ALCHEMY_VERSION="${MCP_ALCHEMY_VERSION:-2025.8.15.91819}"
DBHUB_VERSION="${DBHUB_VERSION:-0.22.6}"
SERVER_POSTGRES_VERSION="${SERVER_POSTGRES_VERSION:-0.6.2}"
PG_CONTAINER="madbench-pg"
MYSQL_CONTAINER="madbench-mysql"

# ---- Colored logging ----
log()  { printf "\033[1;34m[%s]\033[0m %s\n" "${SETUP_STEP:-setup}" "$*"; }
ok()   { printf "\033[1;32m[%s] ✓\033[0m %s\n" "${SETUP_STEP:-setup}" "$*"; }
warn() { printf "\033[1;33m[%s] !\033[0m %s\n" "${SETUP_STEP:-setup}" "$*" >&2; }
err()  { printf "\033[1;31m[%s] ✗\033[0m %s\n" "${SETUP_STEP:-setup}" "$*" >&2; exit 1; }

# ---- OS detection ----
detect_os() {
  case "$(uname -s)" in
    Linux*)  echo "linux"  ;;
    Darwin*) echo "macos"  ;;
    *)       echo "unknown" ;;
  esac
}

# ---- Idempotency helpers ----
have_cmd() { command -v "$1" >/dev/null 2>&1; }

# ---- Scratch / data root selection ----
# Returns a path with enough room for TPC-H data + DB clusters.
# Priority: SCRATCH_DIR env > /tmp on Linux > $HOME/.madbench-scratch on macOS.
scratch_dir() {
  if [[ -n "${SCRATCH_DIR:-}" ]]; then
    echo "${SCRATCH_DIR}"
    return
  fi
  case "$(detect_os)" in
    linux) echo "/tmp/madbench-${USER}" ;;
    *)     echo "${HOME}/.madbench-scratch" ;;
  esac
}

# ---- Network reachability check ----
require_network() {
  if ! curl -sI --max-time 5 https://github.com >/dev/null; then
    err "Cannot reach github.com — network required for setup."
  fi
}

# ---- Step header ----
step_banner() {
  local title="$1"
  printf "\n\033[1;36m============================================================\033[0m\n"
  printf "\033[1;36m  %s\033[0m\n" "$title"
  printf "\033[1;36m============================================================\033[0m\n\n"
}
