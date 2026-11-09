#!/usr/bin/env bash
# Step 02: Create .venv/ and install the project with its locked dependencies.
#
# Idempotent: an existing .venv is reused; uv sync only changes what differs
# from uv.lock.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="02-repo-venv"
step_banner "Step 02: Provision .venv and install the project"

cd "${REPO_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"
have_cmd uv || err "uv not found. Run scripts/setup/01_install_uv_python.sh first."

log "Installing project + locked dependencies (uv sync --frozen --all-extras)..."
uv sync --frozen --all-extras --python "${PYTHON_VERSION}"

"${REPO_ROOT}/.venv/bin/python" -c "import mcp_dbs_bench, mcp_alchemy; print('imports OK')"
ok "Step 02 complete: .venv ready."
