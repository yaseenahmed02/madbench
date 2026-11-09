#!/usr/bin/env bash
# Step 01: Install uv and provision Python 3.11.
#
# Idempotent: detects existing uv and Python installations.
# Works on Linux and macOS. Does not require root.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="01-uv-python"
step_banner "Step 01: Install uv + Python ${PYTHON_VERSION}"

if have_cmd uv; then
  ok "uv already installed: $(uv --version)"
else
  log "Installing uv ${UV_VERSION} via the official installer..."
  require_network
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  have_cmd uv || err "uv install ran but 'uv' is not on PATH. Add ~/.local/bin to your PATH."
  ok "Installed uv $(uv --version)"
fi

log "Ensuring Python ${PYTHON_VERSION} is available via uv..."
uv python install "${PYTHON_VERSION}"
ok "Python ${PYTHON_VERSION} at: $(uv python find "${PYTHON_VERSION}")"
ok "Step 01 complete."
