#!/usr/bin/env bash
# Step 03: Install the three upstream MCP servers at pinned versions.
#
#   1. mcp-alchemy          (Python, via uv tool install + DB drivers)
#   2. dbhub                (Node.js, @bytebase/dbhub via npm)
#   3. mcp-server-postgres  (Node.js, @modelcontextprotocol/server-postgres via npm)
#
# Node.js ${NODE_VERSION} is installed with nvm when npm is not on PATH.
# The Track B DuckDB cells run mcp-alchemy from the project .venv (step 02)
# through scripts/track_b/mcp_alchemy_shim.py.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="03-mcp-servers"
step_banner "Step 03: Install the upstream MCP servers"

cd "${REPO_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"

log "Installing mcp-alchemy ${MCP_ALCHEMY_VERSION} + DB drivers..."
uv tool install --force "mcp-alchemy==${MCP_ALCHEMY_VERSION}" \
  --with "psycopg[binary]" --with psycopg2-binary --with pymysql --with duckdb-engine
have_cmd mcp-alchemy || err "mcp-alchemy install ran but the binary is not on PATH (~/.local/bin)."
ok "mcp-alchemy: $(command -v mcp-alchemy)"

if [[ -d "${HOME}/.nvm" ]]; then
  export NVM_DIR="${HOME}/.nvm"
  # shellcheck source=/dev/null
  source "${NVM_DIR}/nvm.sh"
fi
if ! have_cmd npm || [[ "$(node --version 2>/dev/null)" != v${NODE_VERSION}.* ]]; then
  log "Installing Node.js ${NODE_VERSION} via nvm ${NVM_VERSION}..."
  export NVM_DIR="${HOME}/.nvm"
  if [[ ! -d "${NVM_DIR}" ]]; then
    require_network
    curl -o- "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | bash
  fi
  # shellcheck source=/dev/null
  source "${NVM_DIR}/nvm.sh"
  nvm install "${NODE_VERSION}"
  nvm use "${NODE_VERSION}"
fi
ok "Node.js $(node --version) / npm $(npm --version)"

log "Installing @bytebase/dbhub@${DBHUB_VERSION}..."
npm install -g "@bytebase/dbhub@${DBHUB_VERSION}"
have_cmd dbhub || err "dbhub install ran but the binary is not on PATH."

log "Installing @modelcontextprotocol/server-postgres@${SERVER_POSTGRES_VERSION}..."
npm install -g "@modelcontextprotocol/server-postgres@${SERVER_POSTGRES_VERSION}"
have_cmd mcp-server-postgres || err "mcp-server-postgres install ran but the binary is not on PATH."

printf "  %-22s -> %s\n" "mcp-alchemy" "$(command -v mcp-alchemy)"
printf "  %-22s -> %s\n" "dbhub" "$(command -v dbhub)"
printf "  %-22s -> %s\n" "mcp-server-postgres" "$(command -v mcp-server-postgres)"
ok "Step 03 complete."
