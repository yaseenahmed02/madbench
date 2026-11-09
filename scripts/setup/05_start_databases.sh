#!/usr/bin/env bash
# Step 05: Start the PostgreSQL and MySQL containers via docker-compose.
#
# Uses docker-compose.yml at the repo root (PG on :55432, MySQL on :53306).
# Waits for both containers to report healthy.
#
# Requires:
#   - docker engine + docker compose plugin
#   - the user MUST be in the `docker` group (or docker rootless set up).
#
# Idempotent: containers already running are reused.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="05-databases"
step_banner "Step 05: Start PostgreSQL + MySQL containers"

cd "${REPO_ROOT}"

if ! have_cmd docker; then
  err "docker not found. On Ubuntu: 'sudo apt install docker.io' or have your admin add you to the 'docker' group."
fi

if ! docker info >/dev/null 2>&1; then
  err "docker is installed but the daemon is unreachable. \
Check: (a) you're in the 'docker' group (run 'id | grep docker'), \
(b) the daemon is running ('systemctl status docker'), or \
(c) try rootless docker via 'dockerd-rootless-setuptool.sh install'."
fi

if ! docker compose version >/dev/null 2>&1; then
  err "'docker compose' subcommand not available. Install docker-compose-plugin."
fi

log "Bringing up PostgreSQL + MySQL with docker compose..."
docker compose up -d postgres mysql

log "Waiting for health checks (up to 120s)..."
for i in $(seq 1 60); do
  PG_STATUS="$(docker inspect -f '{{.State.Health.Status}}' ${PG_CONTAINER} 2>/dev/null || echo missing)"
  MY_STATUS="$(docker inspect -f '{{.State.Health.Status}}' ${MYSQL_CONTAINER} 2>/dev/null || echo missing)"
  if [[ "${PG_STATUS}" == "healthy" && "${MY_STATUS}" == "healthy" ]]; then
    ok "Both containers healthy after ${i} polls."
    break
  fi
  if (( i % 5 == 0 )); then
    log "Still waiting: postgres=${PG_STATUS}, mysql=${MY_STATUS}"
  fi
  sleep 2
done

if [[ "${PG_STATUS}" != "healthy" || "${MY_STATUS}" != "healthy" ]]; then
  err "Timeout waiting for containers to be healthy (pg=${PG_STATUS}, mysql=${MY_STATUS}). Check 'docker compose logs'."
fi

# Connectivity smoke test
log "Smoke test: connect to PostgreSQL inside the container..."
docker exec "${PG_CONTAINER}" psql -U bench -d tpch -c "SELECT 1 AS ok;" >/dev/null
ok "PostgreSQL on :55432 is accepting connections."

# Track B caps every SQL statement at 10 minutes so a runaway agent query cannot stall a run.
docker exec "${PG_CONTAINER}" psql -U bench -d tpch -c "ALTER ROLE bench SET statement_timeout = '10min';" >/dev/null
ok "PostgreSQL statement_timeout for role bench set to 10min."

if have_cmd mysql; then
  log "Smoke test: mysql connect to localhost:53306..."
  mysql -h 127.0.0.1 -P 53306 -ubench -pbench -e "SELECT 1 AS ok;" tpch 2>/dev/null >/dev/null
  ok "MySQL on :53306 is accepting connections."
else
  warn "mysql client not installed; skipping client-side smoke test. The container health check already passed."
fi

ok "Step 05 complete: PG + MySQL containers ready."
