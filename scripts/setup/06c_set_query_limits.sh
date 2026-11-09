#!/usr/bin/env bash
# Step 06c: cap every benchmark SQL statement at 10 minutes.
#
# Track B agents can write queries that run for a very long time. The published
# runs capped each statement at 10 minutes on both PostgreSQL and MySQL. This
# runs after loading, because bulk loads and constraint checks take longer.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
SETUP_STEP="06c-query-limits"

step_banner "Step 06c: Set the 10-minute statement limit"

PG_LIMIT_SQL="ALTER ROLE bench SET statement_timeout = '10min';"
for role in bench_sf1 bench_sf3; do
  PG_LIMIT_SQL+=" DO \$\$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${role}') THEN ALTER ROLE ${role} SET statement_timeout = '10min'; END IF; END \$\$;"
done
docker exec "${PG_CONTAINER}" psql -v ON_ERROR_STOP=1 -U bench -d tpch -c "${PG_LIMIT_SQL}" >/dev/null
ok "PostgreSQL statement_timeout = 10min for the benchmark roles."

docker exec "${MYSQL_CONTAINER}" mysql -uroot -pbench -e "SET PERSIST max_execution_time = 600000;" 2>/dev/null
ok "MySQL max_execution_time = 600000 ms."
