-- Logins for dbhub on PostgreSQL. Each can read one TPC-H schema only, so an
-- agent cannot answer a question from another scale factor's data.
-- The query limit matches the bench role.
DO $$
DECLARE
  scale_factor text;
  login text;
BEGIN
  FOREACH scale_factor IN ARRAY ARRAY['1', '3'] LOOP
    login := 'bench_sf' || scale_factor;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = login) THEN
      EXECUTE format('CREATE ROLE %I LOGIN PASSWORD %L', login, 'bench');
    END IF;
    EXECUTE format('ALTER ROLE %I SET statement_timeout = %L', login, '10min');
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO %I', 'sf' || scale_factor, login);
    EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO %I', 'sf' || scale_factor, login);
  END LOOP;
END
$$;
