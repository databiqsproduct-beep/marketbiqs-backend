-- Production RLS notes for MarketBiqs on Supabase
-- Application isolation is enforced in FastAPI via agency_id on every query.
-- RLS is defense-in-depth when PostgREST or anon keys are ever exposed.

-- 1) Dashboard → Project Settings → Database → Connection string (URI)
-- 2) Convert to async form in backend/.env:
--    DATABASE_URL=postgresql+asyncpg://postgres.[ref]:[PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres
--    or set SUPABASE_DB_PASSWORD + SUPABASE_PROJECT_REF
-- 3) Run schema.sql, restart API, optionally:
--    python -m scripts.migrate_sqlite_to_postgres

-- Recommended posture:
-- - FastAPI uses the Postgres pooler URL (bypasses RLS like service role)
-- - Frontend never talks to Postgres tables directly
-- - White-label API uses hashed keys + monthly_quota
-- - Usage meters gate scrapes and reports (scrape_units_used / reports_used)

-- Optional: set agency context per request if using Supabase Auth JWTs:
--   select set_config('app.agency_id', '<agency-uuid>', true);
