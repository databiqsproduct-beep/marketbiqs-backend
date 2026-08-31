-- ==============================================================================
-- MARKETBIQS — DEMO CREDENTIALS SEED MIGRATION (POSTGRES / SUPABASE / SQLITE)
-- Run this script directly in your database (Supabase SQL Editor, psql, Railway console)
-- to seed the standard demo agency owner and client lead accounts.
-- ==============================================================================

-- 1. SEED AGENCY USER (agency@marketbiqs.com / AgencyPass123!)
INSERT INTO users (id, email, full_name, hashed_password, is_active, created_at)
VALUES (
  'usr-agency-demo-001',
  'agency@marketbiqs.com',
  'Maarij Agency Admin',
  '$2b$12$dIqEBsPrI.aaBB6dddGY0O4Dv8eWqWSZyx1QQfoN9zuT5GcHN42F6',
  TRUE,
  NOW()
)
ON CONFLICT (email) DO UPDATE SET
  full_name = EXCLUDED.full_name,
  hashed_password = EXCLUDED.hashed_password,
  is_active = TRUE;

-- 2. SEED AGENCY WORKSPACE (Apex Growth Agency)
INSERT INTO agencies (
  id, name, slug, workspace_mode, plan, brand_color, brand_secondary,
  billing_status, billing_model, cancel_at_period_end,
  included_clients, client_pack_count, scrape_pack_count,
  reports_used, scrape_units_used, reports_quota, scrape_quota,
  budget_remaining_cents, byok_discount_percent, onboarding_completed,
  created_at, updated_at
)
VALUES (
  'agc-demo-apex-001',
  'Apex Growth Agency',
  'apex-growth-agency',
  'agency',
  'agency',
  '#0F766E',
  '#134E4A',
  'active',
  'plan',
  FALSE,
  20,
  0,
  0,
  0,
  0,
  100,
  1000,
  24900,
  0,
  TRUE,
  NOW(),
  NOW()
)
ON CONFLICT (slug) DO UPDATE SET
  name = EXCLUDED.name,
  billing_status = 'active',
  billing_model = 'plan',
  onboarding_completed = TRUE;

-- 3. LINK AGENCY OWNER MEMBERSHIP
INSERT INTO agency_members (id, agency_id, user_id, role, is_active, created_at)
VALUES (
  'mem-agency-owner-001',
  'agc-demo-apex-001',
  'usr-agency-demo-001',
  'owner',
  TRUE,
  NOW()
)
ON CONFLICT (agency_id, user_id) DO UPDATE SET
  role = 'owner',
  is_active = TRUE;

-- 4. SEED CLIENT USER (client@acmeretail.com / ClientPass123!)
INSERT INTO users (id, email, full_name, hashed_password, is_active, created_at)
VALUES (
  'usr-client-demo-001',
  'client@acmeretail.com',
  'Sarah Client Lead',
  '$2b$12$CVX0/Es0VM1DQwtBKDDVqO.etPteOT59SlMEfRjwj0JaQWD0trvDe',
  TRUE,
  NOW()
)
ON CONFLICT (email) DO UPDATE SET
  full_name = EXCLUDED.full_name,
  hashed_password = EXCLUDED.hashed_password,
  is_active = TRUE;

-- 5. LINK CLIENT USER MEMBERSHIP (Analyst / Client Lead)
INSERT INTO agency_members (id, agency_id, user_id, role, is_active, created_at)
VALUES (
  'mem-client-analyst-001',
  'agc-demo-apex-001',
  'usr-client-demo-001',
  'analyst',
  TRUE,
  NOW()
)
ON CONFLICT (agency_id, user_id) DO UPDATE SET
  role = 'analyst',
  is_active = TRUE;

-- 6. SEED CLIENT BRAND 1: Acme Retail Co.
INSERT INTO client_brands (
  id, agency_id, name, industry, niche, website, tagline,
  goals, delivery_emails, delivery_channel, delivery_schedule_cron,
  is_active, created_at, updated_at
)
VALUES (
  'cli-acme-retail-001',
  'agc-demo-apex-001',
  'Acme Retail Co.',
  'E-Commerce & Direct-to-Consumer',
  'Sustainable Apparel & Activewear',
  'https://acmeretail.example.com',
  'High-performance eco-activewear for modern athletes.',
  '[]'::json,
  '["client@acmeretail.com"]'::json,
  'email',
  '0 9 * * 1',
  TRUE,
  NOW(),
  NOW()
)
ON CONFLICT (id) DO UPDATE SET
  name = EXCLUDED.name,
  tagline = EXCLUDED.tagline,
  is_active = TRUE;

-- 7. SEED COMPETITORS FOR ACME RETAIL CO.
INSERT INTO competitors (
  id, agency_id, client_id, name, website, description,
  threat_level, is_tracking, is_pinned, overlap_score, feature_list, created_at
)
VALUES
(
  'cmp-rival-athletic-001',
  'agc-demo-apex-001',
  'cli-acme-retail-001',
  'Rival Athletic Co.',
  'https://rivalathletic.example.com',
  'Main direct rival focusing on budget gym wear',
  'high',
  TRUE,
  TRUE,
  88.5,
  '[]'::json,
  NOW()
),
(
  'cmp-ecoflex-001',
  'agc-demo-apex-001',
  'cli-acme-retail-001',
  'EcoFlex Performance',
  'https://ecoflex.example.com',
  'Eco-friendly premium yoga apparel',
  'medium',
  TRUE,
  FALSE,
  74.0,
  '[]'::json,
  NOW()
),
(
  'cmp-stride-001',
  'agc-demo-apex-001',
  'cli-acme-retail-001',
  'Stride Athletics',
  'https://stride.example.com',
  'Fast shipping, subscription gym apparel',
  'medium',
  TRUE,
  FALSE,
  65.0,
  '[]'::json,
  NOW()
)
ON CONFLICT (id) DO NOTHING;

-- 8. SEED CLIENT BRAND 2: Databiqs Analytics
INSERT INTO client_brands (
  id, agency_id, name, industry, niche, website, tagline,
  goals, delivery_emails, delivery_channel, delivery_schedule_cron,
  is_active, created_at, updated_at
)
VALUES (
  'cli-databiqs-002',
  'agc-demo-apex-001',
  'Databiqs Analytics',
  'SaaS & Big Data',
  'Product Intelligence & Behavioral Analytics',
  'https://www.databiqs.com',
  'Real-time product analytics for modern product teams.',
  '[]'::json,
  '[]'::json,
  'email',
  '0 9 * * 1',
  TRUE,
  NOW(),
  NOW()
)
ON CONFLICT (id) DO NOTHING;

-- Verification query:
SELECT u.email, u.full_name, a.name AS agency, m.role
FROM users u
JOIN agency_members m ON m.user_id = u.id
JOIN agencies a ON a.id = m.agency_id;
