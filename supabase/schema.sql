-- MarketBiqs / Supabase production helpers
-- Run in Supabase SQL Editor after creating the project.
-- FastAPI create_all owns table DDL; this enables RLS + vector memory.

create extension if not exists "pgcrypto";
create extension if not exists "vector";

-- Memory table for RAG / GPT workspace (also created by SQLAlchemy)
create table if not exists intel_embeddings (
  id text primary key,
  agency_id text not null,
  client_id text not null,
  source text not null default 'intel',
  content text not null,
  embedding jsonb,
  meta jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

create index if not exists intel_embeddings_agency_client_idx
  on intel_embeddings (agency_id, client_id);

-- Enable RLS on core tenant tables (service-role / FastAPI bypasses RLS)
do $$
declare
  t text;
begin
  foreach t in array array[
    'agencies','users','agency_members','client_brands','competitors',
    'competitor_snapshots','product_features','feature_comparisons',
    'gap_reports','goal_alerts','feature_tickets','biqs_tickets','insight_feedback',
    'reports','delivery_logs','usage_events','integrations','jira_tickets',
    'api_key_vaults','white_label_api_keys','insights','trend_signals',
    'sentiment_records','tracking_jobs','intel_embeddings','chat_messages'
  ]
  loop
    begin
      execute format('alter table if exists %I enable row level security', t);
    exception when undefined_table then
      null;
    end;
  end loop;
end $$;

-- Defense-in-depth policies for PostgREST / anon keys (FastAPI uses direct DB URL)
-- Set request.jwt.claims or app.agency_id via SET LOCAL when using Supabase Auth later.

do $$
declare
  t text;
begin
  foreach t in array array[
    'client_brands','competitors','competitor_snapshots','product_features',
    'feature_comparisons','gap_reports','goal_alerts','feature_tickets',
    'biqs_tickets','insight_feedback','reports','delivery_logs','insights','trend_signals',
    'sentiment_records','tracking_jobs','intel_embeddings','chat_messages',
    'jira_tickets','usage_events'
  ]
  loop
    begin
      execute format('drop policy if exists tenant_select on %I', t);
      execute format(
        'create policy tenant_select on %I for select using (
           agency_id::text = coalesce(current_setting(''app.agency_id'', true), '''')
           or current_setting(''role'', true) = ''service_role''
         )',
        t
      );
    exception when undefined_table then
      null;
    when undefined_column then
      null;
    end;
  end loop;
end $$;
