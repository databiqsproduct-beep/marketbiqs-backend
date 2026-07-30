from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Biqs Agency Intelligence"
    app_env: str = "development"
    secret_key: str = "change-me-in-production-biqs-agency-secret"
    access_token_expire_minutes: int = 60 * 24 * 7
    algorithm: str = "HS256"
    # Prefer Supabase Postgres. Example:
    # DATABASE_URL=postgresql+asyncpg://postgres.[ref]:[PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres
    database_url: str = ""
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "https://marketbiqsfrontend-production.up.railway.app"
    )
    frontend_url: str = "https://marketbiqsfrontend-production.up.railway.app"
    api_base_url: str = "http://localhost:8000"

    # Primary Supabase access (same vars for localhost + production)
    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_secret_key: str = ""
    # Legacy aliases (optional fallbacks)
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_db_url: str = ""
    supabase_db_password: str = ""
    supabase_project_ref: str = "wcseztlbcajqegmzpzqm"
    # Must match Supabase Dashboard → Project Settings → Database → Connection string (pooler host).
    # Wrong region/cluster causes: "Tenant or user not found" and login/register 500s.
    supabase_db_region: str = "ap-northeast-2"
    # Full pooler hostname from dashboard, e.g. aws-1-ap-northeast-2.pooler.supabase.com
    # Do not guess aws-0 — new projects often land on aws-1 / aws-2.
    supabase_pooler_host: str = "aws-1-ap-northeast-2.pooler.supabase.com"

    def resolved_publishable_key(self) -> str:
        return (self.supabase_publishable_key or self.supabase_anon_key or "").strip()

    def resolved_secret_key(self) -> str:
        return (self.supabase_secret_key or self.supabase_service_role_key or "").strip()

    @property
    def supabase_ready(self) -> bool:
        return bool((self.supabase_url or "").strip() and self.resolved_secret_key())

    groq_api_key: str = ""
    apify_key: str = ""
    serp_api: str = ""
    firecrawl_api_key: str = ""

    stripe_secret_key: str = ""
    stripe_publishable_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_agency_price_id: str = ""
    stripe_client_pack_price_id: str = ""

    resend_api_key: str = ""
    email_from: str = "Biqs <reports@biqs.ai>"

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_from: str = ""

    scrape_interval_hours: int = 24
    agency_base_price: int = 45000
    client_pack_price: int = 4900
    included_clients: int = 10
    included_reports_per_month: int = 40
    included_scrape_units: int = 5000

    encryption_key: str = "biqs-fernet-key-change-in-prod-32b!"

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        frontend = (self.frontend_url or "").strip().rstrip("/")
        if frontend and frontend not in origins:
            origins.append(frontend)
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()
