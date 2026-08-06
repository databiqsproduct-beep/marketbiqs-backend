from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, HttpUrl, field_validator


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    agency_id: str | None = None


class UserOut(BaseModel):
    id: str
    email: EmailStr
    full_name: str

    model_config = {"from_attributes": True}


class AgencyOut(BaseModel):
    id: str
    name: str
    slug: str
    logo_url: str | None = None
    brand_color: str
    brand_secondary: str
    report_footer: str | None = None
    workspace_mode: str
    plan: str
    billing_status: str
    included_clients: int
    client_pack_count: int
    reports_used: int
    scrape_units_used: int
    reports_quota: int
    scrape_quota: int
    budget_remaining_cents: int
    byok_discount_percent: int
    onboarding_completed: bool

    model_config = {"from_attributes": True}


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str
    agency_name: str
    workspace_mode: str = "agency"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class BootstrapRequest(BaseModel):
    agency_name: str = Field(min_length=1, max_length=200)
    workspace_mode: str = "agency"
    full_name: str | None = Field(default=None, max_length=255)


class OnboardingRequest(BaseModel):
    brand_color: str | None = None
    logo_url: str | None = None
    report_footer: str | None = None
    first_client_name: str | None = None
    first_client_website: str | None = None
    first_competitor_name: str | None = None
    first_competitor_website: str | None = None


class MemberInvite(BaseModel):
    email: EmailStr
    full_name: str
    role: str = "analyst"
    # Optional legacy field — ignored. Invites use Supabase Auth email invite.


class MemberOut(BaseModel):
    id: str
    role: str
    is_active: bool
    user: UserOut

    model_config = {"from_attributes": True}


class ClientCreate(BaseModel):
    name: str
    industry: str | None = None
    website: str | None = None
    logo_url: str | None = None
    notes: str | None = None
    delivery_channel: str = "email"
    delivery_emails: list[str] = Field(default_factory=list)
    delivery_whatsapp: str | None = None
    delivery_schedule_cron: str = "0 9 * * 1"


class ClientUpdate(BaseModel):
    name: str | None = None
    industry: str | None = None
    website: str | None = None
    logo_url: str | None = None
    notes: str | None = None
    is_active: bool | None = None
    delivery_channel: str | None = None
    delivery_emails: list[str] | None = None
    delivery_whatsapp: str | None = None
    delivery_schedule_cron: str | None = None


class ClientOut(BaseModel):
    id: str
    agency_id: str
    name: str
    industry: str | None = None
    niche: str | None = None
    tagline: str | None = None
    website: str | None = None
    logo_url: str | None = None
    notes: str | None = None
    goals: list[Any] = Field(default_factory=list)
    is_active: bool
    delivery_channel: str
    delivery_emails: list[Any] = Field(default_factory=list)
    delivery_whatsapp: str | None = None
    delivery_schedule_cron: str
    last_delivered_at: datetime | None = None
    created_at: datetime
    rivals_count: int = 0
    features_count: int = 0
    reports_count: int = 0
    tickets_count: int = 0
    alerts_open: int = 0

    model_config = {"from_attributes": True}

    @field_validator("goals", "delivery_emails", mode="before")
    @classmethod
    def empty_list(cls, value: Any) -> Any:
        return value or []


class CompetitorCreate(BaseModel):
    name: str
    website: str | None = None
    twitter_handle: str | None = None
    instagram_handle: str | None = None
    tiktok_handle: str | None = None
    linkedin_url: str | None = None
    facebook_page: str | None = None
    meta_ads_query: str | None = None


class CompetitorOut(BaseModel):
    id: str
    client_id: str
    name: str
    website: str | None = None
    tagline: str | None = None
    description: str | None = None
    headquarters: str | None = None
    twitter_handle: str | None = None
    instagram_handle: str | None = None
    tiktok_handle: str | None = None
    linkedin_url: str | None = None
    facebook_page: str | None = None
    meta_ads_query: str | None = None
    overlap_score: float = 0
    threat_level: str = "medium"
    why_dangerous: str | None = None
    evidence_snippet: str | None = None
    feature_list: list[Any] = Field(default_factory=list)
    is_tracking: bool
    is_pinned: bool = False
    last_scraped_at: datetime | None = None

    model_config = {"from_attributes": True}

    @field_validator("feature_list", mode="before")
    @classmethod
    def empty_features(cls, value: Any) -> Any:
        return value or []

    @field_validator("overlap_score", mode="before")
    @classmethod
    def zero_score(cls, value: Any) -> Any:
        return value or 0

    @field_validator("threat_level", mode="before")
    @classmethod
    def default_threat(cls, value: Any) -> Any:
        return value or "medium"

    @field_validator("is_pinned", mode="before")
    @classmethod
    def default_pinned(cls, value: Any) -> Any:
        return bool(value)


class SnapshotOut(BaseModel):
    id: str
    competitor_id: str
    source: str
    payload: dict
    summary: str | None = None
    scraped_at: datetime

    model_config = {"from_attributes": True}


class InsightOut(BaseModel):
    id: str
    client_id: str
    category: str
    title: str
    body: str
    priority: str
    source_refs: list[Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class TrendOut(BaseModel):
    id: str
    client_id: str
    topic: str
    platform: str
    velocity_score: float
    summary: str
    keywords: list[Any]
    detected_at: datetime

    model_config = {"from_attributes": True}


class SentimentOut(BaseModel):
    id: str
    client_id: str
    subject: str
    source: str
    score: float
    label: str
    themes: list[Any]
    sample_quotes: list[Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class ReportOut(BaseModel):
    id: str
    client_id: str
    title: str
    period_label: str
    status: str
    summary: str
    sections: list[Any]
    pdf_path: str | None = None
    white_labeled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ReportGenerateRequest(BaseModel):
    period_label: str = "Weekly"
    include_trends: bool = True
    include_sentiment: bool = True
    include_competitors: bool = True


class ChatRequest(BaseModel):
    message: str


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ByokUpsert(BaseModel):
    provider: str
    api_key: str
    label: str | None = None


class ByokOut(BaseModel):
    id: str
    provider: str
    label: str | None = None
    is_active: bool
    key_hint: str
    created_at: datetime


class BudgetOut(BaseModel):
    plan: str
    billing_status: str
    base_price_cents: int
    client_pack_count: int
    client_pack_price_cents: int
    included_clients: int
    max_clients: int
    active_clients: int
    reports_used: int
    reports_quota: int
    scrape_units_used: int
    scrape_quota: int
    budget_remaining_cents: int
    byok_discount_percent: int
    estimated_monthly_cents: int


class CheckoutRequest(BaseModel):
    add_client_packs: int = 0
    success_url: str | None = None
    cancel_url: str | None = None


class JiraConnectRequest(BaseModel):
    base_url: HttpUrl
    email: EmailStr
    api_token: str
    project_key: str
    epic_name_field: str | None = None


class JiraTicketCreate(BaseModel):
    title: str
    description: str
    insight_id: str | None = None


class JiraTicketOut(BaseModel):
    id: str
    client_id: str
    jira_key: str | None = None
    jira_url: str | None = None
    title: str
    description: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DeliveryRequest(BaseModel):
    report_id: str | None = None
    channel: str | None = None
    message: str | None = None


class DeliveryLogOut(BaseModel):
    id: str
    client_id: str
    report_id: str | None = None
    channel: str
    recipient: str
    status: str
    detail: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class WhiteLabelKeyCreate(BaseModel):
    name: str
    monthly_quota: int = 10000


class WhiteLabelKeyOut(BaseModel):
    id: str
    name: str
    key_prefix: str
    api_key: str | None = None
    is_active: bool
    requests_used: int
    monthly_quota: int
    created_at: datetime


class DashboardOut(BaseModel):
    agency: AgencyOut
    clients_count: int
    competitors_count: int
    reports_count: int
    open_insights: int
    recent_trends: list[TrendOut]
    recent_insights: list[InsightOut]
    usage: BudgetOut
    roi: dict[str, Any] = Field(default_factory=dict)
    charts: dict[str, Any] = Field(default_factory=dict)
    portfolio: list[dict[str, Any]] = Field(default_factory=list)


class AgencyBrandUpdate(BaseModel):
    name: str | None = None
    logo_url: str | None = None
    brand_color: str | None = None
    brand_secondary: str | None = None
    report_footer: str | None = None
    workspace_mode: str | None = None
