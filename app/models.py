import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def uid() -> str:
    return str(uuid.uuid4())


class PlanType(str, enum.Enum):
    agency = "agency"
    medium = "medium"
    creator = "creator"


class WorkspaceMode(str, enum.Enum):
    agency = "agency"
    creator = "creator"


class MemberRole(str, enum.Enum):
    owner = "owner"
    admin = "admin"
    strategist = "strategist"
    analyst = "analyst"
    viewer = "viewer"


class DeliveryChannel(str, enum.Enum):
    email = "email"
    whatsapp = "whatsapp"
    both = "both"


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Agency(Base):
    __tablename__ = "agencies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    logo_url: Mapped[str | None] = mapped_column(String(500))
    brand_color: Mapped[str] = mapped_column(String(20), default="#0F766E")
    brand_secondary: Mapped[str] = mapped_column(String(20), default="#134E4A")
    report_footer: Mapped[str | None] = mapped_column(Text)
    workspace_mode: Mapped[WorkspaceMode] = mapped_column(Enum(WorkspaceMode), default=WorkspaceMode.agency)
    plan: Mapped[PlanType] = mapped_column(Enum(PlanType), default=PlanType.agency)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(120))
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(120))
    billing_status: Mapped[str] = mapped_column(String(40), default="trialing")
    included_clients: Mapped[int] = mapped_column(Integer, default=10)
    client_pack_count: Mapped[int] = mapped_column(Integer, default=0)
    reports_used: Mapped[int] = mapped_column(Integer, default=0)
    scrape_units_used: Mapped[int] = mapped_column(Integer, default=0)
    reports_quota: Mapped[int] = mapped_column(Integer, default=40)
    scrape_quota: Mapped[int] = mapped_column(Integer, default=5000)
    budget_remaining_cents: Mapped[int] = mapped_column(Integer, default=45000)
    byok_discount_percent: Mapped[int] = mapped_column(Integer, default=0)
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    members: Mapped[list["AgencyMember"]] = relationship(back_populates="agency", cascade="all, delete-orphan")
    clients: Mapped[list["ClientBrand"]] = relationship(back_populates="agency", cascade="all, delete-orphan")
    api_keys: Mapped[list["ApiKeyVault"]] = relationship(back_populates="agency", cascade="all, delete-orphan")
    integrations: Mapped[list["Integration"]] = relationship(back_populates="agency", cascade="all, delete-orphan")
    white_label_keys: Mapped[list["WhiteLabelApiKey"]] = relationship(
        back_populates="agency", cascade="all, delete-orphan"
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    memberships: Mapped[list["AgencyMember"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class AgencyMember(Base):
    __tablename__ = "agency_members"
    __table_args__ = (UniqueConstraint("agency_id", "user_id", name="uq_agency_user"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[MemberRole] = mapped_column(Enum(MemberRole), default=MemberRole.analyst)
    invited_email: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    agency: Mapped["Agency"] = relationship(back_populates="members")
    user: Mapped["User"] = relationship(back_populates="memberships")


class ClientBrand(Base):
    __tablename__ = "client_brands"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    industry: Mapped[str | None] = mapped_column(String(120))
    niche: Mapped[str | None] = mapped_column(String(255))
    tagline: Mapped[str | None] = mapped_column(String(500))
    website: Mapped[str | None] = mapped_column(String(500))
    logo_url: Mapped[str | None] = mapped_column(String(500))
    notes: Mapped[str | None] = mapped_column(Text)
    goals: Mapped[list] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    delivery_channel: Mapped[DeliveryChannel] = mapped_column(Enum(DeliveryChannel), default=DeliveryChannel.email)
    delivery_emails: Mapped[list] = mapped_column(JSON, default=list)
    delivery_whatsapp: Mapped[str | None] = mapped_column(String(40))
    delivery_schedule_cron: Mapped[str] = mapped_column(String(60), default="0 9 * * 1")
    last_delivered_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    agency: Mapped["Agency"] = relationship(back_populates="clients")
    competitors: Mapped[list["Competitor"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    insights: Mapped[list["Insight"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    reports: Mapped[list["Report"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    trends: Mapped[list["TrendSignal"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    sentiments: Mapped[list["SentimentRecord"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    tracking_jobs: Mapped[list["TrackingJob"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    chat_messages: Mapped[list["ChatMessage"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    jira_tickets: Mapped[list["JiraTicket"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    product_features: Mapped[list["ProductFeature"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    gap_reports: Mapped[list["GapReport"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    goal_alerts: Mapped[list["GoalAlert"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    comparison_rows: Mapped[list["FeatureComparison"]] = relationship(back_populates="client", cascade="all, delete-orphan")
    feature_tickets: Mapped[list["FeatureTicket"]] = relationship(back_populates="client", cascade="all, delete-orphan")


class Competitor(Base):
    __tablename__ = "competitors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    website: Mapped[str | None] = mapped_column(String(500))
    tagline: Mapped[str | None] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    headquarters: Mapped[str | None] = mapped_column(String(120))
    twitter_handle: Mapped[str | None] = mapped_column(String(120))
    instagram_handle: Mapped[str | None] = mapped_column(String(120))
    tiktok_handle: Mapped[str | None] = mapped_column(String(120))
    linkedin_url: Mapped[str | None] = mapped_column(String(500))
    facebook_page: Mapped[str | None] = mapped_column(String(255))
    meta_ads_query: Mapped[str | None] = mapped_column(String(255))
    overlap_score: Mapped[float] = mapped_column(Float, default=0.0)
    threat_level: Mapped[str] = mapped_column(String(20), default="medium")
    why_dangerous: Mapped[str | None] = mapped_column(Text)
    evidence_snippet: Mapped[str | None] = mapped_column(Text)
    feature_list: Mapped[list] = mapped_column(JSON, default=list)
    is_tracking: Mapped[bool] = mapped_column(Boolean, default=True)
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    last_scraped_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="competitors")
    snapshots: Mapped[list["CompetitorSnapshot"]] = relationship(
        back_populates="competitor", cascade="all, delete-orphan"
    )


class CompetitorSnapshot(Base):
    __tablename__ = "competitor_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    competitor_id: Mapped[str] = mapped_column(String(36), ForeignKey("competitors.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(60), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[str | None] = mapped_column(Text)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    competitor: Mapped["Competitor"] = relationship(back_populates="snapshots")


class Insight(Base):
    __tablename__ = "insights"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(20), default="medium")
    source_refs: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="insights")


class TrendSignal(Base):
    __tablename__ = "trend_signals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    topic: Mapped[str] = mapped_column(String(255), nullable=False)
    platform: Mapped[str] = mapped_column(String(60), nullable=False)
    velocity_score: Mapped[float] = mapped_column(Float, default=0.0)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="trends")


class SentimentRecord(Base):
    __tablename__ = "sentiment_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(60), nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    label: Mapped[str] = mapped_column(String(40), default="neutral")
    themes: Mapped[list] = mapped_column(JSON, default=list)
    sample_quotes: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="sentiments")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    period_label: Mapped[str] = mapped_column(String(120), default="Weekly")
    status: Mapped[str] = mapped_column(String(40), default="ready")
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    sections: Mapped[list] = mapped_column(JSON, default=list)
    pdf_path: Mapped[str | None] = mapped_column(String(500))
    white_labeled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="reports")


class TrackingJob(Base):
    __tablename__ = "tracking_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    job_type: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.pending)
    detail: Mapped[str | None] = mapped_column(Text)
    result_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="tracking_jobs")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="chat_messages")


class ApiKeyVault(Base):
    __tablename__ = "api_key_vault"
    __table_args__ = (UniqueConstraint("agency_id", "provider", name="uq_agency_provider"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(60), nullable=False)
    encrypted_key: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str | None] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    agency: Mapped["Agency"] = relationship(back_populates="api_keys")


class Integration(Base):
    __tablename__ = "integrations"
    __table_args__ = (UniqueConstraint("agency_id", "provider", name="uq_agency_integration"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(60), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    encrypted_credentials: Mapped[str | None] = mapped_column(Text)
    is_connected: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    agency: Mapped["Agency"] = relationship(back_populates="integrations")


class JiraTicket(Base):
    __tablename__ = "jira_tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    insight_id: Mapped[str | None] = mapped_column(String(36))
    jira_key: Mapped[str | None] = mapped_column(String(60))
    jira_url: Mapped[str | None] = mapped_column(String(500))
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="jira_tickets")


class DeliveryLog(Base):
    __tablename__ = "delivery_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    report_id: Mapped[str | None] = mapped_column(String(36))
    channel: Mapped[str] = mapped_column(String(40), nullable=False)
    recipient: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="sent")
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WhiteLabelApiKey(Base):
    __tablename__ = "white_label_api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)
    hashed_key: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    requests_used: Mapped[int] = mapped_column(Integer, default=0)
    monthly_quota: Mapped[int] = mapped_column(Integer, default=10000)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    agency: Mapped["Agency"] = relationship(back_populates="white_label_keys")


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(60), nullable=False)
    units: Mapped[int] = mapped_column(Integer, default=1)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProductFeature(Base):
    __tablename__ = "product_features"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(120), default="General")
    description: Mapped[str] = mapped_column(Text, default="")
    is_loved: Mapped[bool] = mapped_column(Boolean, default=False)
    is_wishlisted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="product_features")
    tickets: Mapped[list["FeatureTicket"]] = relationship(back_populates="feature", cascade="all, delete-orphan")


class FeatureComparison(Base):
    __tablename__ = "feature_comparisons"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    competitor_id: Mapped[str] = mapped_column(String(36), ForeignKey("competitors.id", ondelete="CASCADE"), index=True)
    competitor_name: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(120), default="General")
    our_status: Mapped[str] = mapped_column(String(40), default="parity")
    competitor_status: Mapped[str] = mapped_column(String(40), default="parity")
    note: Mapped[str] = mapped_column(Text, default="")
    how_competitor_leads: Mapped[str] = mapped_column(Text, default="")
    how_to_improve: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list] = mapped_column(JSON, default=list)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_strength: Mapped[str] = mapped_column(String(20), default="medium")
    feedback: Mapped[str | None] = mapped_column(String(20))
    is_contested_move: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="comparison_rows")


class GapReport(Base):
    __tablename__ = "gap_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    competitor_id: Mapped[str] = mapped_column(String(36), ForeignKey("competitors.id", ondelete="CASCADE"), index=True)
    competitor_name: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    leading: Mapped[list] = mapped_column(JSON, default=list)
    lagging: Mapped[list] = mapped_column(JSON, default=list)
    opportunities: Mapped[list] = mapped_column(JSON, default=list)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_strength: Mapped[str] = mapped_column(String(20), default="medium")
    feedback: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="gap_reports")


class GoalAlert(Base):
    __tablename__ = "goal_alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    goal: Mapped[str] = mapped_column(String(500), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    why_it_matters: Mapped[str] = mapped_column(Text, nullable=False)
    impact: Mapped[str] = mapped_column(String(20), default="medium")
    action: Mapped[str] = mapped_column(Text, nullable=False)
    content_draft: Mapped[str] = mapped_column(Text, default="")
    estimated_cost: Mapped[str] = mapped_column(String(120), default="")
    competitor_trigger: Mapped[str] = mapped_column(String(255), default="")
    citations: Mapped[list] = mapped_column(JSON, default=list)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_strength: Mapped[str] = mapped_column(String(20), default="medium")
    feedback: Mapped[str | None] = mapped_column(String(20))
    acted_on: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="goal_alerts")


class FeatureTicket(Base):
    __tablename__ = "feature_tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    feature_id: Mapped[str] = mapped_column(String(36), ForeignKey("product_features.id", ondelete="CASCADE"), index=True)
    heading: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    acceptance_criteria: Mapped[list] = mapped_column(JSON, default=list)
    priority: Mapped[str] = mapped_column(String(20), default="medium")
    ticket_type: Mapped[str] = mapped_column(String(40), default="story")
    labels: Mapped[list] = mapped_column(JSON, default=list)
    estimated_effort: Mapped[str] = mapped_column(String(80), default="")
    story_points: Mapped[int | None] = mapped_column(Integer)
    why_useful: Mapped[str] = mapped_column(Text, default="")
    competitor_context: Mapped[str] = mapped_column(Text, default="")
    evidence_links: Mapped[list] = mapped_column(JSON, default=list)
    parent_ticket_id: Mapped[str | None] = mapped_column(String(36))
    jira_key: Mapped[str | None] = mapped_column(String(60))
    jira_url: Mapped[str | None] = mapped_column(String(500))
    jira_epic_key: Mapped[str | None] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(40), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped["ClientBrand"] = relationship(back_populates="feature_tickets")
    feature: Mapped["ProductFeature"] = relationship(back_populates="tickets")


class InsightFeedback(Base):
    __tablename__ = "insight_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    rating: Mapped[str] = mapped_column(String(20), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class IntelEmbedding(Base):
    """RAG memory for GPT workspace. embedding JSON is optional; Postgres pgvector can mirror later."""

    __tablename__ = "intel_embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    agency_id: Mapped[str] = mapped_column(String(36), ForeignKey("agencies.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("client_brands.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(60), default="intel")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
