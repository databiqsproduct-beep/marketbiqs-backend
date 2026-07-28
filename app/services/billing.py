from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Agency, ClientBrand, UsageEvent
from app.schemas import BudgetOut

settings = get_settings()

REPORTS_PER_PACK = 8
SCRAPES_PER_PACK = 800


def sync_pack_quotas(agency: Agency) -> None:
    packs = max(0, agency.client_pack_count or 0)
    agency.reports_quota = settings.included_reports_per_month + packs * REPORTS_PER_PACK
    agency.scrape_quota = settings.included_scrape_units + packs * SCRAPES_PER_PACK
    base = settings.agency_base_price + packs * settings.client_pack_price
    discount = agency.byok_discount_percent or 0
    agency.budget_remaining_cents = int(base * (100 - discount) / 100)


def compute_budget(agency: Agency, active_clients: int) -> BudgetOut:
    sync_pack_quotas(agency)
    max_clients = agency.included_clients + agency.client_pack_count
    pack_total = agency.client_pack_count * settings.client_pack_price
    base = settings.agency_base_price
    discount = agency.byok_discount_percent or 0
    estimated = int((base + pack_total) * (100 - discount) / 100)
    return BudgetOut(
        plan=agency.plan.value if hasattr(agency.plan, "value") else str(agency.plan),
        billing_status=agency.billing_status,
        base_price_cents=base,
        client_pack_count=agency.client_pack_count,
        client_pack_price_cents=settings.client_pack_price,
        included_clients=agency.included_clients,
        max_clients=max_clients,
        active_clients=active_clients,
        reports_used=agency.reports_used,
        reports_quota=agency.reports_quota,
        scrape_units_used=agency.scrape_units_used,
        scrape_quota=agency.scrape_quota,
        budget_remaining_cents=agency.budget_remaining_cents,
        byok_discount_percent=discount,
        estimated_monthly_cents=estimated,
    )


async def ensure_client_capacity(db: AsyncSession, agency: Agency) -> None:
    count = (
        await db.execute(
            select(func.count())
            .select_from(ClientBrand)
            .where(ClientBrand.agency_id == agency.id, ClientBrand.is_active.is_(True))
        )
    ).scalar_one()
    max_clients = agency.included_clients + agency.client_pack_count
    if count >= max_clients:
        raise ValueError(
            f"Client limit reached ({max_clients}). Purchase a per-client add-on pack to continue."
        )


def apply_byok_discount(agency: Agency, providers: list[str]) -> None:
    mapping = {"groq": 10, "apify": 8, "serpapi": 5, "firecrawl": 5}
    discount = min(35, sum(mapping.get(p, 0) for p in providers))
    agency.byok_discount_percent = discount
    base = settings.agency_base_price + agency.client_pack_count * settings.client_pack_price
    agency.budget_remaining_cents = int(base * (100 - discount) / 100)


async def create_checkout_session(
    db: AsyncSession,
    agency: Agency,
    add_client_packs: int,
    success_url: str,
    cancel_url: str,
) -> dict:
    packs = max(0, add_client_packs)
    if not settings.stripe_secret_key:
        agency.client_pack_count += packs
        agency.billing_status = "active"
        sync_pack_quotas(agency)
        db.add(
            UsageEvent(
                agency_id=agency.id,
                event_type="pack_purchase_local",
                units=packs,
                meta={"reports_quota": agency.reports_quota, "scrape_quota": agency.scrape_quota},
            )
        )
        return {
            "mode": "local",
            "url": success_url,
            "message": f"Packs applied locally. Quotas now reports {agency.reports_quota}, scrapes {agency.scrape_quota}.",
            "client_pack_count": agency.client_pack_count,
            "reports_quota": agency.reports_quota,
            "scrape_quota": agency.scrape_quota,
        }

    import stripe

    stripe.api_key = settings.stripe_secret_key
    line_items = []
    if settings.stripe_agency_price_id:
        line_items.append({"price": settings.stripe_agency_price_id, "quantity": 1})
    if packs > 0 and settings.stripe_client_pack_price_id:
        line_items.append({"price": settings.stripe_client_pack_price_id, "quantity": packs})
    if not line_items:
        raise ValueError("Stripe price IDs are not configured. Set STRIPE_AGENCY_PRICE_ID and STRIPE_CLIENT_PACK_PRICE_ID.")

    customer = agency.stripe_customer_id
    if not customer:
        created = stripe.Customer.create(name=agency.name, metadata={"agency_id": agency.id})
        customer = created["id"]
        agency.stripe_customer_id = customer

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer,
        line_items=line_items,
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"agency_id": agency.id, "add_client_packs": str(packs)},
    )
    return {"mode": "stripe", "url": session.url, "session_id": session.id}


def apply_paid_packs(agency: Agency, packs: int) -> None:
    agency.client_pack_count += max(0, packs)
    agency.billing_status = "active"
    sync_pack_quotas(agency)
