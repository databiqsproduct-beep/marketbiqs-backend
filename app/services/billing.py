import asyncio
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Agency, ClientBrand, PlanType, Report
from app.schemas import (
    BillingCatalogOut,
    BudgetOut,
    PackCatalogOut,
    PaygCatalogOut,
    PlanCatalogOut,
    ScrapePackCatalogOut,
    UsageLineOut,
)

settings = get_settings()

REPORTS_PER_PACK = 8
SCRAPES_PER_PACK = 800
CLIENTS_PER_PACK = 1
SCRAPE_PACK_OPTIONS = [0, 100, 200, 500, 1000, 2000, 5000]
PAYG_CLIENT_CAP = 100
PAYG_REPORT_CAP = 100_000
PAYG_SCRAPE_CAP = 10_000_000


def _as_dict(value: Any) -> dict[str, Any]:
    """Normalize Stripe objects and mappings to a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict_recursive = getattr(value, "to_dict_recursive", None)
    if callable(to_dict_recursive):
        data = to_dict_recursive()
        return data if isinstance(data, dict) else {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        return data if isinstance(data, dict) else {}
    return {}


def _plan_id(agency: Agency) -> str:
    value = agency.plan.value if hasattr(agency.plan, "value") else str(agency.plan)
    return "creator" if value == "creator" else "agency"


def is_agency_workspace(agency: Agency) -> bool:
    mode = agency.workspace_mode.value if hasattr(agency.workspace_mode, "value") else str(agency.workspace_mode or "agency")
    return mode != "creator" and _plan_id(agency) != "creator"


def is_payg(agency: Agency) -> bool:
    return (
        (agency.billing_model or "plan") == "payg"
        and is_agency_workspace(agency)
        and bool(agency.stripe_subscription_id)
    )


def normalize_billing_model(agency: Agency) -> None:
    """Drop unpaid PAYG marks so abandoned checkout cannot zero client seats."""
    if (agency.billing_model or "plan") != "payg":
        return
    if is_agency_workspace(agency) and agency.stripe_subscription_id:
        return
    agency.billing_model = "plan"
    sync_entitlements(agency)


def plan_values(plan_id: str) -> dict[str, Any]:
    if plan_id == "creator":
        return {
            "id": "creator",
            "name": "Individual",
            "price_cents": settings.creator_base_price,
            "included_clients": settings.creator_included_clients,
            "included_reports": settings.creator_included_reports_per_month,
            "included_scrapes": settings.creator_included_scrape_units,
            "stripe_price_id": settings.stripe_individual_price_id,
        }
    return {
        "id": "agency",
        "name": "Agency",
        "price_cents": settings.agency_base_price,
        "included_clients": settings.included_clients,
        "included_reports": settings.included_reports_per_month,
        "included_scrapes": settings.included_scrape_units,
        "stripe_price_id": settings.stripe_agency_price_id,
    }


def billing_catalog(plan_id: str | None = None) -> BillingCatalogOut:
    plans = []
    wanted = (plan_id,) if plan_id in {"agency", "creator"} else ("agency", "creator")
    for current_id in wanted:
        values = plan_values(current_id)
        plans.append(
            PlanCatalogOut(
                id=values["id"],
                name=values["name"],
                price_cents=values["price_cents"],
                included_clients=values["included_clients"],
                included_reports=values["included_reports"],
                included_scrapes=values["included_scrapes"],
                checkout_ready=bool(settings.stripe_secret_key and values["stripe_price_id"]),
            )
        )
    return BillingCatalogOut(
        plans=plans,
        pack=PackCatalogOut(
            name="Client add-on pack",
            price_cents=settings.client_pack_price,
            extra_clients=CLIENTS_PER_PACK,
            extra_reports=REPORTS_PER_PACK,
            extra_scrapes=SCRAPES_PER_PACK,
            checkout_ready=bool(settings.stripe_secret_key and settings.stripe_client_pack_price_id),
        ),
        scrape_pack=ScrapePackCatalogOut(
            name="Scrape units",
            price_cents=settings.scrape_pack_price,
            units=settings.scrape_pack_units,
            options=SCRAPE_PACK_OPTIONS,
            checkout_ready=bool(settings.stripe_secret_key and settings.stripe_scrape_pack_price_id),
        ),
        payg=PaygCatalogOut(
            name="Agency PAYG",
            checkout_ready=bool(settings.stripe_secret_key and settings.stripe_payg_price_id),
            client_cents=settings.payg_client_cents,
            intel_run_cents=settings.payg_intel_run_cents,
            report_cents=settings.payg_report_cents,
            scrape_unit_cents=settings.payg_scrape_unit_cents,
        ),
    )


def extra_scrape_lots_needed(used: int, included: int, lot_size: int | None = None) -> int:
    """How many $5 scrape lots are required to cover usage above included units."""
    size = max(1, int(lot_size or settings.scrape_pack_units))
    extra = max(0, int(used or 0) - max(0, int(included or 0)))
    if extra == 0:
        return 0
    return min(200, (extra + size - 1) // size)


def _scrape_lots(units: int) -> int:
    size = max(1, settings.scrape_pack_units)
    units = max(0, int(units or 0))
    if units % size:
        raise ValueError(f"Scrape units must be in lots of {size}.")
    return min(200, units // size)


def payg_rates() -> PaygCatalogOut:
    return PaygCatalogOut(
        name="Agency PAYG",
        checkout_ready=bool(settings.stripe_secret_key and settings.stripe_payg_price_id),
        client_cents=settings.payg_client_cents,
        intel_run_cents=settings.payg_intel_run_cents,
        report_cents=settings.payg_report_cents,
        scrape_unit_cents=settings.payg_scrape_unit_cents,
    )


def payg_usage_lines(
    *,
    clients: int,
    intel_runs: int,
    reports: int,
    scrapes: int,
) -> list[UsageLineOut]:
    rates = payg_rates()
    rows = [
        ("clients", "Clients", max(0, int(clients or 0)), rates.client_cents),
        ("intel_runs", "Intel runs", max(0, int(intel_runs or 0)), rates.intel_run_cents),
        ("reports", "Reports", max(0, int(reports or 0)), rates.report_cents),
        ("scrapes", "Scrape units", max(0, int(scrapes or 0)), rates.scrape_unit_cents),
    ]
    return [
        UsageLineOut(
            key=key,
            label=label,
            quantity=qty,
            unit_cents=unit,
            amount_cents=qty * unit,
        )
        for key, label, qty, unit in rows
    ]


def payg_usage_total(
    *,
    clients: int,
    intel_runs: int,
    reports: int,
    scrapes: int,
) -> int:
    return sum(
        line.amount_cents
        for line in payg_usage_lines(
            clients=clients, intel_runs=intel_runs, reports=reports, scrapes=scrapes
        )
    )


def sync_entitlements(agency: Agency) -> None:
    """Derive quotas from the selected base plan and verified pack quantity."""
    values = plan_values(_plan_id(agency))
    packs = max(0, agency.client_pack_count or 0)
    scrape_lots = max(0, agency.scrape_pack_count or 0)
    payg = is_payg(agency)
    if payg:
        agency.included_clients = PAYG_CLIENT_CAP
        agency.reports_quota = PAYG_REPORT_CAP
        agency.scrape_quota = PAYG_SCRAPE_CAP
        agency.budget_remaining_cents = 0
        return
    agency.included_clients = values["included_clients"]
    agency.reports_quota = values["included_reports"] + packs * REPORTS_PER_PACK
    agency.scrape_quota = (
        values["included_scrapes"]
        + packs * SCRAPES_PER_PACK
        + scrape_lots * settings.scrape_pack_units
    )
    agency.budget_remaining_cents = max(
        0,
        values["price_cents"]
        + packs * settings.client_pack_price
        + scrape_lots * settings.scrape_pack_price,
    )


# Backwards-compatible alias used by existing BYOK and bootstrap paths.
sync_pack_quotas = sync_entitlements


def included_scrape_allowance(agency: Agency) -> int:
    """Scrape units included with the plan / client packs, before $5 extra lots."""
    if is_payg(agency):
        return PAYG_SCRAPE_CAP
    values = plan_values(_plan_id(agency))
    packs = max(0, agency.client_pack_count or 0)
    return values["included_scrapes"] + packs * SCRAPES_PER_PACK


def compute_budget(agency: Agency, active_clients: int, *, intel_runs_used: int = 0) -> BudgetOut:
    normalize_billing_model(agency)
    payg = is_payg(agency)
    if payg:
        sync_entitlements(agency)
    values = plan_values(_plan_id(agency))
    packs = max(0, agency.client_pack_count or 0)
    scrape_lots = max(0, agency.scrape_pack_count or 0)
    extra_scrapes = scrape_lots * settings.scrape_pack_units
    payg = is_payg(agency)
    intel_runs = max(0, int(intel_runs_used or 0))
    usage_lines = (
        payg_usage_lines(
            clients=active_clients,
            intel_runs=intel_runs,
            reports=agency.reports_used or 0,
            scrapes=agency.scrape_units_used or 0,
        )
        if payg
        else []
    )
    if payg:
        base_price = 0
        estimated = sum(line.amount_cents for line in usage_lines)
        max_clients = PAYG_CLIENT_CAP
        included_scrapes = 0
        overage_lots = 0
        extra_scrapes = 0
        scrape_quota = PAYG_SCRAPE_CAP
    else:
        base_price = values["price_cents"]
        included_scrapes = included_scrape_allowance(agency)
        overage_lots = extra_scrape_lots_needed(agency.scrape_units_used or 0, included_scrapes)
        billed_lots = max(scrape_lots, overage_lots)
        extra_scrapes = billed_lots * settings.scrape_pack_units
        estimated = (
            base_price
            + packs * settings.client_pack_price
            + billed_lots * settings.scrape_pack_price
        )
        max_clients = agency.included_clients + packs
        scrape_quota = max(agency.scrape_quota or 0, included_scrapes + extra_scrapes)
        scrape_lots = billed_lots
    list_price = estimated
    percent = max(0, int(agency.byok_discount_percent or 0))
    estimated = discounted_cents(list_price, percent)
    return BudgetOut(
        plan=values["id"],
        plan_name="PAYG" if payg else values["name"],
        billing_status=agency.billing_status,
        cancel_at_period_end=bool(agency.cancel_at_period_end),
        billing_period_start=agency.billing_period_start,
        billing_period_end=agency.billing_period_end,
        base_price_cents=base_price,
        client_pack_count=packs,
        client_pack_price_cents=settings.client_pack_price,
        scrape_pack_count=scrape_lots,
        scrape_pack_price_cents=settings.scrape_pack_price,
        scrape_pack_units=settings.scrape_pack_units,
        extra_scrape_units=extra_scrapes,
        included_clients=agency.included_clients,
        max_clients=max_clients,
        active_clients=active_clients,
        reports_used=agency.reports_used,
        reports_quota=agency.reports_quota,
        scrape_units_used=agency.scrape_units_used,
        scrape_quota=scrape_quota,
        included_scrape_units=0 if payg else included_scrapes,
        scrape_overage_lots=overage_lots,
        intel_runs_used=intel_runs,
        usage_lines=usage_lines,
        payg_rates=payg_rates() if payg else None,
        budget_remaining_cents=estimated,
        byok_discount_percent=percent,
        list_price_cents=list_price,
        estimated_monthly_cents=estimated,
        amount_paid_cents=0,
        upcoming_invoice_cents=estimated,
        stripe_configured=bool(settings.stripe_secret_key),
        has_subscription=bool(agency.stripe_subscription_id),
        billing_model="payg" if payg else "plan",
        payg_available=is_agency_workspace(agency),
        catalog=billing_catalog(values["id"]),
    )


async def _active_client_count(db: AsyncSession, agency_id: str) -> int:
    return (
        await db.execute(
            select(func.count())
            .select_from(ClientBrand)
            .where(ClientBrand.agency_id == agency_id, ClientBrand.is_active.is_(True))
        )
    ).scalar_one()


async def count_intel_runs(db: AsyncSession, agency: Agency) -> int:
    """Successful auto-intel runs this period (each run writes one 'AI Auto Brief' report)."""
    filters = [
        Report.agency_id == agency.id,
        Report.period_label == "AI Auto Brief",
    ]
    start = agency.billing_period_start
    if start is not None:
        if getattr(start, "tzinfo", None) is not None:
            start = start.replace(tzinfo=None)
        filters.append(Report.created_at >= start)
    return int(
        (
            await db.execute(select(func.count()).select_from(Report).where(*filters))
        ).scalar_one()
        or 0
    )


async def sync_payg_seats(db: AsyncSession, agency: Agency, *, needed: int | None = None) -> bool:
    """Legacy no-op: PAYG no longer sells $49 client packs."""
    return False


async def release_payg_seat(db: AsyncSession, agency: Agency) -> None:
    """Legacy no-op: PAYG bills active clients at month end, not Stripe seats."""
    return


async def sync_payg_usage_charges(
    agency: Agency,
    *,
    clients: int,
    intel_runs: int,
    reports: int,
    scrapes: int,
) -> None:
    """Replace pending Stripe invoice items with this period's usage (billed at month end)."""
    if not is_payg(agency) or not agency.stripe_customer_id or not agency.stripe_subscription_id:
        return
    if not settings.stripe_secret_key:
        return
    stripe = _stripe()
    pending = await _call(stripe.InvoiceItem.list, customer=agency.stripe_customer_id, pending=True, limit=100)
    items = pending.data if getattr(pending, "data", None) is not None else (_as_dict(pending).get("data") or [])
    for item in items:
        row = _as_dict(item)
        meta = _as_dict(row.get("metadata"))
        if meta.get("marketbiqs_usage") != "1":
            continue
        item_id = row.get("id")
        if item_id:
            await _call(stripe.InvoiceItem.delete, item_id)
    percent = max(0, int(agency.byok_discount_percent or 0))
    for line in payg_usage_lines(
        clients=clients, intel_runs=intel_runs, reports=reports, scrapes=scrapes
    ):
        amount = discounted_cents(line.amount_cents, percent)
        if amount <= 0:
            continue
        note = f" after {percent}% BYOK" if percent else ""
        await _call(
            stripe.InvoiceItem.create,
            customer=agency.stripe_customer_id,
            subscription=agency.stripe_subscription_id,
            currency="usd",
            amount=amount,
            description=f"PAYG {line.label}: {line.quantity} × ${line.unit_cents / 100:.2f}{note}",
            metadata={"marketbiqs_usage": "1", "line": line.key},
        )


async def migrate_payg_off_fixed_packs(agency: Agency) -> None:
    """Existing $49 PAYG subs keep the cycle but stop charging client/scrape packs."""
    if not is_payg(agency) or not agency.stripe_subscription_id or not settings.stripe_secret_key:
        return
    stripe = _stripe()
    try:
        await _call(
            stripe.Subscription.modify,
            agency.stripe_subscription_id,
            metadata={"agency_id": agency.id, "plan": "payg", "billing_model": "payg"},
        )
    except Exception:
        pass
    if settings.stripe_payg_price_id:
        try:
            sub = await _call(
                stripe.Subscription.retrieve,
                agency.stripe_subscription_id,
                expand=["items.data.price"],
            )
            apply_subscription(agency, sub)
            item_rows = (_as_dict(_as_dict(sub).get("items")).get("data") or [])
            has_payg_price = any(
                _price_id(item) == settings.stripe_payg_price_id for item in item_rows
            )
            if not has_payg_price:
                await _call(
                    stripe.SubscriptionItem.create,
                    subscription=agency.stripe_subscription_id,
                    price=settings.stripe_payg_price_id,
                    quantity=1,
                    proration_behavior="none",
                    idempotency_key=f"marketbiqs-payg-anchor-{agency.id}",
                )
        except Exception:
            pass
    if agency.client_pack_count:
        try:
            await update_pack_quantity(agency, 0)
        except Exception:
            pass
    if agency.scrape_pack_count:
        try:
            await update_scrape_pack_quantity(agency, 0)
        except Exception:
            pass


async def load_stripe_spend(agency: Agency, *, estimated: int | None = None) -> dict[str, int]:
    fallback = max(0, int(estimated if estimated is not None else agency.budget_remaining_cents or 0))
    empty = {"amount_paid_cents": 0, "upcoming_invoice_cents": fallback}
    if not agency.stripe_customer_id or not settings.stripe_secret_key:
        return empty
    try:
        stripe = _stripe()
        invoices = await _call(
            stripe.Invoice.list,
            customer=agency.stripe_customer_id,
            status="paid",
            limit=24,
        )
    except Exception:
        return empty
    items = invoices.data if getattr(invoices, "data", None) is not None else (_as_dict(invoices).get("data") or [])
    paid = 0
    for invoice in items:
        row = _as_dict(invoice)
        paid += max(0, int(row.get("amount_paid") or 0))
    upcoming = fallback
    try:
        preview = await _call(
            stripe.Invoice.create_preview,
            customer=agency.stripe_customer_id,
            subscription=agency.stripe_subscription_id,
        )
        preview_data = _as_dict(preview)
        upcoming = max(0, int(preview_data.get("amount_due") or preview_data.get("total") or fallback))
    except Exception:
        try:
            preview = await _call(
                stripe.Invoice.upcoming,
                customer=agency.stripe_customer_id,
                subscription=agency.stripe_subscription_id,
            )
            preview_data = _as_dict(preview)
            upcoming = max(0, int(preview_data.get("amount_due") or preview_data.get("total") or fallback))
        except Exception:
            upcoming = fallback
    return {"amount_paid_cents": paid, "upcoming_invoice_cents": upcoming}


async def sync_scrape_overage(agency: Agency, *, used: int | None = None) -> bool:
    """Buy extra $5/100 scrape lots when usage exceeds included client-pack scrapes."""
    if not agency.stripe_subscription_id or not settings.stripe_secret_key:
        return False
    if is_payg(agency):
        return False
    used_units = used if used is not None else (agency.scrape_units_used or 0)
    included = included_scrape_allowance(agency)
    lots_needed = extra_scrape_lots_needed(used_units, included)
    current = max(0, agency.scrape_pack_count or 0)
    if lots_needed <= current:
        return False
    await update_scrape_pack_quantity(agency, lots_needed * settings.scrape_pack_units)
    return True


async def ensure_client_capacity(db: AsyncSession, agency: Agency) -> None:
    await recover_subscription_if_missing(agency)
    normalize_billing_model(agency)
    count = await _active_client_count(db, agency.id)
    if is_payg(agency):
        if count >= PAYG_CLIENT_CAP:
            raise ValueError(f"Client limit reached ({PAYG_CLIENT_CAP}). Archive a client to add another.")
        return
    max_clients = agency.included_clients + agency.client_pack_count
    if count >= max_clients:
        raise ValueError(
            f"Client limit reached ({max_clients}). Purchase a per-client add-on pack to continue."
        )


def discounted_cents(amount: int, percent: int | None) -> int:
    """Apply a BYOK percent-off to a cent total (capped at 35%)."""
    amount = max(0, int(amount or 0))
    pct = max(0, min(35, int(percent or 0)))
    if pct <= 0:
        return amount
    return max(0, int(round(amount * (100 - pct) / 100.0)))


def apply_byok_discount(agency: Agency, providers: list[str]) -> None:
    mapping = {"groq": 10, "apify": 8, "serpapi": 5, "firecrawl": 5}
    agency.byok_discount_percent = min(35, sum(mapping.get(p, 0) for p in providers))


def _stripe():
    if not settings.stripe_secret_key:
        raise ValueError("Stripe is not configured. Set STRIPE_SECRET_KEY in the backend.")
    import stripe

    stripe.api_key = settings.stripe_secret_key
    return stripe


async def _call(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def sync_byok_stripe_coupon(agency: Agency) -> None:
    """Attach a forever percent-off coupon so the next Stripe invoice matches BYOK."""
    if is_payg(agency) or not agency.stripe_subscription_id or not settings.stripe_secret_key:
        return
    percent = max(0, min(35, int(agency.byok_discount_percent or 0)))
    stripe = _stripe()
    if percent <= 0:
        try:
            await _call(stripe.Subscription.delete_discount, agency.stripe_subscription_id)
        except Exception:
            try:
                await _call(stripe.Subscription.modify, agency.stripe_subscription_id, discounts=[])
            except Exception:
                pass
        return
    coupon_id = f"marketbiqs_byok_{percent}"
    try:
        await _call(stripe.Coupon.retrieve, coupon_id)
    except Exception:
        await _call(
            stripe.Coupon.create,
            id=coupon_id,
            percent_off=percent,
            duration="forever",
            name=f"BYOK {percent}%",
        )
    try:
        await _call(
            stripe.Subscription.modify,
            agency.stripe_subscription_id,
            discounts=[{"coupon": coupon_id}],
        )
    except Exception:
        await _call(
            stripe.Subscription.modify,
            agency.stripe_subscription_id,
            coupon=coupon_id,
        )


async def ensure_stripe_customer(agency: Agency) -> str:
    if agency.stripe_customer_id:
        return agency.stripe_customer_id
    stripe = _stripe()
    customer = await _call(
        stripe.Customer.create,
        name=agency.name,
        metadata={"agency_id": agency.id},
        idempotency_key=f"marketbiqs-customer-{agency.id}",
    )
    agency.stripe_customer_id = customer["id"]
    return agency.stripe_customer_id


async def recover_subscription_if_missing(agency: Agency) -> bool:
    """Re-attach an active Stripe subscription that a failed webhook never saved."""
    if agency.stripe_subscription_id or not agency.stripe_customer_id or not settings.stripe_secret_key:
        return False
    try:
        stripe = _stripe()
        result = await _call(
            stripe.Subscription.list,
            customer=agency.stripe_customer_id,
            status="all",
            limit=10,
        )
    except Exception:
        return False
    items = result.data if getattr(result, "data", None) is not None else (_as_dict(result).get("data") or [])
    chosen_id = None
    for sub in items:
        row = _as_dict(sub)
        if row.get("status") in {"active", "trialing", "past_due"}:
            chosen_id = row.get("id")
            break
    if not chosen_id:
        return False
    try:
        await retrieve_and_apply_subscription(agency, str(chosen_id))
    except Exception:
        return False
    return True


async def create_checkout_session(
    agency: Agency,
    add_client_packs: int,
    success_url: str,
    cancel_url: str,
    request_id: str,
    add_scrape_units: int = 0,
    billing_model: str = "plan",
    existing_clients: int = 0,
) -> dict[str, str]:
    if agency.stripe_subscription_id:
        raise ValueError("A subscription already exists. Change packs or open the billing portal.")
    if await recover_subscription_if_missing(agency):
        raise ValueError("A subscription already exists. Change packs or open the billing portal.")
    payg = str(billing_model or "plan").strip().lower() == "payg"
    if payg and not is_agency_workspace(agency):
        raise ValueError("Pay-as-you-go is only available for Agency workspaces.")
    stripe = _stripe()
    values = plan_values(_plan_id(agency))
    packs = max(0, min(50, int(add_client_packs or 0)))
    if payg:
        if not settings.stripe_payg_price_id:
            raise ValueError("STRIPE_PAYG_PRICE_ID is not configured.")
        line_items = [{"price": settings.stripe_payg_price_id, "quantity": 1}]
        metadata_plan = "payg"
    else:
        price_id = values["stripe_price_id"]
        if not price_id:
            raise ValueError(f"Stripe price ID for the {values['name']} plan is not configured.")
        line_items = [{"price": price_id, "quantity": 1}]
        if packs:
            if not settings.stripe_client_pack_price_id:
                raise ValueError("STRIPE_CLIENT_PACK_PRICE_ID is not configured.")
            line_items.append({"price": settings.stripe_client_pack_price_id, "quantity": packs})
        metadata_plan = values["id"]
    scrape_lots = 0 if payg else _scrape_lots(add_scrape_units)
    if scrape_lots:
        if not settings.stripe_scrape_pack_price_id:
            raise ValueError("STRIPE_SCRAPE_PACK_PRICE_ID is not configured.")
        line_items.append({"price": settings.stripe_scrape_pack_price_id, "quantity": scrape_lots})
    customer_id = await ensure_stripe_customer(agency)
    metadata = {"agency_id": agency.id, "plan": metadata_plan, "billing_model": "payg" if payg else "plan"}
    session = await _call(
        stripe.checkout.Session.create,
        mode="subscription",
        customer=customer_id,
        line_items=line_items,
        success_url=success_url,
        cancel_url=cancel_url,
        client_reference_id=agency.id,
        metadata=metadata,
        subscription_data={"metadata": metadata},
        allow_promotion_codes=True,
        payment_method_collection="always" if payg else "if_required",
        idempotency_key=f"marketbiqs-checkout-{agency.id}-{request_id}",
    )
    return {"mode": "stripe", "url": session.url, "session_id": session.id}


def _timestamp(value: Any) -> datetime | None:
    try:
        return (
            datetime.fromtimestamp(int(value), timezone.utc).replace(tzinfo=None)
            if value
            else None
        )
    except (TypeError, ValueError, OSError):
        return None


def _price_id(item: Any) -> str:
    data = _as_dict(item)
    price = _as_dict(data.get("price"))
    return str(price.get("id") or "")


def apply_subscription(agency: Agency, subscription: Any) -> None:
    """Synchronize local state from Stripe's subscription object."""
    subscription = _as_dict(subscription)
    agency.stripe_subscription_id = subscription.get("id") or agency.stripe_subscription_id
    agency.billing_status = str(subscription.get("status") or agency.billing_status)
    agency.cancel_at_period_end = bool(subscription.get("cancel_at_period_end"))
    items = ((_as_dict(subscription.get("items"))).get("data") or [])
    first_item = _as_dict(items[0] if items else {})
    new_period_start = _timestamp(
        subscription.get("current_period_start") or first_item.get("current_period_start")
    )
    if (
        new_period_start
        and agency.billing_period_start
        and new_period_start > agency.billing_period_start
    ):
        agency.reports_used = 0
        agency.scrape_units_used = 0
    agency.billing_period_start = new_period_start
    agency.billing_period_end = _timestamp(
        subscription.get("current_period_end") or first_item.get("current_period_end")
    )
    packs = 0
    scrape_lots = 0
    scrape_item_id = None
    saw_base = False
    saw_payg = False
    metadata = _as_dict(subscription.get("metadata"))
    for item in items:
        row = _as_dict(item)
        price_id = _price_id(row)
        if price_id == settings.stripe_payg_price_id:
            saw_payg = True
            agency.stripe_base_item_id = row.get("id")
        elif price_id in {settings.stripe_agency_price_id, settings.stripe_individual_price_id}:
            saw_base = True
            agency.stripe_base_item_id = row.get("id")
            if price_id == settings.stripe_individual_price_id:
                agency.plan = PlanType.creator
                agency.billing_model = "plan"
            elif price_id == settings.stripe_agency_price_id:
                agency.plan = PlanType.agency
                agency.billing_model = "plan"
        elif price_id == settings.stripe_client_pack_price_id:
            agency.stripe_pack_item_id = row.get("id")
            packs = max(0, int(row.get("quantity") or 0))
        elif price_id == settings.stripe_scrape_pack_price_id:
            scrape_item_id = row.get("id")
            scrape_lots = max(0, int(row.get("quantity") or 0))
    agency.client_pack_count = packs
    agency.stripe_scrape_item_id = scrape_item_id
    agency.scrape_pack_count = scrape_lots
    if saw_payg or metadata.get("billing_model") == "payg" or (not saw_base and packs):
        if is_agency_workspace(agency):
            agency.billing_model = "payg"
            agency.plan = PlanType.agency
    sync_entitlements(agency)


async def retrieve_and_apply_subscription(agency: Agency, subscription_id: str) -> Any:
    stripe = _stripe()
    subscription = await _call(
        stripe.Subscription.retrieve,
        subscription_id,
        expand=["items.data.price"],
    )
    apply_subscription(agency, subscription)
    return subscription


async def update_pack_quantity(agency: Agency, quantity: int) -> Any:
    if is_payg(agency):
        raise ValueError("PAYG bills usage directly — client add-on packs are only for the Agency plan.")
    if not agency.stripe_subscription_id:
        raise ValueError("Subscribe to a plan before adding PAYG packs.")
    if not settings.stripe_client_pack_price_id:
        raise ValueError("STRIPE_CLIENT_PACK_PRICE_ID is not configured.")
    stripe = _stripe()
    quantity = max(0, min(50, int(quantity)))
    subscription = await _call(
        stripe.Subscription.retrieve,
        agency.stripe_subscription_id,
        expand=["items.data.price"],
    )
    apply_subscription(agency, subscription)
    previous_quantity = max(0, agency.client_pack_count or 0)
    operation_key = f"marketbiqs-pack-{agency.id}-{previous_quantity}-to-{quantity}"
    if agency.stripe_pack_item_id:
        if quantity:
            await _call(
                stripe.SubscriptionItem.modify,
                agency.stripe_pack_item_id,
                quantity=quantity,
                proration_behavior="create_prorations",
                idempotency_key=operation_key,
            )
        else:
            await _call(
                stripe.SubscriptionItem.delete,
                agency.stripe_pack_item_id,
                proration_behavior="create_prorations",
                idempotency_key=operation_key,
            )
    elif quantity:
        await _call(
            stripe.SubscriptionItem.create,
            subscription=agency.stripe_subscription_id,
            price=settings.stripe_client_pack_price_id,
            quantity=quantity,
            proration_behavior="create_prorations",
            idempotency_key=operation_key,
        )
    return await retrieve_and_apply_subscription(agency, agency.stripe_subscription_id)


async def _set_subscription_addon(
    agency: Agency,
    *,
    item_id: str | None,
    price_id: str,
    quantity: int,
    operation_key: str,
) -> Any:
    stripe = _stripe()
    quantity = max(0, int(quantity))
    if item_id:
        if quantity:
            await _call(
                stripe.SubscriptionItem.modify,
                item_id,
                quantity=quantity,
                proration_behavior="create_prorations",
                idempotency_key=operation_key,
            )
        else:
            await _call(
                stripe.SubscriptionItem.delete,
                item_id,
                proration_behavior="create_prorations",
                idempotency_key=operation_key,
            )
    elif quantity:
        await _call(
            stripe.SubscriptionItem.create,
            subscription=agency.stripe_subscription_id,
            price=price_id,
            quantity=quantity,
            proration_behavior="create_prorations",
            idempotency_key=operation_key,
        )
    return await retrieve_and_apply_subscription(agency, agency.stripe_subscription_id)


async def update_scrape_pack_quantity(agency: Agency, scrape_units: int) -> Any:
    if is_payg(agency):
        raise ValueError("PAYG bills scrape units from usage — extra scrape packs are only for the Agency plan.")
    if not agency.stripe_subscription_id:
        raise ValueError("Subscribe to a plan before adding scrape units.")
    if not settings.stripe_scrape_pack_price_id:
        raise ValueError("STRIPE_SCRAPE_PACK_PRICE_ID is not configured.")
    lots = _scrape_lots(scrape_units)
    stripe = _stripe()
    subscription = await _call(
        stripe.Subscription.retrieve,
        agency.stripe_subscription_id,
        expand=["items.data.price"],
    )
    apply_subscription(agency, subscription)
    previous = max(0, agency.scrape_pack_count or 0)
    return await _set_subscription_addon(
        agency,
        item_id=agency.stripe_scrape_item_id,
        price_id=settings.stripe_scrape_pack_price_id,
        quantity=lots,
        operation_key=f"marketbiqs-scrapes-{agency.id}-{previous}-to-{lots}",
    )


async def create_portal_session(agency: Agency, return_url: str) -> dict[str, str]:
    if not agency.stripe_customer_id:
        raise ValueError("No Stripe customer exists for this workspace.")
    stripe = _stripe()
    session = await _call(
        stripe.billing_portal.Session.create,
        customer=agency.stripe_customer_id,
        return_url=return_url,
    )
    return {"url": session.url}


async def verify_checkout_session(agency: Agency, session_id: str) -> dict[str, Any]:
    stripe = _stripe()
    session = await _call(
        stripe.checkout.Session.retrieve,
        session_id,
        expand=["subscription.items.data.price"],
    )
    session = _as_dict(session)
    metadata = _as_dict(session.get("metadata"))
    if metadata.get("agency_id") != agency.id:
        raise ValueError("Checkout session does not belong to this workspace.")
    subscription = session.get("subscription")
    if subscription and not isinstance(subscription, str):
        apply_subscription(agency, subscription)
    elif subscription:
        await retrieve_and_apply_subscription(agency, subscription)
    return {
        "session_id": session.get("id"),
        "status": session.get("status"),
        "payment_status": session.get("payment_status"),
        "billing_status": agency.billing_status,
        "ready": agency.billing_status in {"active", "trialing"},
    }
