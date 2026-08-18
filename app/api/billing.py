from urllib.parse import urlparse
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.deps import AuthContext, get_auth_context
from app.models import Agency, ApiKeyVault, ClientBrand, StripeEvent
from app.schemas import (
    BillingCatalogOut,
    BudgetOut,
    ByokOut,
    ByokUpsert,
    CheckoutRequest,
    PackUpdateRequest,
    PortalRequest,
    ScrapePackUpdateRequest,
)
from app.security import encrypt_secret, mask_key, decrypt_secret
from app.services.billing import (
    apply_byok_discount,
    apply_subscription,
    billing_catalog,
    compute_budget,
    create_checkout_session,
    create_portal_session,
    load_stripe_spend,
    recover_subscription_if_missing,
    retrieve_and_apply_subscription,
    sync_payg_usage_charges,
    migrate_payg_off_fixed_packs,
    count_intel_runs,
    is_payg,
    sync_byok_stripe_coupon,
    update_pack_quantity,
    update_scrape_pack_quantity,
    verify_checkout_session,
    sync_entitlements,
    _as_dict,
)

router = APIRouter(prefix="/billing", tags=["billing"])
settings = get_settings()
logger = logging.getLogger("marketbiqs.billing")


def _require_billing_admin(ctx: AuthContext) -> None:
    role = ctx.membership.role.value if hasattr(ctx.membership.role, "value") else str(ctx.membership.role)
    if role not in {"owner", "admin"}:
        raise HTTPException(status_code=403, detail="Only owners/admins can manage billing")


def _safe_frontend_url(candidate: str | None, fallback_path: str) -> str:
    frontend = settings.frontend_url.rstrip("/")
    fallback = f"{frontend}{fallback_path}"
    if not candidate:
        return fallback
    parsed = urlparse(candidate)
    allowed = urlparse(frontend)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != allowed.netloc:
        return fallback
    return candidate


@router.get("/catalog", response_model=BillingCatalogOut)
async def catalog(_: AuthContext = Depends(get_auth_context)):
    return billing_catalog()


@router.get("/budget", response_model=BudgetOut)
async def budget(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    await recover_subscription_if_missing(ctx.agency)
    if is_payg(ctx.agency):
        try:
            await migrate_payg_off_fixed_packs(ctx.agency)
        except Exception:
            logger.warning("PAYG pack migration failed for agency=%s", ctx.agency.id, exc_info=True)
    else:
        try:
            from app.services.billing import sync_scrape_overage

            await sync_scrape_overage(ctx.agency)
        except Exception:
            logger.warning("Scrape overage sync failed for agency=%s", ctx.agency.id, exc_info=True)
    active = (
        await db.execute(
            select(func.count())
            .select_from(ClientBrand)
            .where(ClientBrand.agency_id == ctx.agency.id, ClientBrand.is_active.is_(True))
        )
    ).scalar_one()
    intel_runs = await count_intel_runs(db, ctx.agency)
    if is_payg(ctx.agency):
        try:
            await sync_payg_usage_charges(
                ctx.agency,
                clients=int(active or 0),
                intel_runs=intel_runs,
                reports=ctx.agency.reports_used or 0,
                scrapes=ctx.agency.scrape_units_used or 0,
            )
        except Exception:
            logger.warning("PAYG usage charge sync failed for agency=%s", ctx.agency.id, exc_info=True)
    elif ctx.agency.byok_discount_percent and ctx.agency.stripe_subscription_id:
        try:
            await sync_byok_stripe_coupon(ctx.agency)
        except Exception:
            logger.warning("BYOK Stripe coupon sync failed for agency=%s", ctx.agency.id, exc_info=True)
    snapshot = compute_budget(ctx.agency, active, intel_runs_used=intel_runs)
    spend = await load_stripe_spend(ctx.agency, estimated=snapshot.estimated_monthly_cents)
    upcoming = int(spend.get("upcoming_invoice_cents") or 0)
    if snapshot.byok_discount_percent and upcoming > snapshot.estimated_monthly_cents:
        spend["upcoming_invoice_cents"] = snapshot.estimated_monthly_cents
    return snapshot.model_copy(update=spend)


@router.post("/checkout")
async def checkout(
    payload: CheckoutRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    _require_billing_admin(ctx)
    success = _safe_frontend_url(
        payload.success_url,
        "/billing?checkout=success&session_id={CHECKOUT_SESSION_ID}",
    )
    if "{CHECKOUT_SESSION_ID}" not in success:
        separator = "&" if "?" in success else "?"
        success = f"{success}{separator}session_id={{CHECKOUT_SESSION_ID}}"
    cancel = _safe_frontend_url(payload.cancel_url, "/billing?checkout=canceled")
    active_clients = (
        await db.execute(
            select(func.count())
            .select_from(ClientBrand)
            .where(ClientBrand.agency_id == ctx.agency.id, ClientBrand.is_active.is_(True))
        )
    ).scalar_one()
    try:
        result = await create_checkout_session(
            ctx.agency,
            payload.add_client_packs,
            success,
            cancel,
            payload.request_id,
            add_scrape_units=payload.add_scrape_units,
            billing_model=payload.billing_model,
            existing_clients=active_clients,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.flush()
    return result


@router.post("/packs")
async def update_packs(
    payload: PackUpdateRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    _require_billing_admin(ctx)
    if is_payg(ctx.agency):
        raise HTTPException(
            status_code=400,
            detail="PAYG bills clients and scrapes from usage. Add-on packs are only for the Agency plan.",
        )
    try:
        await update_pack_quantity(ctx.agency, payload.client_pack_count)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.flush()
    return {"ok": True, "client_pack_count": ctx.agency.client_pack_count}


@router.post("/scrape-units")
async def update_scrape_units(
    payload: ScrapePackUpdateRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    _require_billing_admin(ctx)
    if is_payg(ctx.agency):
        raise HTTPException(
            status_code=400,
            detail="PAYG bills scrape units from usage. Extra scrape packs are only for the Agency plan.",
        )
    try:
        await update_scrape_pack_quantity(ctx.agency, payload.scrape_units)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.flush()
    return {
        "ok": True,
        "scrape_pack_count": ctx.agency.scrape_pack_count,
        "scrape_quota": ctx.agency.scrape_quota,
    }


@router.post("/portal")
async def portal(
    payload: PortalRequest,
    ctx: AuthContext = Depends(get_auth_context),
):
    _require_billing_admin(ctx)
    return_url = _safe_frontend_url(payload.return_url, "/billing")
    try:
        return await create_portal_session(ctx.agency, return_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/checkout-session/{session_id}")
async def checkout_status(
    session_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    _require_billing_admin(ctx)
    try:
        result = await verify_checkout_session(ctx.agency, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.flush()
    return result


async def _agency_for_stripe_object(db: AsyncSession, obj) -> Agency | None:
    obj = _as_dict(obj)
    metadata = _as_dict(obj.get("metadata"))
    agency_id = metadata.get("agency_id") or obj.get("client_reference_id")
    if agency_id:
        agency = await db.get(Agency, agency_id)
        if agency:
            return agency
    customer_id = obj.get("customer")
    if customer_id:
        return (
            await db.execute(select(Agency).where(Agency.stripe_customer_id == str(customer_id)))
        ).scalar_one_or_none()
    return None


def _stripe_object_id(value) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        return value
    data = _as_dict(value)
    return data.get("id")


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    if not settings.stripe_webhook_secret or not settings.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Stripe webhook is not configured")
    import stripe

    stripe.api_key = settings.stripe_secret_key
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, settings.stripe_webhook_secret)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    event_id = str(event["id"])
    if (
        await db.execute(select(StripeEvent.id).where(StripeEvent.event_id == event_id))
    ).scalar_one_or_none():
        return {"ok": True, "duplicate": True}

    event_type = str(event["type"])
    obj = _as_dict(event["data"]["object"])
    agency = await _agency_for_stripe_object(db, obj)

    if event_type in {
        "checkout.session.completed",
        "checkout.session.async_payment_succeeded",
    }:
        subscription_id = _stripe_object_id(obj.get("subscription"))
        if agency and subscription_id:
            agency.stripe_customer_id = obj.get("customer") or agency.stripe_customer_id
            await retrieve_and_apply_subscription(agency, subscription_id)
    elif event_type.startswith("customer.subscription."):
        if agency:
            if event_type == "customer.subscription.deleted":
                agency.billing_status = "canceled"
                agency.cancel_at_period_end = False
                agency.stripe_subscription_id = None
                agency.client_pack_count = 0
                agency.scrape_pack_count = 0
                agency.stripe_base_item_id = None
                agency.stripe_pack_item_id = None
                agency.stripe_scrape_item_id = None
                agency.billing_model = "plan"
                sync_entitlements(agency)
            else:
                apply_subscription(agency, obj)
    elif event_type == "invoice.paid":
        subscription_id = _stripe_object_id(obj.get("subscription"))
        if agency and subscription_id:
            await retrieve_and_apply_subscription(agency, subscription_id)
            agency.billing_status = "active"
    elif event_type in {
        "invoice.payment_failed",
        "checkout.session.async_payment_failed",
    }:
        if agency:
            agency.billing_status = "past_due"

    try:
        payload_dict = json.loads(payload)
        if not isinstance(payload_dict, dict):
            payload_dict = {"id": event_id, "type": event_type}
    except Exception:
        payload_dict = {"id": event_id, "type": event_type}
    db.add(
        StripeEvent(
            event_id=event_id,
            event_type=event_type,
            agency_id=agency.id if agency else None,
            payload=payload_dict,
        )
    )
    await db.flush()
    return {"ok": True, "handled": event_type}


@router.get("/byok", response_model=list[ByokOut])
async def list_byok(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ApiKeyVault).where(ApiKeyVault.agency_id == ctx.agency.id))
    items = []
    for row in result.scalars().all():
        try:
            hint = mask_key(decrypt_secret(row.encrypted_key))
        except Exception:
            hint = "****"
        items.append(
            ByokOut(
                id=row.id,
                provider=row.provider,
                label=row.label,
                is_active=row.is_active,
                key_hint=hint,
                created_at=row.created_at,
            )
        )
    return items


@router.put("/byok", response_model=ByokOut)
async def upsert_byok(
    payload: ByokUpsert,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    provider = payload.provider.lower().strip()
    if provider not in {"groq", "apify", "serpapi", "firecrawl"}:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    result = await db.execute(
        select(ApiKeyVault).where(ApiKeyVault.agency_id == ctx.agency.id, ApiKeyVault.provider == provider)
    )
    row = result.scalar_one_or_none()
    if not row:
        row = ApiKeyVault(
            agency_id=ctx.agency.id,
            provider=provider,
            encrypted_key=encrypt_secret(payload.api_key),
            label=payload.label,
            is_active=True,
        )
        db.add(row)
    else:
        row.encrypted_key = encrypt_secret(payload.api_key)
        row.label = payload.label
        row.is_active = True
    await db.flush()
    providers = (
        await db.execute(
            select(ApiKeyVault.provider).where(ApiKeyVault.agency_id == ctx.agency.id, ApiKeyVault.is_active.is_(True))
        )
    ).scalars().all()
    apply_byok_discount(ctx.agency, list(providers))
    await db.flush()
    try:
        await sync_byok_stripe_coupon(ctx.agency)
    except Exception:
        logger.warning("BYOK Stripe coupon sync failed for agency=%s", ctx.agency.id, exc_info=True)
    return ByokOut(
        id=row.id,
        provider=row.provider,
        label=row.label,
        is_active=row.is_active,
        key_hint=mask_key(payload.api_key),
        created_at=row.created_at,
    )


@router.delete("/byok/{provider}")
async def delete_byok(
    provider: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ApiKeyVault).where(ApiKeyVault.agency_id == ctx.agency.id, ApiKeyVault.provider == provider.lower())
    )
    row = result.scalar_one_or_none()
    if row:
        await db.delete(row)
        await db.flush()
    providers = (
        await db.execute(
            select(ApiKeyVault.provider).where(ApiKeyVault.agency_id == ctx.agency.id, ApiKeyVault.is_active.is_(True))
        )
    ).scalars().all()
    apply_byok_discount(ctx.agency, list(providers))
    await db.flush()
    try:
        await sync_byok_stripe_coupon(ctx.agency)
    except Exception:
        logger.warning("BYOK Stripe coupon sync failed for agency=%s", ctx.agency.id, exc_info=True)
    return {"ok": True}
