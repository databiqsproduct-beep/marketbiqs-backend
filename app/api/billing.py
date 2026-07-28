from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.deps import AuthContext, get_auth_context
from app.models import ApiKeyVault, ClientBrand, UsageEvent
from app.schemas import BudgetOut, ByokOut, ByokUpsert, CheckoutRequest
from app.security import encrypt_secret, mask_key, decrypt_secret
from app.services.billing import apply_byok_discount, apply_paid_packs, compute_budget, create_checkout_session

router = APIRouter(prefix="/billing", tags=["billing"])
settings = get_settings()


@router.get("/budget", response_model=BudgetOut)
async def budget(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    active = (
        await db.execute(
            select(func.count())
            .select_from(ClientBrand)
            .where(ClientBrand.agency_id == ctx.agency.id, ClientBrand.is_active.is_(True))
        )
    ).scalar_one()
    return compute_budget(ctx.agency, active)


@router.post("/checkout")
async def checkout(
    payload: CheckoutRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    success = payload.success_url or f"{settings.frontend_url}/billing?success=1"
    cancel = payload.cancel_url or f"{settings.frontend_url}/billing?canceled=1"
    try:
        result = await create_checkout_session(db, ctx.agency, payload.add_client_packs, success, cancel)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.flush()
    return result


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    if not settings.stripe_webhook_secret or not settings.stripe_secret_key:
        return {"ok": True, "mode": "disabled"}
    import stripe
    from app.models import Agency

    stripe.api_key = settings.stripe_secret_key
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, settings.stripe_webhook_secret)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        agency_id = (session.get("metadata") or {}).get("agency_id")
        packs = int((session.get("metadata") or {}).get("add_client_packs") or 0)
        if agency_id:
            agency = await db.get(Agency, agency_id)
            if agency:
                agency.stripe_subscription_id = session.get("subscription")
                apply_paid_packs(agency, packs)
                db.add(
                    UsageEvent(
                        agency_id=agency.id,
                        event_type="pack_purchase_stripe",
                        units=packs,
                        meta={"session_id": session.get("id"), "reports_quota": agency.reports_quota},
                    )
                )
                await db.flush()
    return {"ok": True}


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
    return {"ok": True}
