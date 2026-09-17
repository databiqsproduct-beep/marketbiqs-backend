from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import AuthContext, get_auth_context, get_tenant_client
from app.models import ClientBrand, Competitor, FeatureTicket, GoalAlert, ProductFeature, Report
from app.schemas import ClientCreate, ClientOut, ClientUpdate, CompetitorCreate, CompetitorOut
from app.services.billing import ensure_client_capacity, max_tracked_rivals
from app.services.competitive import collapse_duplicate_competitors, _find_matching_competitor

router = APIRouter(prefix="/clients", tags=["clients"])


async def _bulk_enrich_clients(db: AsyncSession, clients: list[ClientBrand]) -> list[ClientOut]:
    if not clients:
        return []

    client_ids = [c.id for c in clients]

    rivals_subq = (
        select(func.count())
        .where(Competitor.client_id == ClientBrand.id, Competitor.is_tracking.is_(True))
        .scalar_subquery()
    )
    features_subq = (
        select(func.count())
        .where(ProductFeature.client_id == ClientBrand.id)
        .scalar_subquery()
    )
    reports_subq = (
        select(func.count())
        .where(Report.client_id == ClientBrand.id)
        .scalar_subquery()
    )
    tickets_subq = (
        select(func.count())
        .where(FeatureTicket.client_id == ClientBrand.id)
        .scalar_subquery()
    )
    alerts_subq = (
        select(func.count())
        .where(GoalAlert.client_id == ClientBrand.id, GoalAlert.acted_on.is_(False))
        .scalar_subquery()
    )

    query = select(
        ClientBrand.id,
        rivals_subq.label("rivals_count"),
        features_subq.label("features_count"),
        reports_subq.label("reports_count"),
        tickets_subq.label("tickets_count"),
        alerts_subq.label("alerts_open"),
    ).where(ClientBrand.id.in_(client_ids))

    result = await db.execute(query)
    counts_map = {row.id: row for row in result.all()}

    results = []
    for client in clients:
        counts = counts_map.get(client.id)
        data = ClientOut.model_validate(client)
        if counts:
            results.append(
                data.model_copy(
                    update={
                        "rivals_count": counts.rivals_count or 0,
                        "features_count": counts.features_count or 0,
                        "reports_count": counts.reports_count or 0,
                        "tickets_count": counts.tickets_count or 0,
                        "alerts_open": counts.alerts_open or 0,
                    }
                )
            )
        else:
            results.append(data.model_copy())
    return results


@router.get("", response_model=list[ClientOut])
async def list_clients(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ClientBrand)
        .where(ClientBrand.agency_id == ctx.agency.id)
        .order_by(ClientBrand.created_at.desc())
    )
    clients = list(result.scalars().all())
    return await _bulk_enrich_clients(db, clients)


def _merge_notes_with_metadata(
    existing_notes: str | None,
    *,
    country: str | None = None,
    city: str | None = None,
    primary_offering: str | None = None,
    customer_type: str | None = None,
    niche: str | None = None,
) -> str | None:
    lines = [ln for ln in (existing_notes or "").splitlines() if ln.strip()]
    clean_lines = []
    for ln in lines:
        low = ln.lower().strip()
        if (
            (country is not None and (low.startswith("market:") or low.startswith("country:")))
            or (city is not None and low.startswith("city:"))
            or (primary_offering is not None and (low.startswith("primary offering:") or low.startswith("offering:")))
            or (customer_type is not None and low.startswith("customer type:"))
            or (niche is not None and low.startswith("niche:"))
        ):
            continue
        clean_lines.append(ln)

    meta_headers = []
    if country:
        meta_headers.append(f"Market: {country.strip()}")
    if city:
        meta_headers.append(f"City: {city.strip()}")
    if niche:
        meta_headers.append(f"Niche: {niche.strip()}")
    if customer_type:
        meta_headers.append(f"Customer type: {customer_type.strip()}")
    if primary_offering:
        meta_headers.append(f"Primary offering: {primary_offering.strip()}")

    all_lines = meta_headers + clean_lines
    return "\n".join(all_lines).strip() or None


@router.post("", response_model=ClientOut)
async def create_client(
    payload: ClientCreate,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    try:
        await ensure_client_capacity(db, ctx.agency)
    except ValueError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    data = payload.model_dump()
    country = data.pop("country", None)
    city = data.pop("city", None)
    primary_offering = data.pop("primary_offering", None)
    customer_type = data.pop("customer_type", None)
    niche_val = data.get("niche")
    if any([country, city, primary_offering, customer_type, niche_val]):
        data["notes"] = _merge_notes_with_metadata(
            data.get("notes"),
            country=country,
            city=city,
            primary_offering=primary_offering,
            customer_type=customer_type,
            niche=niche_val,
        )
    client = ClientBrand(agency_id=ctx.agency.id, **data)
    db.add(client)
    await db.flush()
    # Intel is started explicitly by the UI via POST /clients/{id}/auto-run
    # (background create-time runs raced the UI and often finished with no feedback).
    return (await _bulk_enrich_clients(db, [client]))[0]


@router.get("/{client_id}", response_model=ClientOut)
async def get_client(client: ClientBrand = Depends(get_tenant_client), db: AsyncSession = Depends(get_db)):
    return (await _bulk_enrich_clients(db, [client]))[0]


@router.patch("/{client_id}", response_model=ClientOut)
async def update_client(
    payload: ClientUpdate,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    activating = payload.is_active is True and not client.is_active
    if activating:
        try:
            await ensure_client_capacity(db, ctx.agency)
        except ValueError as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc
    data = payload.model_dump(exclude_unset=True)
    country = data.pop("country", None)
    city = data.pop("city", None)
    primary_offering = data.pop("primary_offering", None)
    customer_type = data.pop("customer_type", None)
    niche_val = data.get("niche")
    if any(x is not None for x in [country, city, primary_offering, customer_type, niche_val]):
        data["notes"] = _merge_notes_with_metadata(
            data.get("notes", client.notes),
            country=country,
            city=city,
            primary_offering=primary_offering,
            customer_type=customer_type,
            niche=niche_val,
        )
    for key, value in data.items():
        setattr(client, key, value)
    await db.flush()
    return (await _bulk_enrich_clients(db, [client]))[0]


@router.delete("/{client_id}")
async def delete_client(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    await db.delete(client)
    await db.flush()
    return {"ok": True, "deleted_id": client.id}


@router.get("/{client_id}/competitors", response_model=list[CompetitorOut])
async def list_competitors(
    client: ClientBrand = Depends(get_tenant_client),
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = Depends(get_auth_context),
    include_hidden: bool = False,
):
    stmt = select(Competitor).where(
        Competitor.client_id == client.id,
        Competitor.agency_id == ctx.agency.id,
    )
    if not include_hidden:
        stmt = stmt.where(Competitor.is_tracking.is_(True))
    result = await db.execute(
        stmt.order_by(Competitor.is_pinned.desc(), Competitor.is_tracking.desc(), Competitor.overlap_score.desc())
    )
    rows = list(result.scalars().all())
    if not include_hidden:
        collapse_duplicate_competitors(rows)
        await db.flush()
        rows = [row for row in rows if row.is_tracking]
    return rows


@router.post("/{client_id}/competitors", response_model=CompetitorOut)
async def add_competitor(
    payload: CompetitorCreate,
    client: ClientBrand = Depends(get_tenant_client),
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = Depends(get_auth_context),
):
    data = payload.model_dump()
    existing = (
        await db.execute(
            select(Competitor).where(
                Competitor.client_id == client.id,
                Competitor.agency_id == ctx.agency.id,
            )
        )
    ).scalars().all()
    competitor = _find_matching_competitor(list(existing), data.get("name") or "", data.get("website"))
    cap = max_tracked_rivals(ctx.agency)
    tracking = sum(1 for row in existing if row.is_tracking)
    enabling = not (competitor and competitor.is_tracking)
    if cap is not None and enabling and tracking >= cap:
        raise HTTPException(
            status_code=400,
            detail=f"Individual plans track up to {cap} competitors. Remove one before adding another.",
        )
    if competitor:
        competitor.website = data.get("website") or competitor.website
        competitor.is_tracking = True
        competitor.is_pinned = True
        if (competitor.overlap_score or 0) < 75:
            competitor.overlap_score = 75.0
        competitor.threat_level = competitor.threat_level or "high"
        await db.flush()
        await db.refresh(competitor)
        return competitor
    from app.services.competitive import _market_area_from_client
    client_mkt = _market_area_from_client(client) or "Pakistan"
    competitor = Competitor(
        agency_id=ctx.agency.id,
        client_id=client.id,
        **data,
        headquarters=data.get("headquarters") or client_mkt,
        # Keep manual rivals in intel runs (protected from AI prune + count slice)
        is_tracking=True,
        is_pinned=True,
        overlap_score=75.0,
        threat_level="high",
        why_dangerous=f"Manually added rival for {client.name}",
    )
    db.add(competitor)
    await db.flush()
    await db.refresh(competitor)
    return competitor


@router.delete("/{client_id}/competitors/{competitor_id}")
async def remove_competitor(
    competitor_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = Depends(get_auth_context),
):
    competitor = await db.get(Competitor, competitor_id)
    if not competitor or competitor.client_id != client.id or competitor.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Competitor not found")
    await db.delete(competitor)
    await db.flush()
    return {"ok": True}
