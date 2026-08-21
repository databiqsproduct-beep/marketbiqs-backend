from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import AuthContext, get_auth_context, get_white_label_agency
from app.models import Agency, ClientBrand, Competitor, ProductFeature, Report, TrendSignal, WhiteLabelApiKey
from app.schemas import TrendOut, WhiteLabelKeyCreate, WhiteLabelKeyOut
from app.security import generate_api_key
from app.services.intelligence import run_client_intelligence
from app.services.reports import generate_client_report

router = APIRouter(tags=["white-label"])


class WhiteLabelClientOut(BaseModel):
    id: str
    name: str
    industry: str | None = None
    niche: str | None = None
    website: str | None = None
    tagline: str | None = None


class WhiteLabelClientSnapshot(BaseModel):
    client: WhiteLabelClientOut
    competitors: list[dict]
    features: list[dict]
    trends: list[dict]
    reports: list[dict]


@router.post("/agency/white-label-keys", response_model=WhiteLabelKeyOut)
async def create_key(
    payload: WhiteLabelKeyCreate,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    raw, prefix, hashed = generate_api_key()
    row = WhiteLabelApiKey(
        agency_id=ctx.agency.id,
        name=payload.name,
        key_prefix=prefix,
        hashed_key=hashed,
        monthly_quota=payload.monthly_quota,
    )
    db.add(row)
    await db.flush()
    return WhiteLabelKeyOut(
        id=row.id,
        name=row.name,
        key_prefix=row.key_prefix,
        api_key=raw,
        is_active=row.is_active,
        requests_used=row.requests_used,
        monthly_quota=row.monthly_quota,
        created_at=row.created_at,
    )


@router.get("/agency/white-label-keys", response_model=list[WhiteLabelKeyOut])
async def list_keys(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(WhiteLabelApiKey).where(WhiteLabelApiKey.agency_id == ctx.agency.id))
    return [
        WhiteLabelKeyOut(
            id=r.id,
            name=r.name,
            key_prefix=r.key_prefix,
            api_key=None,
            is_active=r.is_active,
            requests_used=r.requests_used,
            monthly_quota=r.monthly_quota,
            created_at=r.created_at,
        )
        for r in result.scalars().all()
    ]


@router.get("/v1/intelligence/{client_id}/trends", response_model=list[TrendOut])
async def embed_trends(
    client_id: str,
    agency: Agency = Depends(get_white_label_agency),
    db: AsyncSession = Depends(get_db),
):
    client = await db.get(ClientBrand, client_id)
    if not client or client.agency_id != agency.id:
        raise HTTPException(status_code=404, detail="Client not found")
    result = await db.execute(
        select(TrendSignal)
        .where(TrendSignal.client_id == client.id, TrendSignal.agency_id == agency.id)
        .order_by(TrendSignal.detected_at.desc())
        .limit(25)
    )
    return list(result.scalars().all())


@router.get("/v1/clients", response_model=list[WhiteLabelClientOut])
async def embed_list_clients(
    agency: Agency = Depends(get_white_label_agency),
    db: AsyncSession = Depends(get_db),
):
    """List agency clients available to this white-label API key."""
    rows = (
        await db.execute(
            select(ClientBrand)
            .where(ClientBrand.agency_id == agency.id, ClientBrand.is_active.is_(True))
            .order_by(ClientBrand.name.asc())
        )
    ).scalars().all()
    return [
        WhiteLabelClientOut(
            id=c.id,
            name=c.name,
            industry=c.industry,
            niche=c.niche,
            website=c.website,
            tagline=c.tagline,
        )
        for c in rows
    ]


@router.get("/v1/intelligence/{client_id}/snapshot", response_model=WhiteLabelClientSnapshot)
async def embed_client_snapshot(
    client_id: str,
    agency: Agency = Depends(get_white_label_agency),
    db: AsyncSession = Depends(get_db),
):
    """Read-only pack: client + tracked rivals + features + recent trends/reports (no AI)."""
    client = await db.get(ClientBrand, client_id)
    if not client or client.agency_id != agency.id:
        raise HTTPException(status_code=404, detail="Client not found")

    competitors = (
        await db.execute(
            select(Competitor).where(
                Competitor.client_id == client.id,
                Competitor.agency_id == agency.id,
                Competitor.is_tracking.is_(True),
            )
        )
    ).scalars().all()
    features = (
        await db.execute(
            select(ProductFeature).where(
                ProductFeature.client_id == client.id,
                ProductFeature.agency_id == agency.id,
            )
        )
    ).scalars().all()
    trends = (
        await db.execute(
            select(TrendSignal)
            .where(TrendSignal.client_id == client.id, TrendSignal.agency_id == agency.id)
            .order_by(TrendSignal.detected_at.desc())
            .limit(15)
        )
    ).scalars().all()
    reports = (
        await db.execute(
            select(Report)
            .where(Report.client_id == client.id, Report.agency_id == agency.id)
            .order_by(Report.created_at.desc())
            .limit(5)
        )
    ).scalars().all()

    return WhiteLabelClientSnapshot(
        client=WhiteLabelClientOut(
            id=client.id,
            name=client.name,
            industry=client.industry,
            niche=client.niche,
            website=client.website,
            tagline=client.tagline,
        ),
        competitors=[
            {
                "id": c.id,
                "name": c.name,
                "website": c.website,
                "overlap_score": c.overlap_score,
                "threat_level": c.threat_level,
                "is_pinned": c.is_pinned,
            }
            for c in competitors
        ],
        features=[
            {
                "id": f.id,
                "name": f.name,
                "category": f.category,
                "description": f.description,
            }
            for f in features
        ],
        trends=[
            {
                "id": t.id,
                "topic": t.topic,
                "platform": t.platform,
                "summary": t.summary,
                "velocity_score": t.velocity_score,
                "detected_at": t.detected_at.isoformat() if t.detected_at else None,
            }
            for t in trends
        ],
        reports=[
            {
                "id": r.id,
                "title": r.title,
                "summary": r.summary,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in reports
        ],
    )


@router.post("/v1/intelligence/{client_id}/run")
async def embed_run(
    client_id: str,
    agency: Agency = Depends(get_white_label_agency),
    db: AsyncSession = Depends(get_db),
):
    client = await db.get(ClientBrand, client_id)
    if not client or client.agency_id != agency.id:
        raise HTTPException(status_code=404, detail="Client not found")
    job = await run_client_intelligence(db, agency, client)
    return {"status": job.status.value, "result": job.result_meta}


@router.post("/v1/intelligence/{client_id}/report")
async def embed_report(
    client_id: str,
    agency: Agency = Depends(get_white_label_agency),
    db: AsyncSession = Depends(get_db),
):
    client = await db.get(ClientBrand, client_id)
    if not client or client.agency_id != agency.id:
        raise HTTPException(status_code=404, detail="Client not found")
    report = await generate_client_report(db, agency, client)
    return {
        "id": report.id,
        "title": report.title,
        "summary": report.summary,
        "sections": report.sections,
        "pdf_path": report.pdf_path,
    }
