from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import AuthContext, get_auth_context, get_tenant_client
from app.models import ClientBrand, CompetitorSnapshot, Insight, SentimentRecord, TrackingJob, TrendSignal
from app.schemas import InsightOut, SentimentOut, SnapshotOut, TrendOut
from app.services.intelligence import run_client_intelligence

router = APIRouter(tags=["intelligence"])


@router.post("/clients/{client_id}/run")
async def run_intelligence(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    if ctx.agency.scrape_units_used >= ctx.agency.scrape_quota:
        raise HTTPException(status_code=402, detail="Scrape quota exceeded. Upgrade packs or enable BYOK.")
    job = await run_client_intelligence(db, ctx.agency, client)
    return {
        "job_id": job.id,
        "status": job.status.value,
        "detail": job.detail,
        "result": job.result_meta,
    }


@router.get("/clients/{client_id}/trends", response_model=list[TrendOut])
async def list_trends(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrendSignal)
        .where(TrendSignal.client_id == client.id, TrendSignal.agency_id == ctx.agency.id)
        .order_by(TrendSignal.detected_at.desc())
        .limit(50)
    )
    return list(result.scalars().all())


@router.get("/clients/{client_id}/sentiment", response_model=list[SentimentOut])
async def list_sentiment(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SentimentRecord)
        .where(SentimentRecord.client_id == client.id, SentimentRecord.agency_id == ctx.agency.id)
        .order_by(SentimentRecord.created_at.desc())
        .limit(50)
    )
    return list(result.scalars().all())


@router.get("/clients/{client_id}/insights", response_model=list[InsightOut])
async def list_insights(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Insight)
        .where(Insight.client_id == client.id, Insight.agency_id == ctx.agency.id)
        .order_by(Insight.created_at.desc())
        .limit(50)
    )
    return list(result.scalars().all())


@router.get("/clients/{client_id}/snapshots", response_model=list[SnapshotOut])
async def list_snapshots(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    from app.models import Competitor

    competitor_ids = (
        await db.execute(select(Competitor.id).where(Competitor.client_id == client.id, Competitor.agency_id == ctx.agency.id))
    ).scalars().all()
    if not competitor_ids:
        return []
    result = await db.execute(
        select(CompetitorSnapshot)
        .where(
            CompetitorSnapshot.agency_id == ctx.agency.id,
            CompetitorSnapshot.competitor_id.in_(list(competitor_ids)),
        )
        .order_by(CompetitorSnapshot.scraped_at.desc())
        .limit(100)
    )
    return list(result.scalars().all())


@router.get("/clients/{client_id}/jobs")
async def list_jobs(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrackingJob)
        .where(TrackingJob.client_id == client.id, TrackingJob.agency_id == ctx.agency.id)
        .order_by(TrackingJob.created_at.desc())
        .limit(30)
    )
    jobs = result.scalars().all()
    return [
        {
            "id": j.id,
            "job_type": j.job_type,
            "status": j.status.value,
            "detail": j.detail,
            "result_meta": j.result_meta,
            "created_at": j.created_at,
            "finished_at": j.finished_at,
        }
        for j in jobs
    ]
