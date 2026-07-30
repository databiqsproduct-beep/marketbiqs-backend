from contextlib import asynccontextmanager
from datetime import datetime
import logging
import time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from croniter import croniter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.api import agency, auth, billing, chat, clients, competitive, delivery, intelligence, integrations, reports, supabase_api, whitelabel
from app.config import get_settings
from app.database import DATABASE_BACKEND, AsyncSessionLocal, init_db, ping_db
from app.models import Agency, ClientBrand, Report
from app.services.actions import action_deliver, action_run_intel
from app.services.supabase_client import ensure_reports_bucket, ping_supabase, supabase_configured

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("marketbiqs")

settings = get_settings()
scheduler = AsyncIOScheduler()


async def scheduled_ai_pipeline() -> None:
    async with AsyncSessionLocal() as db:
        clients = (await db.execute(select(ClientBrand).where(ClientBrand.is_active.is_(True)))).scalars().all()
        for client in clients:
            agency = await db.get(Agency, client.agency_id)
            if not agency:
                continue
            try:
                await action_run_intel(db, agency, client, push_jira=True, generate_report=True)
                await db.commit()
            except Exception:
                logger.exception("Scheduled intel failed for client %s", client.id)
                await db.rollback()


async def scheduled_delivery_pipeline() -> None:
    """Send due client deliveries based on delivery_schedule_cron."""
    now = datetime.utcnow()
    async with AsyncSessionLocal() as db:
        clients = (
            await db.execute(select(ClientBrand).where(ClientBrand.is_active.is_(True)))
        ).scalars().all()
        for client in clients:
            cron_expr = (client.delivery_schedule_cron or "").strip()
            if not cron_expr:
                continue
            try:
                base = now.replace(second=0, microsecond=0)
                itr = croniter(cron_expr, base)
                prev = itr.get_prev(datetime)
                if prev != base:
                    continue
            except Exception:
                continue
            agency = await db.get(Agency, client.agency_id)
            if not agency:
                continue
            report = (
                await db.execute(
                    select(Report)
                    .where(Report.client_id == client.id)
                    .order_by(Report.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            try:
                await action_deliver(db, agency, client, report=report, message=None)
                await db.commit()
            except Exception:
                logger.exception("Scheduled delivery failed for client %s", client.id)
                await db.rollback()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.app_env == "production" and settings.secret_key.startswith("biqs-dev"):
        logger.warning("Insecure SECRET_KEY detected in production — rotate before go-live")
    if settings.app_env == "production" and DATABASE_BACKEND == "sqlite":
        logger.error(
            "PRODUCTION IS USING SQLITE — set DATABASE_URL or SUPABASE_DB_PASSWORD on Railway. "
            "Ephemeral disk wipes users/memberships on every deploy (causes auth 403s)."
        )
    await init_db()
    if supabase_configured():
        try:
            status = await ping_supabase()
            logger.info("Supabase ping: %s", status)
            await ensure_reports_bucket()
        except Exception:
            logger.exception("Supabase startup check failed")
    else:
        logger.warning("Supabase API keys not configured — set SUPABASE_URL + SUPABASE_SECRET_KEY")
    scheduler.add_job(
        scheduled_ai_pipeline,
        "interval",
        hours=settings.scrape_interval_hours,
        id="agency_ai_pipeline",
        replace_existing=True,
    )
    scheduler.add_job(
        scheduled_delivery_pipeline,
        "interval",
        minutes=1,
        id="agency_delivery_pipeline",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("MarketBiqs API started (db=%s, scrape_every=%sh)", DATABASE_BACKEND, settings.scrape_interval_hours)
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)[:200]})
    response.headers["X-Process-Time-Ms"] = str(int((time.perf_counter() - started) * 1000))
    return response


app.include_router(auth.router, prefix="/api")
app.include_router(agency.router, prefix="/api")
app.include_router(clients.router, prefix="/api")
app.include_router(competitive.router, prefix="/api")
app.include_router(intelligence.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(billing.router, prefix="/api")
app.include_router(integrations.router, prefix="/api")
app.include_router(delivery.router, prefix="/api")
app.include_router(whitelabel.router, prefix="/api")
app.include_router(supabase_api.router, prefix="/api")


@app.get("/health")
async def health():
    db_ok = await ping_db()
    supabase = await ping_supabase()
    healthy = db_ok and (supabase.get("ok") if supabase.get("configured") else True)
    return {
        "status": "ok" if healthy else "degraded",
        "service": settings.app_name,
        "database": DATABASE_BACKEND,
        "database_reachable": db_ok,
        "supabase": supabase,
        "env": settings.app_env,
        "schedulers": ["agency_ai_pipeline", "agency_delivery_pipeline"],
    }
