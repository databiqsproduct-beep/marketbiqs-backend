from collections.abc import AsyncGenerator
import logging
import re
from urllib.parse import quote_plus, unquote, urlparse

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import get_settings

logger = logging.getLogger("marketbiqs.db")
settings = get_settings()


def _normalize_postgres_url(url: str) -> str:
    url = url.strip()
    if url.startswith("postgres+asyncpg://"):
        return url.replace("postgres+asyncpg://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1).replace(
            "postgres://", "postgresql+asyncpg://", 1
        )
    return url


def _pooler_host() -> str:
    explicit = (settings.supabase_pooler_host or "").strip()
    if explicit:
        return explicit.replace("https://", "").replace("http://", "").split(":")[0].strip()
    region = (settings.supabase_db_region or "ap-northeast-2").strip()
    # Prefer aws-1 for newer projects; override with SUPABASE_POOLER_HOST when unsure
    return f"aws-1-{region}.pooler.supabase.com"


def _pooler_url(ref: str, password: str, region: str | None = None) -> str:
    host = _pooler_host()
    return f"postgresql+asyncpg://postgres.{ref}:{quote_plus(password)}@{host}:6543/postgres"


def _build_supabase_pooler_url() -> str | None:
    """Transaction pooler URL — works on Railway (IPv4). Direct db.*.supabase.co:5432 often fails."""
    if not (settings.supabase_db_password and settings.supabase_project_ref):
        return None
    return _pooler_url(settings.supabase_project_ref.strip(), settings.supabase_db_password)


def _rewrite_direct_supabase_to_pooler(url: str) -> str | None:
    """
    Convert direct Supabase DB URL to transaction pooler.

    Direct host (db.<ref>.supabase.co:5432) is often IPv6-only → Railway gets
    OSError: [Errno 101] Network is unreachable.
    """
    normalized = _normalize_postgres_url(url)
    parsed = urlparse(normalized)
    host = parsed.hostname or ""
    match = re.match(r"^db\.([a-z0-9]+)\.supabase\.co$", host, re.I)
    if not match:
        return None
    ref = match.group(1)
    password = unquote(parsed.password or "") or (settings.supabase_db_password or "")
    if not password:
        logger.error(
            "DATABASE_URL points at direct Supabase host but password is missing — "
            "set SUPABASE_DB_PASSWORD or put the password in DATABASE_URL"
        )
        return None
    return _pooler_url(ref, password)


def _resolve_database_url() -> tuple[str, str]:
    """
    Production: Supabase Postgres when DATABASE_URL or SUPABASE_DB_PASSWORD is set.
    Local/dev: SQLite (./biqs.db) when Postgres is not configured.

    Prefer password+ref pooler (correct host) over a stale/wrong DATABASE_URL.
    """
    pooler = _build_supabase_pooler_url()
    url = (settings.database_url or "").strip()

    # Fix direct IPv6 host
    if url:
        rewritten = _rewrite_direct_supabase_to_pooler(url)
        if rewritten:
            logger.warning(
                "DATABASE_URL used direct Supabase host (IPv6) — rewritten to pooler %s",
                _pooler_host(),
            )
            return rewritten, "postgres"

    # Prefer known-good pooler from SUPABASE_DB_PASSWORD + host settings
    if pooler:
        if url and "pooler.supabase.com" in url:
            want = urlparse(pooler).hostname
            have = urlparse(_normalize_postgres_url(url)).hostname
            if have != want:
                logger.warning(
                    "DATABASE_URL pooler host %s is wrong for this project — using %s",
                    have,
                    want,
                )
                return pooler, "postgres"
            return _normalize_postgres_url(url), "postgres"
        if not url or "pooler.supabase.com" not in url:
            if url:
                logger.warning("Using SUPABASE_DB_PASSWORD pooler URL instead of non-pooler DATABASE_URL")
            return pooler, "postgres"

    if url:
        if "sqlite" in url:
            return url if "aiosqlite" in url else url.replace("sqlite://", "sqlite+aiosqlite://", 1), "sqlite"
        return _normalize_postgres_url(url), "postgres"

    if settings.supabase_db_url:
        rewritten = _rewrite_direct_supabase_to_pooler(settings.supabase_db_url)
        if rewritten:
            return rewritten, "postgres"
        return _normalize_postgres_url(settings.supabase_db_url), "postgres"

    logger.warning(
        "Supabase Postgres not configured (set DATABASE_URL or SUPABASE_DB_PASSWORD). "
        "Using local SQLite biqs.db for development."
    )
    return "sqlite+aiosqlite:///./biqs.db", "sqlite"


DATABASE_URL, DATABASE_BACKEND = _resolve_database_url()
try:
    _resolved_host = urlparse(DATABASE_URL).hostname
except Exception:
    _resolved_host = "?"
logger.info("Resolved database backend=%s host=%s", DATABASE_BACKEND, _resolved_host)

if settings.app_env == "production" and DATABASE_BACKEND == "sqlite":
    # Do not raise at import — that kills uvicorn before /health and Railway never deploys.
    logger.critical(
        "Production is resolving to SQLite. Set DATABASE_URL to the Supabase TRANSACTION pooler "
        "(aws-0-<region>.pooler.supabase.com:6543) or SUPABASE_DB_PASSWORD + SUPABASE_PROJECT_REF."
    )

_engine_kwargs: dict = {"echo": False, "future": True}
if DATABASE_BACKEND == "postgres":
    # statement_cache_size=0 is required for Supabase transaction pooler (port 6543)
    _engine_kwargs.update(
        pool_pre_ping=True,
        pool_size=8,
        max_overflow=12,
        pool_timeout=8,
        pool_recycle=180,
        connect_args={
            "timeout": 8,
            "command_timeout": 20,
            "statement_cache_size": 0,
            "ssl": "require",
        },
    )
else:
    # NullPool + WAL/busy_timeout avoids most "database is locked" under local concurrency
    _engine_kwargs["poolclass"] = NullPool
    _engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 60}

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


if DATABASE_BACKEND == "sqlite":

    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_on_connect(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def _ensure_pg_extensions() -> None:
    if DATABASE_BACKEND != "postgres":
        return
    try:
        async with engine.begin() as conn:
            await conn.execute(text('create extension if not exists "pgcrypto"'))
            try:
                await conn.execute(text('create extension if not exists "vector"'))
            except Exception:
                logger.warning("pgvector extension unavailable; text RAG still works without vectors")
    except Exception:
        # Transaction pooler / managed Supabase often blocks CREATE EXTENSION — safe to skip
        logger.warning("Could not ensure PG extensions (normal on Supabase pooler); continuing")


async def _apply_schema_patches() -> None:
    """Apply additive patches that create_all cannot add to existing databases."""
    common_patches = [
        "ALTER TABLE agencies ADD COLUMN stripe_base_item_id VARCHAR(120)",
        "ALTER TABLE agencies ADD COLUMN stripe_pack_item_id VARCHAR(120)",
        "ALTER TABLE agencies ADD COLUMN stripe_scrape_item_id VARCHAR(120)",
        "ALTER TABLE agencies ADD COLUMN scrape_pack_count INTEGER DEFAULT 0",
        "ALTER TABLE agencies ADD COLUMN billing_model VARCHAR(20) DEFAULT 'plan'",
        "ALTER TABLE agencies ADD COLUMN cancel_at_period_end BOOLEAN DEFAULT FALSE",
        "ALTER TABLE agencies ADD COLUMN billing_period_start TIMESTAMP",
        "ALTER TABLE agencies ADD COLUMN billing_period_end TIMESTAMP",
    ]
    postgres_patches = [
        "ALTER TABLE goal_alerts ALTER COLUMN impact TYPE VARCHAR(255)",
        "ALTER TABLE goal_alerts ALTER COLUMN evidence_strength TYPE VARCHAR(40)",
        "ALTER TABLE feature_comparisons ALTER COLUMN evidence_strength TYPE VARCHAR(40)",
        "ALTER TABLE gap_reports ALTER COLUMN evidence_strength TYPE VARCHAR(40)",
        # Supabase SQL sometimes created uuid ids; app uses text UUIDs
        "ALTER TABLE intel_embeddings ALTER COLUMN id TYPE text USING id::text",
        "ALTER TABLE intel_embeddings ALTER COLUMN agency_id TYPE text USING agency_id::text",
        "ALTER TABLE intel_embeddings ALTER COLUMN client_id TYPE text USING client_id::text",
        # Prefer jsonb over vector for optional embedding payload from the ORM
        "ALTER TABLE intel_embeddings ALTER COLUMN embedding TYPE jsonb USING NULL",
        # Supabase Auth: passwords live in Auth; app users.hashed_password is optional
        "ALTER TABLE users ALTER COLUMN hashed_password DROP NOT NULL",
    ]
    for stmt in common_patches + (postgres_patches if DATABASE_BACKEND == "postgres" else []):
        try:
            # One transaction per patch: an expected duplicate-column error must
            # not abort all later patches on Postgres.
            async with engine.begin() as conn:
                await conn.execute(text(stmt))
        except Exception as exc:
            logger.debug("Schema patch skipped (%s): %s", stmt[:60], exc)


async def init_db() -> None:
    from app import models  # noqa: F401

    await _ensure_pg_extensions()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _apply_schema_patches()
    # Log host only (never password) so Railway logs show pooler vs direct
    try:
        host = urlparse(DATABASE_URL).hostname
    except Exception:
        host = "?"
    logger.info("Database ready (%s, host=%s)", DATABASE_BACKEND, host)


async def ping_db() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
