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


def _pooler_url(ref: str, password: str, region: str | None = None) -> str:
    region = (region or settings.supabase_db_region or "us-east-1").strip()
    host = f"aws-0-{region}.pooler.supabase.com"
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

    Always prefer the Supabase pooler (6543) — required for Railway IPv4.
    """
    url = (settings.database_url or "").strip()

    # Fix the common Railway failure: direct db.*:5432 → pooler :6543
    if url:
        rewritten = _rewrite_direct_supabase_to_pooler(url)
        if rewritten:
            logger.warning(
                "DATABASE_URL used direct Supabase host (IPv6) — rewritten to pooler :6543 for Railway"
            )
            return rewritten, "postgres"

    pooler = _build_supabase_pooler_url()

    # Prefer password+ref pooler over a non-pooler DATABASE_URL on cloud
    if pooler and (not url or "pooler.supabase.com" not in url):
        if url and "pooler.supabase.com" not in url:
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

    if pooler:
        return pooler, "postgres"

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
        pool_size=5,
        max_overflow=10,
        connect_args={
            "timeout": 10,
            "command_timeout": 15,
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


async def init_db() -> None:
    from app import models  # noqa: F401

    await _ensure_pg_extensions()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
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
