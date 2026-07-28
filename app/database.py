from collections.abc import AsyncGenerator
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

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


def _resolve_database_url() -> tuple[str, str]:
    """
    Production: Supabase Postgres when DATABASE_URL or SUPABASE_DB_PASSWORD is set.
    Local/dev: SQLite (./biqs.db) when Postgres is not configured — matches architecture PDF.
    """
    url = (settings.database_url or "").strip()
    if url:
        if "sqlite" in url:
            return url if "aiosqlite" in url else url.replace("sqlite://", "sqlite+aiosqlite://", 1), "sqlite"
        return _normalize_postgres_url(url), "postgres"

    if settings.supabase_db_url:
        return _normalize_postgres_url(settings.supabase_db_url), "postgres"

    if settings.supabase_db_password and settings.supabase_project_ref:
        ref = settings.supabase_project_ref
        region = settings.supabase_db_region or "us-east-1"
        password = settings.supabase_db_password
        host = f"aws-0-{region}.pooler.supabase.com"
        built = f"postgresql+asyncpg://postgres.{ref}:{password}@{host}:6543/postgres"
        return built, "postgres"

    # Architecture PDF: development uses SQLite until Supabase credentials are provided.
    logger.warning(
        "Supabase Postgres not configured (set DATABASE_URL or SUPABASE_DB_PASSWORD). "
        "Using local SQLite biqs.db for development."
    )
    return "sqlite+aiosqlite:///./biqs.db", "sqlite"


DATABASE_URL, DATABASE_BACKEND = _resolve_database_url()

_engine_kwargs: dict = {"echo": False, "future": True}
if DATABASE_BACKEND == "postgres":
    _engine_kwargs.update(pool_pre_ping=True, pool_size=5, max_overflow=10)
else:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


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
    async with engine.begin() as conn:
        await conn.execute(text('create extension if not exists "pgcrypto"'))
        try:
            await conn.execute(text('create extension if not exists "vector"'))
        except Exception:
            logger.warning("pgvector extension unavailable; text RAG still works without vectors")


async def init_db() -> None:
    from app import models  # noqa: F401

    await _ensure_pg_extensions()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database ready (%s)", DATABASE_BACKEND)


async def ping_db() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
