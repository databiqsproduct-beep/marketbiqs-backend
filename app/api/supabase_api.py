"""Expose public Supabase config for localhost + production frontends."""

from fastapi import APIRouter

from app.config import get_settings
from app.services.supabase_client import ping_supabase, supabase_configured

router = APIRouter(prefix="/supabase", tags=["supabase"])


@router.get("/config")
async def supabase_public_config():
    s = get_settings()
    return {
        "configured": supabase_configured(),
        "url": s.supabase_url,
        "publishable_key": s.resolved_publishable_key(),
        "env": s.app_env,
    }


@router.get("/status")
async def supabase_status():
    return await ping_supabase()
