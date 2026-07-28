"""Supabase access via SUPABASE_URL + publishable/secret keys.

Works the same on localhost and production — only env values change.
Backend always uses the secret key (bypasses RLS). Frontend uses the publishable key.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger("marketbiqs.supabase")


class SupabaseNotConfigured(RuntimeError):
    pass


def supabase_configured() -> bool:
    return get_settings().supabase_ready


def _headers(secret: bool = True) -> dict[str, str]:
    s = get_settings()
    key = s.resolved_secret_key() if secret else s.resolved_publishable_key()
    if not s.supabase_url or not key:
        raise SupabaseNotConfigured(
            "Set SUPABASE_URL and SUPABASE_SECRET_KEY (backend) or SUPABASE_PUBLISHABLE_KEY (public) in env."
        )
    # New sb_* keys go in apikey. Authorization Bearer is set to the same value for SDK compatibility.
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def rest_base() -> str:
    s = get_settings()
    return (s.supabase_url or "").rstrip("/") + "/rest/v1"


def storage_base() -> str:
    s = get_settings()
    return (s.supabase_url or "").rstrip("/") + "/storage/v1"


@lru_cache
def get_supabase_admin():
    """Server-side client (secret key). Never import this in frontend code."""
    from supabase import create_client

    s = get_settings()
    secret = s.resolved_secret_key()
    if not s.supabase_url or not secret:
        raise SupabaseNotConfigured("SUPABASE_URL and SUPABASE_SECRET_KEY are required")
    return create_client(s.supabase_url.rstrip("/"), secret)


@lru_cache
def get_supabase_public():
    """Publishable-key client for non-privileged server reads when needed."""
    from supabase import create_client

    s = get_settings()
    pub = s.resolved_publishable_key()
    if not s.supabase_url or not pub:
        raise SupabaseNotConfigured("SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are required")
    return create_client(s.supabase_url.rstrip("/"), pub)


async def ping_supabase() -> dict[str, Any]:
    """Verify project reachability with the secret key (works on localhost + production)."""
    s = get_settings()
    if not supabase_configured():
        return {"configured": False, "ok": False, "detail": "SUPABASE_URL / SUPABASE_SECRET_KEY missing"}
    url = s.supabase_url.rstrip("/") + "/rest/v1/"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, headers=_headers(secret=True))
        ok = response.status_code < 500
        return {
            "configured": True,
            "ok": ok,
            "status_code": response.status_code,
            "project": s.supabase_url,
            "detail": "reachable" if ok else (response.text or "")[:200],
        }
    except Exception as exc:
        return {"configured": True, "ok": False, "detail": str(exc)[:300]}


async def ensure_reports_bucket() -> bool:
    """Create public-or-private reports bucket if missing."""
    if not supabase_configured():
        return False
    s = get_settings()
    headers = _headers(secret=True)
    bucket = "reports"
    async with httpx.AsyncClient(timeout=20) as client:
        listed = await client.get(f"{storage_base()}/bucket", headers=headers)
        if listed.status_code < 400:
            names = [b.get("name") for b in (listed.json() or []) if isinstance(b, dict)]
            if bucket in names:
                return True
        created = await client.post(
            f"{storage_base()}/bucket",
            headers=headers,
            json={"id": bucket, "name": bucket, "public": False},
        )
        if created.status_code < 400 or created.status_code in (409, 400):
            return True
        logger.warning("Could not ensure reports bucket: %s %s", created.status_code, created.text[:200])
        return False


async def upload_report_pdf(report_id: str, data: bytes, content_type: str = "application/pdf") -> str | None:
    """Upload PDF bytes to Supabase Storage. Returns public/signed path or None."""
    if not supabase_configured():
        return None
    await ensure_reports_bucket()
    path = f"{report_id}.pdf"
    headers = _headers(secret=True)
    headers.pop("Content-Type", None)
    headers["Content-Type"] = content_type
    headers["x-upsert"] = "true"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{storage_base()}/object/reports/{path}",
            headers=headers,
            content=data,
        )
        if response.status_code >= 400:
            # try update
            response = await client.put(
                f"{storage_base()}/object/reports/{path}",
                headers=headers,
                content=data,
            )
        if response.status_code >= 400:
            logger.warning("Supabase storage upload failed: %s %s", response.status_code, response.text[:200])
            return None
    return f"supabase://reports/{path}"


async def download_report_pdf_bytes(storage_ref: str) -> bytes | None:
    """Download a report PDF previously uploaded as supabase://reports/{id}.pdf."""
    if not storage_ref.startswith("supabase://reports/"):
        return None
    if not supabase_configured():
        return None
    path = storage_ref.replace("supabase://reports/", "", 1)
    headers = _headers(secret=True)
    headers.pop("Content-Type", None)
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(f"{storage_base()}/object/reports/{path}", headers=headers)
        if response.status_code >= 400:
            logger.warning("Supabase storage download failed: %s", response.status_code)
            return None
        return response.content


async def rest_select(table: str, *, params: dict[str, str] | None = None, limit: int = 50) -> list[dict]:
    """Generic PostgREST select using secret key (service role)."""
    headers = _headers(secret=True)
    query = {"select": "*", "limit": str(limit)}
    if params:
        query.update(params)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{rest_base()}/{table}", headers=headers, params=query)
        if response.status_code >= 400:
            raise RuntimeError(f"Supabase REST {table}: {response.status_code} {response.text[:300]}")
        data = response.json()
        return data if isinstance(data, list) else []
