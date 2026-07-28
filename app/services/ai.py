import json
from typing import Any

from groq import Groq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import ApiKeyVault
from app.security import decrypt_secret

settings = get_settings()


async def resolve_groq_key(db: AsyncSession, agency_id: str) -> str:
    stmt = select(ApiKeyVault).where(
        ApiKeyVault.agency_id == agency_id,
        ApiKeyVault.provider == "groq",
        ApiKeyVault.is_active.is_(True),
    )
    result = await db.execute(stmt)
    vault = result.scalar_one_or_none()
    if vault:
        return decrypt_secret(vault.encrypted_key)
    return settings.groq_api_key


def _client(api_key: str) -> Groq | None:
    if not api_key:
        return None
    return Groq(api_key=api_key)


async def chat_completion(
    db: AsyncSession,
    agency_id: str,
    system: str,
    user: str,
    temperature: float = 0.3,
) -> str:
    api_key = await resolve_groq_key(db, agency_id)
    client = _client(api_key)
    if not client:
        return _fallback_text(system, user)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or _fallback_text(system, user)
    except Exception:
        return _fallback_text(system, user)


async def structured_json(
    db: AsyncSession,
    agency_id: str,
    system: str,
    user: str,
    temperature: float = 0.25,
) -> dict[str, Any]:
    raw = await chat_completion(
        db,
        agency_id,
        system + " Respond with valid JSON only. No markdown fences.",
        user,
        temperature=temperature,
    )
    try:
        start_obj = raw.find("{")
        end_obj = raw.rfind("}")
        start_arr = raw.find("[")
        end_arr = raw.rfind("]")
        if start_obj >= 0 and end_obj > start_obj and (start_arr < 0 or start_obj < start_arr):
            return json.loads(raw[start_obj : end_obj + 1])
        if start_arr >= 0 and end_arr > start_arr:
            return {"items": json.loads(raw[start_arr : end_arr + 1])}
    except Exception:
        pass
    return {"summary": raw, "sections": []}


def _fallback_text(system: str, user: str) -> str:
    return (
        "Competitive intelligence briefing prepared from available workspace data. "
        f"Focus: {user[:400]}. Guidance: {system[:200]}"
    )
