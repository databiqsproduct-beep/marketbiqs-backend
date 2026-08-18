"""Product help desk — platform GROQ_API_KEY; FAQs + client/competitor intel when available."""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.deps import AuthContext, get_auth_context
from app.models import ClientBrand, Competitor
from app.services.ai import platform_chat_completion_stream
from app.services.helpdesk_knowledge import (
    build_help_system_prompt,
    faqs_public,
    issues_public,
)
from app.services.intelligence import _assistant_context

logger = logging.getLogger("marketbiqs.helpdesk")
router = APIRouter(prefix="/helpdesk", tags=["helpdesk"])
settings = get_settings()

MISSING_KEY = (
    "Help desk AI is not configured yet. Add GROQ_API_KEY on the MarketBiqs API "
    "(platform key — not agency BYOK), restart the API, then try again."
)


class HelpdeskChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    topic: str | None = None
    client_id: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


def _trim_history(history: list[dict[str, str]], limit: int = 10) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for row in history or []:
        role = (row.get("role") or "").strip()
        content = (row.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:2000]})
    return cleaned[-limit:]


async def _portfolio_brief(db: AsyncSession, agency_id: str) -> list[dict]:
    clients = (
        await db.execute(
            select(ClientBrand)
            .where(ClientBrand.agency_id == agency_id, ClientBrand.is_active.is_(True))
            .order_by(ClientBrand.created_at.desc())
            .limit(40)
        )
    ).scalars().all()
    brief: list[dict] = []
    for client in clients:
        comps = (
            await db.execute(
                select(Competitor.name).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency_id,
                )
            )
        ).scalars().all()
        brief.append(
            {
                "id": client.id,
                "name": client.name,
                "industry": client.industry,
                "website": client.website,
                "competitors": [n for n in comps if n][:25],
            }
        )
    return brief


def _match_client(brief: list[dict], message: str, client_id: str | None) -> dict | None:
    if client_id:
        for row in brief:
            if row["id"] == client_id:
                return row
    text = (message or "").lower()
    # Prefer longer names first to avoid partial collisions
    ranked = sorted(brief, key=lambda r: len(r.get("name") or ""), reverse=True)
    for row in ranked:
        name = (row.get("name") or "").strip()
        if len(name) < 2:
            continue
        if name.lower() in text:
            return row
        # soft token match for multi-word brands
        tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if len(t) >= 3]
        if tokens and all(t in text for t in tokens):
            return row
    # competitor name → owning client
    for row in ranked:
        for comp in row.get("competitors") or []:
            cname = str(comp).strip()
            if len(cname) >= 3 and cname.lower() in text:
                return row
    return None


async def _build_user_prompt(
    db: AsyncSession,
    ctx: AuthContext,
    payload: HelpdeskChatRequest,
) -> str:
    brief = await _portfolio_brief(db, ctx.agency.id)
    matched = _match_client(brief, payload.message, payload.client_id)
    topic = (payload.topic or "").strip()

    parts = [
        f"Topic: {topic}" if topic else None,
        f"User question:\n{payload.message.strip()}",
        "Agency portfolio (clients + tracked competitors):\n"
        + json.dumps(
            [
                {
                    "name": r["name"],
                    "industry": r.get("industry"),
                    "competitors": r.get("competitors") or [],
                }
                for r in brief
            ],
            ensure_ascii=False,
        )[:8000],
    ]

    if matched:
        client = await db.get(ClientBrand, matched["id"])
        if client and client.agency_id == ctx.agency.id:
            intel = await _assistant_context(db, ctx.agency, client, payload.message)
            parts.append(
                "Focused client workspace intelligence (JSON) — use this for client/competitor questions:\n"
                + json.dumps(intel, ensure_ascii=False)[:12000]
            )
            parts.append(
                f"Focused client: {client.name}. Answer with this client's rivals/intel when relevant."
            )
    else:
        parts.append(
            "No single client was selected/matched. For portfolio-level questions use the list above. "
            "If they ask about one brand, ask which client — or tell them to pick a client in Help desk."
        )

    return "\n\n".join(p for p in parts if p)


@router.get("/knowledge")
async def helpdesk_knowledge(ctx: AuthContext = Depends(get_auth_context)):
    """Compact FAQ + issue labels for the help desk quick-ask menu."""
    _ = ctx
    return {
        "faqs": faqs_public()[:8],
        "common_issues": issues_public()[:6],
    }


@router.post("/chat")
async def helpdesk_chat(
    payload: HelpdeskChatRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    if not (settings.groq_api_key or "").strip():
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured for help desk")
    system = build_help_system_prompt()
    user_text = await _build_user_prompt(db, ctx, payload)
    chunks: list[str] = []
    async for delta in platform_chat_completion_stream(
        system,
        user_text,
        history=_trim_history(payload.history),
        missing_key_message=MISSING_KEY,
    ):
        chunks.append(delta)
    return {"role": "assistant", "content": "".join(chunks).strip()}


@router.post("/chat/stream")
async def helpdesk_chat_stream(
    payload: HelpdeskChatRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    history = _trim_history(payload.history)
    system = build_help_system_prompt()
    user_text = await _build_user_prompt(db, ctx, payload)

    async def event_gen():
        full: list[str] = []
        try:
            async for delta in platform_chat_completion_stream(
                system,
                user_text,
                history=history,
                missing_key_message=MISSING_KEY,
            ):
                full.append(delta)
                yield f"data: {json.dumps({'type': 'delta', 'content': delta})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'content': ''.join(full).strip()})}\n\n"
        except Exception as exc:
            logger.exception("Help desk stream failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
