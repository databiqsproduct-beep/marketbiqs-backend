"""pgvector-ready intel memory for GPT workspace RAG."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    ClientBrand,
    FeatureComparison,
    GapReport,
    GoalAlert,
    Insight,
    IntelEmbedding,
    TrendSignal,
)


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}


def _score(query: str, content: str) -> float:
    q = _tokenize(query)
    c = _tokenize(content)
    if not q or not c:
        return 0.0
    return len(q & c) / max(len(q), 1)


async def index_client_intel(db: AsyncSession, agency_id: str, client: ClientBrand) -> int:
    """Snapshot key intel texts into intel_embeddings for assistant RAG."""
    chunks: list[tuple[str, str, dict[str, Any]]] = []

    insights = (
        await db.execute(
            select(Insight)
            .where(Insight.client_id == client.id, Insight.agency_id == agency_id)
            .order_by(Insight.created_at.desc())
            .limit(20)
        )
    ).scalars().all()
    for row in insights:
        chunks.append(("insight", f"{row.title}\n{row.body}", {"id": row.id, "priority": row.priority}))

    trends = (
        await db.execute(
            select(TrendSignal)
            .where(TrendSignal.client_id == client.id, TrendSignal.agency_id == agency_id)
            .order_by(TrendSignal.detected_at.desc())
            .limit(15)
        )
    ).scalars().all()
    for row in trends:
        chunks.append(("trend", f"{row.topic}\n{row.summary}", {"id": row.id, "platform": row.platform}))

    gaps = (
        await db.execute(
            select(GapReport)
            .where(GapReport.client_id == client.id, GapReport.agency_id == agency_id)
            .order_by(GapReport.created_at.desc())
            .limit(15)
        )
    ).scalars().all()
    for row in gaps:
        chunks.append(
            (
                "gap",
                f"{row.summary}\nLeading: {', '.join(row.leading or [])}\n"
                f"Opportunities: {', '.join(row.opportunities or [])}",
                {"id": row.id},
            )
        )

    alerts = (
        await db.execute(
            select(GoalAlert)
            .where(GoalAlert.client_id == client.id, GoalAlert.agency_id == agency_id)
            .order_by(GoalAlert.created_at.desc())
            .limit(15)
        )
    ).scalars().all()
    for row in alerts:
        chunks.append(("alert", f"{row.title}\n{row.why_it_matters}\n{row.action}", {"id": row.id}))

    comparisons = (
        await db.execute(
            select(FeatureComparison)
            .where(FeatureComparison.client_id == client.id, FeatureComparison.agency_id == agency_id)
            .order_by(FeatureComparison.created_at.desc())
            .limit(25)
        )
    ).scalars().all()
    for row in comparisons:
        chunks.append(
            (
                "comparison",
                f"{row.feature_name}: ours={row.our_status} theirs={row.competitor_status}. "
                f"{row.how_competitor_leads or ''} {row.how_to_improve or ''}",
                {"id": row.id},
            )
        )

    written = 0
    for source, content, meta in chunks:
        content = (content or "").strip()
        if len(content) < 20:
            continue
        db.add(
            IntelEmbedding(
                agency_id=agency_id,
                client_id=client.id,
                source=source,
                content=content[:4000],
                meta=meta,
            )
        )
        written += 1
    await db.flush()
    return written


async def retrieve_relevant(
    db: AsyncSession,
    agency_id: str,
    client_id: str,
    query: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            select(IntelEmbedding)
            .where(IntelEmbedding.agency_id == agency_id, IntelEmbedding.client_id == client_id)
            .order_by(IntelEmbedding.created_at.desc())
            .limit(200)
        )
    ).scalars().all()
    ranked = sorted(rows, key=lambda r: _score(query, r.content), reverse=True)
    out: list[dict[str, Any]] = []
    for row in ranked[:limit]:
        if _score(query, row.content) <= 0 and out:
            break
        out.append({"source": row.source, "content": row.content[:900], "meta": row.meta or {}})
    return out
