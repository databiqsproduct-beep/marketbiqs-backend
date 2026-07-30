import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Agency,
    ClientBrand,
    Competitor,
    Insight,
    JobStatus,
    SentimentRecord,
    TrackingJob,
    TrendSignal,
)
from app.services import ai as ai_service
from app.services.embeddings import index_client_intel, retrieve_relevant
from app.services.tracking import scrape_radar_top_trends


async def run_client_intelligence(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    *,
    competitor_country: str | None = None,
) -> TrackingJob:
    """
    Radar intelligence — cheap Apify Google Trends pull (top 5, ~$0.01 cap).

    Does NOT run Instagram/TikTok/LinkedIn/Meta scrapers per competitor (those are expensive).
    """
    job = TrackingJob(
        agency_id=agency.id,
        client_id=client.id,
        job_type="full_intelligence",
        status=JobStatus.running,
        started_at=datetime.utcnow(),
        detail="Radar: fetching top 5 trends (Apify, cost-capped)",
    )
    db.add(job)
    await db.flush()

    try:
        competitors = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                    Competitor.is_tracking.is_(True),
                )
            )
        ).scalars().all()

        radar = await scrape_radar_top_trends(
            db,
            agency.id,
            country_hint=competitor_country,
            limit=5,
        )
        apify_trends = radar.get("trends") or []

        # Light AI pass: map scraped trends to this client's niche + optional sentiments/insights
        analysis = await ai_service.structured_json(
            db,
            agency.id,
            (
                "You are building a cheap Radar brief. "
                "Using the scraped Google Trends items, return JSON with keys: "
                "trends (exactly up to 5 objects: {topic, platform, velocity_score, summary, keywords[]}), "
                "sentiments (0-3 objects), insights (0-3 objects: {category, title, body, priority}). "
                "Prefer the scraped trend topics. Tie summaries to THIS client and rivals when possible. "
                "platform should be google_trends or multi. Keep it concise."
            ),
            json.dumps(
                {
                    "client": client.name,
                    "industry": client.industry,
                    "niche": client.niche,
                    "competitors": [{"name": c.name, "website": c.website} for c in competitors[:8]],
                    "apify_radar": {
                        "status": radar.get("status"),
                        "geo": radar.get("geo"),
                        "cost_cap_usd": radar.get("cost_cap_usd"),
                        "trends": apify_trends,
                    },
                }
            )[:8000],
            temperature=0.2,
        )

        # Prefer scraped trends; fall back to AI if Apify returned nothing
        trend_rows = apify_trends[:5]
        if not trend_rows and isinstance(analysis.get("trends"), list):
            trend_rows = [t for t in analysis.get("trends") if isinstance(t, dict)][:5]

        for trend in trend_rows[:5]:
            db.add(
                TrendSignal(
                    agency_id=agency.id,
                    client_id=client.id,
                    topic=str(trend.get("topic") or "Emerging topic")[:255],
                    platform=str(trend.get("platform") or "google_trends")[:60],
                    velocity_score=float(trend.get("velocity_score") or 0),
                    summary=str(trend.get("summary") or "")[:2000],
                    keywords=trend.get("keywords") if isinstance(trend.get("keywords"), list) else [],
                )
            )

        for sentiment in (analysis.get("sentiments") or [])[:3]:
            if not isinstance(sentiment, dict):
                continue
            db.add(
                SentimentRecord(
                    agency_id=agency.id,
                    client_id=client.id,
                    subject=sentiment.get("subject", client.name),
                    source=sentiment.get("source", "radar"),
                    score=float(sentiment.get("score") or 0),
                    label=sentiment.get("label", "neutral"),
                    themes=sentiment.get("themes") or [],
                    sample_quotes=sentiment.get("sample_quotes") or [],
                )
            )

        for insight in (analysis.get("insights") or [])[:3]:
            if not isinstance(insight, dict):
                continue
            db.add(
                Insight(
                    agency_id=agency.id,
                    client_id=client.id,
                    category=insight.get("category", "trend"),
                    title=insight.get("title", "Insight"),
                    body=insight.get("body", ""),
                    priority=insight.get("priority", "medium"),
                    source_refs=[{"source": "apify_radar", "geo": radar.get("geo")}],
                )
            )

        indexed = 0
        try:
            indexed = await index_client_intel(db, agency.id, client)
        except Exception:
            indexed = 0

        job.status = JobStatus.completed
        job.finished_at = datetime.utcnow()
        job.result_meta = {
            "mode": "apify_radar_cheap",
            "competitors": len(competitors),
            "snapshots": 0,
            "trends": len(trend_rows[:5]),
            "sentiments": min(3, len(analysis.get("sentiments") or [])),
            "insights": min(3, len(analysis.get("insights") or [])),
            "embeddings_indexed": indexed,
            "apify": {
                "status": radar.get("status"),
                "geo": radar.get("geo"),
                "actor": radar.get("actor"),
                "cost_cap_usd": radar.get("cost_cap_usd"),
                "detail": radar.get("detail"),
            },
        }
        job.detail = (
            f"Radar completed · top {len(trend_rows[:5])} trends "
            f"(Apify cost capped at ${radar.get('cost_cap_usd', 0.01)})"
        )
    except Exception as exc:
        job.status = JobStatus.failed
        job.finished_at = datetime.utcnow()
        job.detail = str(exc)[:800]
    await db.flush()
    return job


ASSISTANT_SYSTEM_PROMPT = """You are MarketBiqs, a friendly competitive intelligence assistant for marketing agencies.

Tone: warm, clear, and helpful — like a sharp teammate, not a stiff report bot.
Always answer in clean Markdown that is easy to skim:
- Open with a short, direct answer in plain language
- Use **bold** for brand names, metrics, and key takeaways
- Use bullet lists for findings; numbered lists for steps
- Use ### short headings when comparing rivals or themes
- When useful, end with a brief **What to do next** tip (1–3 bullets)

Ground every claim in the provided client workspace data and retrieved_memory.
If data is thin or missing, say so honestly and suggest running an intelligence refresh.
Stay specific to this client — no generic marketing fluff."""


async def _assistant_context(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    question: str,
) -> dict:
    trends = (
        await db.execute(
            select(TrendSignal)
            .where(TrendSignal.client_id == client.id, TrendSignal.agency_id == agency.id)
            .order_by(TrendSignal.detected_at.desc())
            .limit(8)
        )
    ).scalars().all()
    insights = (
        await db.execute(
            select(Insight)
            .where(Insight.client_id == client.id, Insight.agency_id == agency.id)
            .order_by(Insight.created_at.desc())
            .limit(8)
        )
    ).scalars().all()
    sentiments = (
        await db.execute(
            select(SentimentRecord)
            .where(SentimentRecord.client_id == client.id, SentimentRecord.agency_id == agency.id)
            .order_by(SentimentRecord.created_at.desc())
            .limit(8)
        )
    ).scalars().all()
    competitors = (
        await db.execute(select(Competitor).where(Competitor.client_id == client.id, Competitor.agency_id == agency.id))
    ).scalars().all()

    rag = await retrieve_relevant(db, agency.id, client.id, question, limit=8)
    return {
        "client": client.name,
        "industry": client.industry,
        "competitors": [c.name for c in competitors],
        "trends": [{"topic": t.topic, "summary": t.summary} for t in trends],
        "insights": [{"title": i.title, "body": i.body} for i in insights],
        "sentiments": [{"subject": s.subject, "label": s.label, "themes": s.themes} for s in sentiments],
        "retrieved_memory": rag,
    }


def _assistant_user_prompt(question: str, context: dict) -> str:
    return (
        f"Client question:\n{question}\n\n"
        f"Workspace intelligence (JSON):\n{json.dumps(context)[:12000]}"
    )


async def answer_client_question(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    context = await _assistant_context(db, agency, client, question)
    return await ai_service.chat_completion(
        db,
        agency.id,
        ASSISTANT_SYSTEM_PROMPT,
        _assistant_user_prompt(question, context),
        temperature=0.45,
        history=history,
    )


async def stream_client_question(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    question: str,
    history: list[dict[str, str]] | None = None,
):
    context = await _assistant_context(db, agency, client, question)
    async for delta in ai_service.chat_completion_stream(
        db,
        agency.id,
        ASSISTANT_SYSTEM_PROMPT,
        _assistant_user_prompt(question, context),
        temperature=0.45,
        history=history,
    ):
        yield delta
