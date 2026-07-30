import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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
from app.services.ingestion import enrich_competitor_via_hub
from app.services.tracking import scrape_competitor


async def run_client_intelligence(db: AsyncSession, agency: Agency, client: ClientBrand) -> TrackingJob:
    job = TrackingJob(
        agency_id=agency.id,
        client_id=client.id,
        job_type="full_intelligence",
        status=JobStatus.running,
        started_at=datetime.utcnow(),
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

        snapshot_count = 0
        for competitor in competitors:
            try:
                snaps = await enrich_competitor_via_hub(db, competitor)
            except Exception:
                snaps = await scrape_competitor(db, competitor)
            snapshot_count += len(snaps)

        analysis = await ai_service.structured_json(
            db,
            agency.id,
            (
                "Analyze competitor and market signals for a marketing agency client. "
                "Return JSON with keys: trends (array of {topic, platform, velocity_score, summary, keywords[]}), "
                "sentiments (array of {subject, source, score, label, themes[], sample_quotes[]}), "
                "insights (array of {category, title, body, priority})."
            ),
            json.dumps(
                {
                    "client": client.name,
                    "industry": client.industry,
                    "competitors": [
                        {
                            "name": c.name,
                            "website": c.website,
                            "instagram": c.instagram_handle,
                            "tiktok": c.tiktok_handle,
                        }
                        for c in competitors
                    ],
                }
            ),
        )

        for trend in analysis.get("trends", [])[:8]:
            db.add(
                TrendSignal(
                    agency_id=agency.id,
                    client_id=client.id,
                    topic=trend.get("topic", "Emerging topic"),
                    platform=trend.get("platform", "multi"),
                    velocity_score=float(trend.get("velocity_score") or 0),
                    summary=trend.get("summary", ""),
                    keywords=trend.get("keywords") or [],
                )
            )

        for sentiment in analysis.get("sentiments", [])[:8]:
            db.add(
                SentimentRecord(
                    agency_id=agency.id,
                    client_id=client.id,
                    subject=sentiment.get("subject", client.name),
                    source=sentiment.get("source", "reviews"),
                    score=float(sentiment.get("score") or 0),
                    label=sentiment.get("label", "neutral"),
                    themes=sentiment.get("themes") or [],
                    sample_quotes=sentiment.get("sample_quotes") or [],
                )
            )

        for insight in analysis.get("insights", [])[:10]:
            db.add(
                Insight(
                    agency_id=agency.id,
                    client_id=client.id,
                    category=insight.get("category", "competitor"),
                    title=insight.get("title", "Insight"),
                    body=insight.get("body", ""),
                    priority=insight.get("priority", "medium"),
                    source_refs=[],
                )
            )

        indexed = await index_client_intel(db, agency.id, client)
        job.status = JobStatus.completed
        job.finished_at = datetime.utcnow()
        job.result_meta = {
            "competitors": len(competitors),
            "snapshots": snapshot_count,
            "trends": len(analysis.get("trends", [])),
            "sentiments": len(analysis.get("sentiments", [])),
            "insights": len(analysis.get("insights", [])),
            "embeddings_indexed": indexed,
        }
        job.detail = "Intelligence run completed"
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
