"""User Actions Hub — single control point for high-level user-triggered actions."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Agency, ClientBrand, Report
from app.services.competitive import run_full_ai_pipeline
from app.services.delivery import deliver_update
from app.services.ingestion import run_ingestion_hub
from app.services.reports import generate_client_report


async def action_run_intel(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    *,
    push_jira: bool = False,
    generate_report: bool = True,
    competitor_scope: str = "local",
    competitor_country: str | None = None,
    competitor_count: int = 5,
    competitor_mode: str = "add",
) -> dict[str, Any]:
    return await run_full_ai_pipeline(
        db,
        agency,
        client,
        push_jira=push_jira,
        generate_report=generate_report,
        competitor_scope=competitor_scope,
        competitor_country=competitor_country,
        competitor_count=competitor_count,
        competitor_mode=competitor_mode,
    )


async def action_ingest_client(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
) -> dict[str, Any]:
    return await run_ingestion_hub(
        db,
        agency.id,
        website=client.website,
        serp_query=f"{client.name} competitors alternatives",
        reddit_query=f"{client.name} OR {client.industry or ''} competitors",
    )


async def action_generate_report(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    period_label: str = "Weekly Brief",
) -> Report:
    return await generate_client_report(db, agency, client, period_label=period_label)


async def action_deliver(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    *,
    report: Report | None = None,
    channel: str | None = None,
    message: str | None = None,
) -> list:
    return await deliver_update(
        db,
        agency.id,
        agency.name,
        client,
        report=report,
        channel=channel,
        custom_message=message,
    )
