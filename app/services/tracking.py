from datetime import datetime
from typing import Any

import httpx
from apify_client import ApifyClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Agency, ApiKeyVault, Competitor, CompetitorSnapshot, UsageEvent
from app.security import decrypt_secret
from sqlalchemy import select

settings = get_settings()


async def _vault_key(db: AsyncSession, agency_id: str, provider: str) -> str | None:
    stmt = select(ApiKeyVault).where(
        ApiKeyVault.agency_id == agency_id,
        ApiKeyVault.provider == provider,
        ApiKeyVault.is_active.is_(True),
    )
    result = await db.execute(stmt)
    vault = result.scalar_one_or_none()
    if vault:
        return decrypt_secret(vault.encrypted_key)
    return None


async def resolve_apify(db: AsyncSession, agency_id: str) -> str:
    return (await _vault_key(db, agency_id, "apify")) or settings.apify_key


async def resolve_serp(db: AsyncSession, agency_id: str) -> str:
    return (await _vault_key(db, agency_id, "serpapi")) or settings.serp_api


async def resolve_firecrawl(db: AsyncSession, agency_id: str) -> str:
    return (await _vault_key(db, agency_id, "firecrawl")) or settings.firecrawl_api_key


async def track_usage(db: AsyncSession, agency_id: str, event_type: str, units: int = 1, meta: dict | None = None) -> None:
    db.add(
        UsageEvent(
            agency_id=agency_id,
            event_type=event_type,
            units=units,
            meta=meta or {},
        )
    )
    agency = await db.get(Agency, agency_id)
    if agency:
        agency.scrape_units_used = (agency.scrape_units_used or 0) + max(1, units)


async def ensure_scrape_quota(db: AsyncSession, agency_id: str) -> None:
    agency = await db.get(Agency, agency_id)
    if agency and agency.scrape_units_used >= agency.scrape_quota:
        raise ValueError("Scrape quota exceeded. Purchase client packs or raise scrape quota.")


async def scrape_website(db: AsyncSession, agency_id: str, url: str) -> dict[str, Any]:
    key = await resolve_firecrawl(db, agency_id)
    if not key or not url:
        return {"url": url, "markdown": "", "status": "skipped", "note": "Missing Firecrawl key or URL"}
    await ensure_scrape_quota(db, agency_id)
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        )
        if response.status_code >= 400:
            return {"url": url, "status": "error", "detail": response.text[:500]}
        data = response.json()
        await track_usage(db, agency_id, "firecrawl_scrape", 1, {"url": url})
        markdown = ((data.get("data") or {}).get("markdown")) or ""
        return {"url": url, "markdown": markdown[:12000], "status": "ok"}


async def serp_visibility(db: AsyncSession, agency_id: str, query: str) -> dict[str, Any]:
    key = await resolve_serp(db, agency_id)
    if not key:
        return {"query": query, "status": "skipped", "organic": []}
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "api_key": key, "num": 10},
        )
        if response.status_code >= 400:
            return {"query": query, "status": "error", "detail": response.text[:500]}
        data = response.json()
        await track_usage(db, agency_id, "serp_search", 1, {"query": query})
        organic = [
            {"position": i.get("position"), "title": i.get("title"), "link": i.get("link"), "snippet": i.get("snippet")}
            for i in data.get("organic_results", [])[:10]
        ]
        return {"query": query, "status": "ok", "organic": organic}


async def run_apify_actor(
    db: AsyncSession,
    agency_id: str,
    actor_id: str,
    run_input: dict[str, Any],
) -> dict[str, Any]:
    token = await resolve_apify(db, agency_id)
    if not token:
        return {"status": "skipped", "items": [], "note": "Missing Apify token"}
    try:
        client = ApifyClient(token)
        run = client.actor(actor_id).call(run_input=run_input, timeout_secs=90)
        dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else None
        items: list[dict[str, Any]] = []
        if dataset_id:
            for item in client.dataset(dataset_id).iterate_items(limit=25):
                items.append(item)
        await track_usage(db, agency_id, "apify_run", max(1, len(items)), {"actor": actor_id})
        return {"status": "ok", "items": items}
    except Exception as exc:
        return {"status": "error", "items": [], "detail": str(exc)[:500]}


async def scrape_competitor(db: AsyncSession, competitor: Competitor) -> list[CompetitorSnapshot]:
    snapshots: list[CompetitorSnapshot] = []
    agency_id = competitor.agency_id

    if competitor.website:
        web = await scrape_website(db, agency_id, competitor.website)
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="website",
                payload=web,
                summary=f"Website scan for {competitor.name}",
            )
        )
        serp = await serp_visibility(db, agency_id, competitor.name)
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="seo",
                payload=serp,
                summary=f"SEO visibility for {competitor.name}",
            )
        )

    if competitor.instagram_handle:
        ig = await run_apify_actor(
            db,
            agency_id,
            "apify~instagram-scraper",
            {"directUrls": [f"https://www.instagram.com/{competitor.instagram_handle.strip('@')}/"], "resultsLimit": 10},
        )
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="instagram",
                payload=ig,
                summary=f"Instagram activity for {competitor.name}",
            )
        )

    if competitor.tiktok_handle:
        tt = await run_apify_actor(
            db,
            agency_id,
            "clockworks~tiktok-scraper",
            {"profiles": [competitor.tiktok_handle.strip("@")], "resultsPerPage": 10},
        )
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="tiktok",
                payload=tt,
                summary=f"TikTok activity for {competitor.name}",
            )
        )

    if competitor.meta_ads_query or competitor.name:
        ads = await run_apify_actor(
            db,
            agency_id,
            "apify~facebook-ads-scraper",
            {"startUrls": [], "query": competitor.meta_ads_query or competitor.name, "maxAds": 15},
        )
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="meta_ads",
                payload=ads,
                summary=f"Meta ads for {competitor.name}",
            )
        )

    if competitor.linkedin_url:
        li = await run_apify_actor(
            db,
            agency_id,
            "harvestapi~linkedin-profile-posts",
            {"urls": [competitor.linkedin_url], "maxPosts": 10},
        )
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="linkedin",
                payload=li,
                summary=f"LinkedIn posts for {competitor.name}",
            )
        )

    competitor.last_scraped_at = datetime.utcnow()
    for snap in snapshots:
        db.add(snap)
    return snapshots
