"""Ingestion Hub — single entry point for all data collection sources."""

from __future__ import annotations

from typing import Any

import feedparser
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Competitor, CompetitorSnapshot
from app.services.tracking import (
    run_apify_actor,
    scrape_website,
    serp_visibility,
    track_usage,
)


async def ingest_website(db: AsyncSession, agency_id: str, url: str) -> dict[str, Any]:
    return await scrape_website(db, agency_id, url)


async def ingest_serp(db: AsyncSession, agency_id: str, query: str) -> dict[str, Any]:
    return await serp_visibility(db, agency_id, query)


async def ingest_reddit(db: AsyncSession, agency_id: str, query: str) -> dict[str, Any]:
    """Pull public Reddit search results (no PRAW key required for basic JSON)."""
    q = (query or "").strip()
    if not q:
        return {"status": "skipped", "items": [], "note": "Empty query"}
    url = "https://www.reddit.com/search.json"
    try:
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "MarketBiqs/1.0"}) as client:
            response = await client.get(url, params={"q": q, "limit": 15, "sort": "relevance"})
            if response.status_code >= 400:
                return {"status": "error", "items": [], "detail": response.text[:400]}
            children = ((response.json() or {}).get("data") or {}).get("children") or []
            items = []
            for child in children[:15]:
                data = child.get("data") or {}
                items.append(
                    {
                        "title": data.get("title"),
                        "subreddit": data.get("subreddit"),
                        "score": data.get("score"),
                        "url": f"https://reddit.com{data.get('permalink') or ''}",
                        "snippet": (data.get("selftext") or "")[:280],
                    }
                )
            await track_usage(db, agency_id, "reddit_search", 1, {"query": q})
            return {"status": "ok", "query": q, "items": items}
    except Exception as exc:
        return {"status": "error", "items": [], "detail": str(exc)[:400]}


async def ingest_rss(db: AsyncSession, agency_id: str, feed_url: str) -> dict[str, Any]:
    if not feed_url:
        return {"status": "skipped", "items": [], "note": "Missing feed URL"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(feed_url)
            if response.status_code >= 400:
                return {"status": "error", "items": [], "detail": response.text[:400]}
            parsed = feedparser.parse(response.text)
            items = [
                {
                    "title": getattr(entry, "title", ""),
                    "link": getattr(entry, "link", ""),
                    "summary": (getattr(entry, "summary", "") or "")[:300],
                    "published": getattr(entry, "published", ""),
                }
                for entry in (parsed.entries or [])[:20]
            ]
            await track_usage(db, agency_id, "rss_feed", 1, {"feed": feed_url})
            return {"status": "ok", "feed": feed_url, "items": items}
    except Exception as exc:
        return {"status": "error", "items": [], "detail": str(exc)[:400]}


async def ingest_x_twitter(db: AsyncSession, agency_id: str, handle: str) -> dict[str, Any]:
    handle = (handle or "").strip().lstrip("@")
    if not handle:
        return {"status": "skipped", "items": [], "note": "Missing X handle"}
    return await run_apify_actor(
        db,
        agency_id,
        "apidojo~tweet-scraper",
        {"twitterHandles": [handle], "maxItems": 15},
    )


async def ingest_facebook(db: AsyncSession, agency_id: str, page: str) -> dict[str, Any]:
    page = (page or "").strip()
    if not page:
        return {"status": "skipped", "items": [], "note": "Missing Facebook page"}
    url = page if page.startswith("http") else f"https://www.facebook.com/{page}"
    return await run_apify_actor(
        db,
        agency_id,
        "apify~facebook-pages-scraper",
        {"startUrls": [{"url": url}], "maxPosts": 15},
    )


async def run_ingestion_hub(
    db: AsyncSession,
    agency_id: str,
    *,
    website: str | None = None,
    serp_query: str | None = None,
    reddit_query: str | None = None,
    rss_url: str | None = None,
    twitter_handle: str | None = None,
    facebook_page: str | None = None,
) -> dict[str, Any]:
    """Central traffic controller for all scrapers."""
    result: dict[str, Any] = {"sources": {}}
    if website:
        result["sources"]["firecrawl"] = await ingest_website(db, agency_id, website)
    if serp_query:
        result["sources"]["serpapi"] = await ingest_serp(db, agency_id, serp_query)
    if reddit_query:
        result["sources"]["reddit"] = await ingest_reddit(db, agency_id, reddit_query)
    if rss_url:
        result["sources"]["rss"] = await ingest_rss(db, agency_id, rss_url)
    if twitter_handle:
        result["sources"]["x"] = await ingest_x_twitter(db, agency_id, twitter_handle)
    if facebook_page:
        result["sources"]["facebook"] = await ingest_facebook(db, agency_id, facebook_page)
    return result


async def enrich_competitor_via_hub(db: AsyncSession, competitor: Competitor) -> list[CompetitorSnapshot]:
    """Full rival scrape through the ingestion hub (live + planned sources)."""
    from datetime import datetime

    agency_id = competitor.agency_id
    snapshots: list[CompetitorSnapshot] = []

    hub = await run_ingestion_hub(
        db,
        agency_id,
        website=competitor.website,
        serp_query=competitor.name,
        reddit_query=f"{competitor.name} review OR alternative",
        twitter_handle=competitor.twitter_handle,
        facebook_page=competitor.facebook_page,
    )

    mapping = {
        "firecrawl": "website",
        "serpapi": "seo",
        "reddit": "reddit",
        "rss": "rss",
        "x": "twitter",
        "facebook": "facebook",
    }
    for key, source in mapping.items():
        payload = (hub.get("sources") or {}).get(key)
        if not payload:
            continue
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source=source,
                payload=payload,
                summary=f"{source} scan for {competitor.name}",
            )
        )

    if competitor.instagram_handle:
        ig = await run_apify_actor(
            db,
            agency_id,
            "apify~instagram-scraper",
            {
                "directUrls": [f"https://www.instagram.com/{competitor.instagram_handle.strip('@')}/"],
                "resultsLimit": 10,
            },
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
