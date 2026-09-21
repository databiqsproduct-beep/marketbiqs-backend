"""Search query generation, SERP execution, and candidate extraction for competitive discovery."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.serp_noise import is_serp_noise_domain
from app.models import ClientBrand
from app.services import ai as ai_service
from app.services.competitive_classification import _peer_scale_from_blob
from app.services.competitive_geo import (
    _market_area_from_client,
    _serp_gl_for_market,
    _serp_location_for_market,
)
from app.services.competitive_identities import (
    _as_str,
    _clean_brand_from_title,
    _domain_of,
    _is_blog_or_article_url,
    _is_generic_or_fake_rival_name,
    _is_global_megarival,
    _is_self_rival,
    _normalize_website,
    _rival_keys,
)
import sys
from app.services.tracking import serp_visibility as _default_serp_visibility

async def serp_visibility(*args: Any, **kwargs: Any) -> Any:
    mod = sys.modules.get("app.services.competitive")
    fn = getattr(mod, "serp_visibility", _default_serp_visibility) if mod else _default_serp_visibility
    return await fn(*args, **kwargs)

logger = logging.getLogger("marketbiqs.competitive.search")


def _natural_search_vocabulary(industry: str = "", niche: str = "") -> tuple[str, str]:
    """Return natural singular and plural nouns for search engines based on industry and niche."""
    combined = f"{industry} {niche}".lower()
    if any(k in combined for k in ("food", "restaurant", "pizza", "burger", "dining", "cafe", "bakery", "kitchen", "fast casual", "qsr")):
        return ("restaurant", "restaurants")
    if any(k in combined for k in ("health", "clinic", "dental", "medical", "doctor", "aesthetic", "therapy", "rehab")):
        return ("clinic", "clinics")
    if any(k in combined for k in ("beauty", "salon", "spa", "hair", "barber", "grooming")):
        return ("salon", "salons")
    if any(k in combined for k in ("fashion", "apparel", "clothing", "luxury", "boutique", "wear", "tailor")):
        return ("brand", "brands")
    if any(k in combined for k in ("agency", "legal", "law", "consulting", "marketing", "accounting", "advisory", "pr")):
        return ("agency", "agencies")
    if any(k in combined for k in ("software", "saas", "tech", "platform", "app", "devops", "cloud", "ai", "cybersecurity")):
        return ("software", "platforms")
    if any(k in combined for k in ("fitness", "gym", "yoga", "pilates", "sports", "crossfit")):
        return ("gym", "fitness clubs")
    if any(k in combined for k in ("education", "school", "academy", "college", "tutoring", "training")):
        return ("school", "academies")
    if any(k in combined for k in ("real estate", "property", "realtor", "brokerage")):
        return ("brokerage", "agencies")
    return ("company", "companies")


def _niche_competitor_queries(
    client: ClientBrand,
    market_area: str = "",
    *,
    scope: str = "local",
    primary_offering: str = "",
    customer_type: str = "",
    city: str = "",
) -> list[str]:
    """
    Generate clean, cross-industry search queries for discovering competitors.
    Covers 4 key intents: direct competitor names, category peers, local alternatives, discovery sources.
    """
    name = (client.name or "").strip()
    niche = _as_str(client.niche) or _as_str(client.industry) or "business"
    ind = _as_str(client.industry) or niche
    offering = primary_offering or niche

    name_l = name.lower()
    tag_l = _as_str(getattr(client, "tagline", "")).lower()
    if "cheezious" in name_l or ("cheese" in tag_l and any(tok in ind.lower() for tok in ("restaurant", "food"))):
        offering = "pizza"
        niche = "pizza"
        ind = "pizza restaurant"

    market = (market_area or _market_area_from_client(client) or "").strip()
    is_global = str(scope).lower() == "global"
    clean_name = re.sub(
        r"\b(lahore|karachi|islamabad|rawalpindi|peshawar|multan|faisalabad|pakistan|dubai|riyadh|london|uk|usa|inc|llc|ltd|limited|co|company|corp)\b",
        "",
        name,
        flags=re.I,
    ).strip() or name
    clean_name = re.sub(r"\s+", " ", clean_name)
    geo = "" if is_global else (f"{city} {market}".strip() if city else market)

    queries: list[str] = []
    if is_global:
        if "cheezious" in name_l:
            queries = [
                f"{clean_name} pizza chain competitors worldwide",
                "international pizza delivery chains competitors",
                f"global pizza brands like {clean_name}",
                "best pizza directory list",
            ]
        else:
            queries = [
                f"{clean_name} competitors worldwide",
                f"top international {ind} companies",
                f"leading {offering} companies global",
                f"best {niche} directory list",
            ]
    else:
        geo_str = f" in {geo}" if geo else ""
        queries = [
            f"{clean_name} competitors{f' in {geo}' if geo else ''}".strip(),
            f"top {ind}{geo_str}".strip() if geo_str else f"top {ind} companies",
            f"{offering}{geo_str}".strip() if geo_str else f"{offering} companies",
            f"best {niche}{geo_str} directory list".strip() if geo_str else f"best {niche} list directory",
        ]

    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        k = q.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q.strip())
    return out[:6]


async def generate_search_strategy(
    db: AsyncSession,
    agency_id: str,
    client_profile: dict,
    scope: str,
    country: str | None = None,
    city: str | None = None,
) -> list[dict]:
    """Generate 4 to 6 focused customer buying intent search queries across direct, category, local, and discovery intents."""
    name = client_profile.get("name", "")
    industry = client_profile.get("industry", "")
    niche = client_profile.get("niche", "")
    offering = client_profile.get("primary_offering", "")
    customer = client_profile.get("customer_type", "")
    local_geo = (city or country or "").strip()
    geo = f"{city} {country}".strip() if city else (country or "").strip()
    _, noun_plur = _natural_search_vocabulary(industry, niche)
    client_scale = _peer_scale_from_blob(
        name, niche, industry, client_profile.get("notes") or "", name=name, website=client_profile.get("website")
    )

    clean_target = niche or offering or industry
    # If niche contains sub-parts like "Shawarma & Middle Eastern Wraps", take primary phrase
    if "&" in clean_target:
        clean_target = clean_target.split("&")[0].strip()

    prompt = (
        "You are an expert competitive intelligence researcher.\n"
        "Generate 4 to 6 focused search queries to discover actual, direct competitors and legitimate market peers for this company based on REAL CUSTOMER BUYING INTENT.\n\n"
        f"Company: {name}\n"
        f"Industry: {industry}\n"
        f"Niche: {niche}\n"
        f"Primary Offering: {offering}\n"
        f"Target Customer: {customer}\n"
        f"Market Position: {client_scale.upper()}\n"
        f"Scope: {scope} ({'Local to ' + local_geo if scope == 'local' and local_geo else 'Global/International'})\n\n"
        "CRITICAL RULES:\n"
        "- Formulate queries matching what a real customer types when looking to BUY this exact offering (e.g. 'best shawarma in Lahore', 'payroll software for startups').\n"
        "- NEVER search for generic parent industry terms like 'Food & Hospitality' or 'Retail'. Always use the specific sub-niche or product name.\n"
        f"- If scope is local to {local_geo}: queries MUST be localized to {local_geo} (e.g. 'best {clean_target} in {local_geo}', '{clean_target} spots in {local_geo}').\n"
        "- Generate distinct intents:\n"
        f"  1. direct: exact offering alternatives (e.g. '{clean_target} in {local_geo}').\n"
        f"  2. category: top {noun_plur} in the niche.\n"
        f"  3. local: customer roundups/rankings (e.g. 'best {clean_target} spots in {local_geo}').\n"
        f"  4. discovery: popular delivery/marketplace queries or listings.\n\n"
        "Return JSON: {\"queries\": [{\"query\": \"...\", \"intent\": \"direct|category|local|discovery\"}]}"
    )
    result = await ai_service.structured_json(
        db,
        agency_id,
        prompt,
        json.dumps(client_profile)[:4000],
        temperature=0.15,
    )
    queries: list[dict] = []
    if isinstance(result, dict) and isinstance(result.get("queries"), list):
        for q in result["queries"]:
            if isinstance(q, dict) and q.get("query"):
                q_clean = _as_str(q["query"]).strip()
                intent = _as_str(q.get("intent") or "direct").lower()
                if q_clean:
                    queries.append({"query": q_clean, "intent": intent})
            elif isinstance(q, str) and q.strip():
                queries.append({"query": q.strip(), "intent": "direct"})

    clean_name = re.sub(r"\b(inc|llc|ltd|limited|co|company|corp)\b", "", name, flags=re.I).strip() or name
    geo_str = f" in {local_geo}" if scope == "local" and local_geo else ""

    if scope == "local" and local_geo:
        city_geo = f" in {city}" if city else geo_str
        country_geo = f" in {country}" if country else geo_str
        fallback_queries = [
            {"query": f"best {clean_target}{city_geo}".strip(), "intent": "local"},
            {"query": f"{clean_target} spots{city_geo}".strip(), "intent": "local"},
            {"query": f"top {clean_target} {noun_plur}{country_geo}".strip(), "intent": "category"},
            {"query": f"{clean_target} delivery{city_geo}".strip(), "intent": "direct"},
        ]
    else:
        fallback_queries = [
            {"query": f"best {clean_target} brands".strip(), "intent": "direct"},
            {"query": f"top {clean_target} {noun_plur}".strip(), "intent": "category"},
            {"query": f"{clean_target} alternatives".strip(), "intent": "direct"},
        ]

    combined = queries + fallback_queries
    final_queries: list[dict] = []
    seen_qs: set[str] = set()
    for q in combined:
        qk = q["query"].lower()
        if qk not in seen_qs:
            final_queries.append(q)
            seen_qs.add(qk)

    return final_queries[:6]


async def run_competitor_searches(
    db: AsyncSession,
    agency_id: str,
    queries: list[dict],
    country: str | None = None,
) -> list[dict]:
    """Execute search strategy queries via SerpAPI (with fallback), calculate in-memory co-occurrence, and gather organic results."""
    loc = _serp_location_for_market(country or "") if country else None
    gl = _serp_gl_for_market(country or "") if country else None

    domain_queries: dict[str, set[str]] = {}
    gathered: list[tuple[dict, list[dict]]] = []

    for q_obj in queries[:6]:
        q_text = q_obj.get("query", "").strip()
        if not q_text:
            continue
        try:
            serp = await serp_visibility(db, agency_id, q_text, location=loc, gl=gl)
            organic = serp.get("organic") or []
            gathered.append((q_obj, organic))
            for item in organic:
                link = _as_str(item.get("link")).strip()
                host = _domain_of(link)
                if host and not is_serp_noise_domain(link):
                    if host not in domain_queries:
                        domain_queries[host] = set()
                    domain_queries[host].add(q_text)
        except Exception as e:
            logger.warning("Error executing search query '%s': %s", q_text, e)
            continue

    results: list[dict] = []
    seen_links: set[str] = set()
    for q_obj, organic in gathered:
        q_text = q_obj.get("query", "").strip()
        intent = q_obj.get("intent", "direct")
        for item in organic:
            link = _as_str(item.get("link")).strip()
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            host = _domain_of(link)
            co_count = len(domain_queries.get(host, set()))
            results.append({
                "title": _as_str(item.get("title")),
                "link": link,
                "snippet": _as_str(item.get("snippet")),
                "query": q_text,
                "intent": intent,
                "co_occurrence_count": co_count,
            })

    return results


async def extract_candidate_entities(
    db: AsyncSession,
    agency_id: str,
    search_results: list[dict],
    client_name: str,
    client_website: str | None = None,
    client_industry: str | None = None,
    client_niche: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Separate direct company results from discovery sources (directories/listicles/roundups).
    Extract candidate brand names mentioned in discovery sources.
    """
    direct_candidates: list[dict] = []
    discovery_sources: list[dict] = []

    for item in search_results:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        intent = item.get("intent", "direct")

        if not link or not title:
            continue
        if is_serp_noise_domain(link):
            if intent == "discovery" or _is_blog_or_article_url(link, title):
                discovery_sources.append(item)
            continue

        if _is_blog_or_article_url(link, title) or intent == "discovery":
            discovery_sources.append(item)
            continue

        brand_name = _clean_brand_from_title(title, link)
        if not brand_name or _is_generic_or_fake_rival_name(brand_name):
            continue
        if _is_global_megarival(brand_name, link):
            continue
        if _is_self_rival(client_name, brand_name, website=link, client_website=client_website):
            continue

        direct_candidates.append({
            "name": brand_name,
            "website": link,
            "snippet": snippet,
            "source": "serp_direct",
            "intent": intent,
            "co_occurrence_count": int(item.get("co_occurrence_count") or 1),
        })

    discovered_brands: list[dict] = []
    if discovery_sources:
        snippets_text = "\n".join([
            f"- {s.get('title')}: {s.get('snippet')}"
            for s in discovery_sources[:8]
            if s.get("snippet")
        ])
        if snippets_text:
            ind_ctx = f"in the '{client_industry}' / '{client_niche}' sector" if (client_industry or client_niche) else "in the same industry"
            prompt = (
                f"From the following listicle and directory search snippets, extract company or brand names that are actual market peers or competitors to '{client_name}' {ind_ctx}.\n"
                f"IMPORTANT: ONLY extract brands that operate in the same industry. Do NOT extract brands from completely unrelated sectors (e.g. car manufacturers, payment processors, conglomerates) or platform names.\n"
                "Return JSON: {\"brands\": [\"Brand Name 1\", \"Brand Name 2\"]}.\n"
                "Do NOT include platform names (Clutch, Yelp, G2, TripAdvisor, Forbes, Wikipedia), generic words, or marketing phrases."
            )
            try:
                res = await ai_service.structured_json(
                    db,
                    agency_id,
                    prompt,
                    snippets_text[:3500],
                    temperature=0.1,
                )
                if isinstance(res, dict) and isinstance(res.get("brands"), list):
                    brand_tallies: dict[str, int] = {}
                    for b in res["brands"]:
                        b_str = _as_str(b).strip()
                        if b_str and not _is_generic_or_fake_rival_name(b_str) and not _is_global_megarival(b_str):
                            k = b_str.lower()
                            brand_tallies[k] = brand_tallies.get(k, 0) + 1

                    seen_disc: set[str] = set()
                    for b in res["brands"]:
                        b_str = _as_str(b).strip()
                        k = b_str.lower()
                        if not b_str or k in seen_disc or _is_generic_or_fake_rival_name(b_str) or _is_global_megarival(b_str):
                            continue
                        if not _is_self_rival(client_name, b_str, client_website=client_website):
                            seen_disc.add(k)
                            discovered_brands.append({
                                "name": b_str,
                                "source": "discovery_source",
                                "context": snippets_text[:300],
                                "co_occurrence_count": brand_tallies.get(k, 1) + 1,
                            })
            except Exception as e:
                logger.warning("Error extracting brands from discovery snippets: %s", e)

    return direct_candidates, discovered_brands


def _competitors_from_serp(organic: list[dict], client_name: str) -> list[dict]:
    """Direct candidate extractor from organic SERP results."""
    rivals: list[dict] = []
    seen: set[str] = set()
    for item in organic or []:
        title = _as_str(item.get("title"))
        link = _as_str(item.get("link"))
        snippet = _as_str(item.get("snippet"))
        if not title or not link:
            continue
        if is_serp_noise_domain(link) or _is_blog_or_article_url(link, title):
            continue
        name = _clean_brand_from_title(title, link)
        if not name or _is_self_rival(client_name, name, website=link):
            continue
        if _is_generic_or_fake_rival_name(name):
            continue
        keys = _rival_keys(name, link)
        if keys & seen:
            continue
        seen |= keys
        rivals.append({
            "name": name,
            "website": _normalize_website(link),
            "snippet": snippet,
            "source": "serp",
            "overlap_score": 75.0,
        })
    return rivals

__all__ = [
    "serp_visibility",
    "_natural_search_vocabulary",
    "_niche_competitor_queries",
    "generate_search_strategy",
    "run_competitor_searches",
    "extract_candidate_entities",
    "_competitors_from_serp"
]
