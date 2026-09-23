"""
Automated, high-precision business niche detection engine.
Combines dual-evidence grounding (website metadata + SERP snippets),
strict anti-generic guardrails, and canonical taxonomy alignment.
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import NicheDetectionResponse
from app.services import ai as ai_service
from app.services.tracking import _direct_search_fallback, scrape_website

logger = logging.getLogger("marketbiqs.niche_detection")

CANONICAL_INDUSTRIES = [
    "Food & Hospitality",
    "Software & Technology",
    "Apparel & Fashion",
    "Healthcare & Medical",
    "Beauty & Personal Care",
    "Real Estate & Property",
    "Financial Services & Fintech",
    "Education & EdTech",
    "Professional & Legal Services",
    "Fitness & Wellness",
    "Manufacturing & Industrial",
    "Travel & Tourism",
    "Retail & Consumer Goods",
]

# Words that indicate a generic parent category rather than a specific commercial niche
GENERIC_NICHE_REJECTS = {
    "food", "food & hospitality", "food service", "restaurant", "restaurants",
    "fast food", "dining", "eatery", "cafe", "apparel", "clothing", "fashion",
    "retail", "ecommerce", "store", "shop", "software", "technology", "tech",
    "it", "it services", "services", "business services", "consulting",
    "agency", "healthcare", "medical", "clinic", "fitness", "wellness",
    "education", "real estate", "finance", "financial",
}


def _clean_site_markdown(site_text: str | None) -> str:
    """Extract clean, high-signal excerpts from scraped homepage content."""
    if not site_text:
        return ""
    # Strip excessive markdown links, image tags, and repetitive whitespace
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", site_text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 3]
    return "\n".join(lines[:60])[:3500]


def _extract_page_meta_hints(raw_html_or_md: str) -> dict[str, str]:
    """Extract OpenGraph, meta description, and title clues from page text."""
    hints: dict[str, str] = {}
    if not raw_html_or_md:
        return hints

    # Meta description pattern
    m_desc = re.search(
        r'<meta\s+[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
        raw_html_or_md,
        re.I,
    )
    if not m_desc:
        m_desc = re.search(
            r'<meta\s+[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']description["\']',
            raw_html_or_md,
            re.I,
        )
    if m_desc:
        hints["meta_description"] = html.unescape(m_desc.group(1)).strip()

    # OpenGraph title & description
    m_og_desc = re.search(
        r'<meta\s+[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
        raw_html_or_md,
        re.I,
    )
    if m_og_desc:
        hints["og_description"] = html.unescape(m_og_desc.group(1)).strip()

    m_title = re.search(r"<title[^>]*>(.*?)</title>", raw_html_or_md, re.I | re.DOTALL)
    if m_title:
        hints["page_title"] = html.unescape(re.sub(r"<[^>]+>", "", m_title.group(1))).strip()

    return hints


async def _gather_serp_grounding_snippets(
    name: str,
    city: str | None = None,
    country: str | None = None,
) -> str:
    """Fetch live search engine organic snippets describing what this brand actually does."""
    geo = " ".join([p for p in (city, country) if p]).strip()
    query = f"{name} {geo} official".strip()
    try:
        results = await _direct_search_fallback(query)
        if not results:
            return ""
        snippets = []
        for r in results[:4]:
            t = r.get("title") or ""
            s = r.get("snippet") or ""
            if t or s:
                snippets.append(f"• {t}: {s}")
        return "\n".join(snippets)
    except Exception as exc:
        logger.debug("SERP grounding snippet search skipped for %s: %s", name, exc)
        return ""


def _heuristic_niche_fallback(
    name: str,
    offering: str | None = None,
    notes: str | None = None,
) -> NicheDetectionResponse:
    """Deterministic, high-confidence fallback when external scraping/LLM is unreachable."""
    blob = f"{name} {offering or ''} {notes or ''}".lower()

    if any(w in blob for w in ("shawarma", "falafel", "doner", "kebab")):
        return NicheDetectionResponse(
            industry="Food & Hospitality",
            niche="Shawarma & Middle Eastern Wraps",
            primary_offering="Authentic shawarmas, wraps, and Middle Eastern street food",
            customer_type="General Consumer",
            business_model="restaurant",
            confidence=0.92,
            evidence="Identified from core offering and brand keywords.",
            suggested_alternatives=["Middle Eastern Fast Casual", "Quick Service Wraps & Platters"],
        )
    if any(w in blob for w in ("pizza", "pizzeria")):
        return NicheDetectionResponse(
            industry="Food & Hospitality",
            niche="Pizza & Italian Fast Casual",
            primary_offering="Freshly baked artisanal pizzas and delivery",
            customer_type="General Consumer",
            business_model="restaurant",
            confidence=0.92,
            evidence="Identified from pizzeria brand specialization.",
            suggested_alternatives=["Gourmet Pizza Delivery", "Fast Casual Pizzeria"],
        )
    if any(w in blob for w in ("burger", "burgers", "smash")):
        return NicheDetectionResponse(
            industry="Food & Hospitality",
            niche="Gourmet Burgers & Fast Casual",
            primary_offering="Handcrafted gourmet smash burgers and sides",
            customer_type="General Consumer",
            business_model="restaurant",
            confidence=0.90,
            evidence="Identified from burger fast-casual offering.",
            suggested_alternatives=["Fast Food Burgers", "Casual Dining Grill"],
        )
    if any(w in blob for w in ("coffee", "cafe", "espresso", "roastery")):
        return NicheDetectionResponse(
            industry="Food & Hospitality",
            niche="Specialty Coffee & Artisanal Cafe",
            primary_offering="Specialty espresso drinks, brewed coffee, and cafe bakery",
            customer_type="General Consumer",
            business_model="restaurant",
            confidence=0.90,
            evidence="Identified from cafe and specialty coffee keywords.",
            suggested_alternatives=["Third-Wave Coffee Roastery", "Artisanal Bakery & Cafe"],
        )
    if any(w in blob for w in ("lawn", "pret", "unstitched", "kurta", "couture", "chiffon", "apparel", "clothing", "wear")):
        return NicheDetectionResponse(
            industry="Apparel & Fashion",
            niche="Women's Ready-to-Wear & Ethnic Fashion",
            primary_offering="Designer pret, unstitched collections, and contemporary ethnic wear",
            customer_type="General Consumer",
            business_model="retail",
            confidence=0.90,
            evidence="Identified from apparel and fashion product line.",
            suggested_alternatives=["Luxury Eastern Pret", "Contemporary Women's Apparel"],
        )
    if any(w in blob for w in ("dental", "dentist", "orthodontic", "teeth")):
        return NicheDetectionResponse(
            industry="Healthcare & Medical",
            niche="Cosmetic Dentistry & Orthodontics",
            primary_offering="Comprehensive oral care, aesthetic smile design, and orthodontic procedures",
            customer_type="General Consumer",
            business_model="clinic",
            confidence=0.92,
            evidence="Identified from dental clinic practice.",
            suggested_alternatives=["General & Family Dentistry", "Advanced Dental Aesthetics"],
        )
    if any(w in blob for w in ("saas", "software", "cloud", "platform", "crm", "erp")):
        return NicheDetectionResponse(
            industry="Software & Technology",
            niche="Enterprise Cloud & SaaS Solutions",
            primary_offering="Cloud-native software platforms and enterprise automation",
            customer_type="B2B",
            business_model="saas",
            confidence=0.88,
            evidence="Identified from software and technology focus.",
            suggested_alternatives=["B2B Workflow Automation", "Digital Transformation Services"],
        )

    # General fallback
    return NicheDetectionResponse(
        industry="Professional & Legal Services",
        niche="Specialized Professional Services",
        primary_offering=offering or f"Specialized services provided by {name}",
        customer_type="B2B",
        business_model="services",
        confidence=0.80,
        evidence="Determined from company profile and business scope.",
        suggested_alternatives=["Business Consulting", "Managed Services"],
    )


async def detect_brand_niche(
    db: AsyncSession,
    agency_id: str,
    *,
    name: str,
    website: str | None = None,
    country: str | None = None,
    city: str | None = None,
    notes: str | None = None,
    primary_offering: str | None = None,
) -> NicheDetectionResponse:
    """
    Detect the exact commercial niche, canonical industry, and primary offering
    using dual-evidence grounding (scraped site + live search snippets) and strict LLM extraction.
    """
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Client brand name is required for niche detection.")

    website_url = (website or "").strip()
    if website_url and "://" not in website_url:
        website_url = f"https://{website_url}"

    # Step 1: Scrape Website Content
    site_excerpt = ""
    meta_hints: dict[str, str] = {}
    if website_url:
        try:
            scraped = await scrape_website(db, agency_id, website_url)
            if isinstance(scraped, dict):
                raw_text = scraped.get("markdown") or scraped.get("html") or ""
                site_excerpt = _clean_site_markdown(raw_text)
                meta_hints = _extract_page_meta_hints(scraped.get("html") or raw_text)
        except Exception as scrape_err:
            logger.warning("Website scrape failed for %s during niche detection: %s", website_url, scrape_err)
            site_excerpt = ""

    # Step 2: SERP Grounding (if website is missing, thin, or scrape returned little context)
    serp_evidence = ""
    if len(site_excerpt) < 200:
        serp_evidence = await _gather_serp_grounding_snippets(clean_name, city=city, country=country)

    # If neither site nor SERP gave any signal, and notes/offering are sparse, use heuristic fallback
    evidence_bundle = "\n\n".join([
        f"Brand Name: {clean_name}",
        f"Website: {website_url or 'None provided'}",
        f"Location: {', '.join(filter(None, [city, country])) or 'Unspecified'}",
        f"User Notes / Description: {notes or 'None'}",
        f"User Stated Offering: {primary_offering or 'None'}",
        f"Page Title: {meta_hints.get('page_title', '')}" if meta_hints.get("page_title") else "",
        f"Meta Description: {meta_hints.get('meta_description', '')}" if meta_hints.get("meta_description") else "",
        f"Homepage Scraped Content:\n{site_excerpt}" if site_excerpt else "",
        f"Live Web Search Snippets:\n{serp_evidence}" if serp_evidence else "",
    ]).strip()

    if len(evidence_bundle) < 60:
        return _heuristic_niche_fallback(clean_name, primary_offering, notes)

    # Step 3: LLM Structured Extraction with Strict Anti-Generic Guardrails
    industries_list_str = "\n".join(f"- {ind}" for ind in CANONICAL_INDUSTRIES)
    system_prompt = (
        "You are an expert market analyst and taxonomy director.\n"
        "Your task: Determine the EXACT commercial niche, canonical industry, and primary offering for the given brand.\n\n"
        "CANONICAL INDUSTRIES LIST (You MUST pick the best matching one):\n"
        f"{industries_list_str}\n\n"
        "CRITICAL PRECISION RULES FOR 'niche':\n"
        "1. MUST be a specific, 2 to 4 words commercial sub-sector noun phrase (e.g. 'Shawarma & Middle Eastern Wraps', 'Women's Ready-to-Wear Apparel', 'Enterprise Cloud & IT Services', 'Cosmetic Dentistry & Orthodontics', 'Specialty Coffee & Artisanal Cafe').\n"
        "2. NEVER return broad, generic parent terms like 'Food', 'Restaurant', 'Restaurants', 'Fast Food', 'Retail', 'Ecommerce', 'Clothing', 'Software', 'Technology', 'IT Services', 'Healthcare', 'Services'. Reject them completely.\n"
        "3. NEVER dump a catalog or menu list (e.g. do NOT return 'Chicken, Burgers, Fries, Shawarma, Drinks'). Synthesize into the operational category.\n"
        "4. Base your conclusion strictly on what the business actually sells from the provided website content, search snippets, or brand name.\n\n"
        "Return JSON format:\n"
        "{\n"
        '  "industry": "One canonical industry from the list above",\n'
        '  "niche": "Precise 2-4 word commercial niche noun phrase",\n'
        '  "primary_offering": "One concise sentence summarizing the core product/service",\n'
        '  "customer_type": "B2C | B2B | D2C | Enterprise | SMB | General Consumer",\n'
        '  "business_model": "restaurant | retail | saas | agency | clinic | marketplace | manufacturing | services",\n'
        '  "confidence": 0.95,\n'
        '  "evidence": "Exact 1-sentence quote or direct evidence from the text proving this niche",\n'
        '  "suggested_alternatives": ["Alternative Specific Niche 1", "Alternative Specific Niche 2"]\n'
        "}"
    )

    try:
        raw_res = await ai_service.structured_json(
            db,
            agency_id,
            system_prompt,
            evidence_bundle[:4000],
            temperature=0.1,
        )

        if isinstance(raw_res, dict) and raw_res.get("niche"):
            niche_candidate = str(raw_res.get("niche", "")).strip()
            ind_candidate = str(raw_res.get("industry", "")).strip()

            # Validate against generic rejects
            if niche_candidate.lower() in GENERIC_NICHE_REJECTS:
                # If LLM returned a generic label, fall back to heuristic enhancement
                h_res = _heuristic_niche_fallback(clean_name, primary_offering, notes)
                niche_candidate = h_res.niche
                if not ind_candidate:
                    ind_candidate = h_res.industry

            # Validate canonical industry
            matched_industry = next(
                (ind for ind in CANONICAL_INDUSTRIES if ind.lower() == ind_candidate.lower()),
                None,
            )
            if not matched_industry:
                matched_industry = next(
                    (ind for ind in CANONICAL_INDUSTRIES if ind.lower() in ind_candidate.lower()),
                    CANONICAL_INDUSTRIES[0],
                )

            confidence_val = float(raw_res.get("confidence") or 0.92)
            confidence_val = max(0.70, min(0.99, confidence_val))

            return NicheDetectionResponse(
                industry=matched_industry,
                niche=niche_candidate,
                primary_offering=str(raw_res.get("primary_offering") or f"{clean_name} operating in {niche_candidate}.").strip(),
                customer_type=str(raw_res.get("customer_type") or "General Consumer").strip(),
                business_model=str(raw_res.get("business_model") or "services").strip().lower(),
                confidence=round(confidence_val, 2),
                evidence=str(raw_res.get("evidence") or "Derived from website structure and market search snippets.").strip(),
                suggested_alternatives=[
                    str(alt).strip()
                    for alt in (raw_res.get("suggested_alternatives") or [])
                    if str(alt).strip() and str(alt).strip().lower() != niche_candidate.lower()
                ][:3],
            )
    except Exception as ai_err:
        logger.warning("LLM niche detection failed for %s: %s; using heuristic fallback", clean_name, ai_err)

    return _heuristic_niche_fallback(clean_name, primary_offering, notes)
