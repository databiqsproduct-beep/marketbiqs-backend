import asyncio
import json
import logging
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Agency,
    ClientBrand,
    Competitor,
    FeatureComparison,
    FeatureTicket,
    GapReport,
    GoalAlert,
    Integration,
    JobStatus,
    ProductFeature,
    TrackingJob,
)
from app.services import ai as ai_service
from app.services import jira as jira_service
from app.services.reports import generate_client_report
from app.services.tracking import scrape_website, serp_visibility

logger = logging.getLogger("marketbiqs.competitive")


def _clip(value: str | None, max_len: int) -> str:
    text = (value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _level_label(value: str | None, default: str = "medium", *, max_len: int = 40) -> str:
    """Normalize AI severity labels; allow short free text up to max_len."""
    raw = _as_str(value, default).strip()
    lowered = raw.lower()
    if lowered in {"low", "medium", "high", "critical"}:
        return lowered
    if not raw:
        return default
    return _clip(raw, max_len)


_WEAK_COMPARISON_MARKERS = (
    "none",
    "n/a",
    "does not have a similar feature",
    "do not have a similar feature",
    "both companies have similar features",
    "continue to enhance and promote",
    "continue to monitor and improve",
    "no similar feature",
    "similar features",
)


def _as_str(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)


def _is_generic_text(value) -> bool:
    text = _as_str(value).strip().lower()
    if len(text) < 12:
        return True
    return any(marker in text for marker in _WEAK_COMPARISON_MARKERS)


def _normalize_comparison_row(row: dict, client_name: str, competitor_name: str) -> dict | None:
    feature_name = _as_str(row.get("feature_name")).strip()
    if not feature_name:
        return None

    our_status = _as_str(row.get("our_status"), "parity").strip().lower()
    competitor_status = _as_str(row.get("competitor_status"), "parity").strip().lower()
    status_map = {
        "lead": "leading",
        "leading": "leading",
        "strong": "leading",
        "has": "leading",
        "available": "leading",
        "parity": "parity",
        "equal": "parity",
        "similar": "parity",
        "lagging": "lagging",
        "weak": "lagging",
        "missing": "lagging",
        "none": "lagging",
        "absent": "lagging",
        "behind": "lagging",
    }
    our_status = status_map.get(our_status, "parity")
    competitor_status = status_map.get(competitor_status, "parity")

    note = _as_str(row.get("note")).strip()
    how_leads = _as_str(row.get("how_competitor_leads")).strip()
    how_improve = _as_str(row.get("how_to_improve")).strip()

    if _is_generic_text(note):
        note = f"{competitor_name} vs {client_name} on {feature_name}: rival posture is {competitor_status}, yours is {our_status}."
    if _is_generic_text(how_leads):
        how_leads = f"{competitor_name} is positioned as {competitor_status} on {feature_name} in public materials and product packaging."
    if _is_generic_text(how_improve):
        how_improve = f"Ship a clearer {feature_name} offer, proof points, and sales narrative to close the gap with {competitor_name}."

    citations = row.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    clean_citations = []
    for c in citations[:4]:
        if isinstance(c, dict) and (c.get("url") or c.get("snippet")):
            clean_citations.append(
                {
                    "url": _as_str(c.get("url")),
                    "snippet": _as_str(c.get("snippet"))[:400],
                    "source": _as_str(c.get("source"), "web"),
                }
            )

    try:
        confidence = float(row.get("confidence_score") or 0.55)
    except (TypeError, ValueError):
        confidence = 0.55
    confidence = max(0.0, min(1.0, confidence))
    if clean_citations:
        confidence = max(confidence, 0.65)
    evidence = _as_str(row.get("evidence_strength"), "medium").strip().lower()
    if evidence not in {"low", "medium", "high"}:
        evidence = "medium" if clean_citations else "low"

    contested = competitor_status == "leading" or our_status == "lagging"

    return {
        "feature_name": feature_name,
        "category": _as_str(row.get("category"), "General").strip() or "General",
        "our_status": our_status,
        "competitor_status": competitor_status,
        "note": note,
        "how_competitor_leads": how_leads,
        "how_to_improve": how_improve,
        "citations": clean_citations,
        "confidence_score": confidence,
        "evidence_strength": evidence,
        "is_contested_move": contested,
    }


async def _generate_competitor_comparisons(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    features: list[ProductFeature],
    competitor: Competitor,
) -> dict:
    result = await ai_service.structured_json(
        db,
        agency.id,
        (
            "You are a senior competitive strategist writing UpdatePromise/Databiqs-quality comparison rows. "
            "Return JSON: {competitor_name, rows:[{feature_name, category, our_status, competitor_status, note, how_competitor_leads, how_to_improve, confidence_score, evidence_strength, citations:[{url, snippet, source}]}]}. "
            "Rules:\n"
            "1) Only include contested features where the rival is leading, we are lagging, or parity is commercially dangerous.\n"
            "2) DO NOT include rows where we clearly lead and the rival lags.\n"
            "3) our_status/competitor_status must be leading|parity|lagging.\n"
            "4) note, how_competitor_leads, and how_to_improve must each be specific (1-3 sentences), concrete, and actionable.\n"
            "5) NEVER write: None, N/A, 'does not have a similar feature', 'both companies have similar features', "
            "'continue to enhance and promote', or 'continue to monitor and improve'.\n"
            "6) how_competitor_leads must explain buyer perception, GTM, packaging, workflow fit, or brand equity.\n"
            "7) how_to_improve must give a concrete counter-move (product packaging, proof, pricing page, demo narrative, content).\n"
            "8) Include citations with url + short snippet whenever possible from competitor website/features.\n"
            "9) confidence_score 0-1 and evidence_strength low|medium|high.\n"
            "10) Produce 3-6 high-signal rows only."
        ),
        json.dumps(
            {
                "client": {
                    "name": client.name,
                    "industry": client.industry,
                    "niche": client.niche,
                    "tagline": client.tagline,
                    "goals": client.goals or [],
                    "features": [
                        {"name": f.name, "category": f.category, "description": f.description} for f in features
                    ],
                },
                "competitor": {
                    "name": competitor.name,
                    "website": competitor.website,
                    "tagline": competitor.tagline,
                    "description": competitor.description,
                    "overlap_score": competitor.overlap_score,
                    "threat_level": competitor.threat_level,
                    "features": competitor.feature_list or [],
                },
            }
        )[:12000],
        temperature=0.4,
    )

    rows = result.get("rows") or []
    cleaned_rows = []
    for row in rows:
        cleaned = _normalize_comparison_row(row, client.name, competitor.name)
        if cleaned:
            cleaned_rows.append(cleaned)

    if len(cleaned_rows) < 2:
        repair = await ai_service.structured_json(
            db,
            agency.id,
            (
                "Rewrite competitive comparison rows. Prioritize where the rival threatens the client. "
                "Return JSON {rows:[...] } with the same schema and strict anti-filler rules. "
                "Every how_competitor_leads and how_to_improve must be specific strategy language."
            ),
            json.dumps(
                {
                    "competitor_name": competitor.name,
                    "client_name": client.name,
                    "client_features": [f.name for f in features],
                    "competitor_features": [
                        (f.get("name") if isinstance(f, dict) else str(f)) for f in (competitor.feature_list or [])
                    ],
                    "weak_draft_rows": rows,
                }
            )[:10000],
            temperature=0.45,
        )
        for row in repair.get("rows") or []:
            cleaned = _normalize_comparison_row(row, client.name, competitor.name)
            if cleaned:
                cleaned_rows.append(cleaned)

    if not cleaned_rows:
        rival_feats = []
        for f in competitor.feature_list or []:
            if isinstance(f, dict) and f.get("name"):
                rival_feats.append(_as_str(f.get("name")))
            elif isinstance(f, str) and f.strip():
                rival_feats.append(f.strip())
        client_names = {f.name.lower() for f in features}
        for name in rival_feats[:5]:
            cleaned_rows.append(
                _normalize_comparison_row(
                    {
                        "feature_name": name,
                        "category": "Competitive",
                        "our_status": "lagging" if name.lower() not in client_names else "parity",
                        "competitor_status": "leading",
                        "note": f"{competitor.name} publicly emphasizes {name}.",
                        "how_competitor_leads": f"{competitor.name} markets {name} as a core differentiator.",
                        "how_to_improve": f"Package and prove a {name} response that sales can demo against {competitor.name}.",
                        "confidence_score": 0.55,
                        "evidence_strength": "medium",
                        "citations": [
                            {
                                "url": competitor.website or "",
                                "snippet": (competitor.evidence_snippet or competitor.description or name)[:280],
                                "source": "website",
                            }
                        ]
                        if competitor.website
                        else [],
                    },
                    client.name,
                    competitor.name,
                )
            )
        for feat in features[:4]:
            if any(r and r.get("feature_name", "").lower() == feat.name.lower() for r in cleaned_rows if r):
                continue
            cleaned_rows.append(
                _normalize_comparison_row(
                    {
                        "feature_name": feat.name,
                        "category": feat.category or "General",
                        "our_status": "parity",
                        "competitor_status": "leading",
                        "note": f"Compare {feat.name} depth vs {competitor.name}.",
                        "how_competitor_leads": f"{competitor.name} may out-package {feat.name} in buyer conversations.",
                        "how_to_improve": f"Tighten messaging, proof, and demo narrative for {feat.name}.",
                        "confidence_score": 0.5,
                        "evidence_strength": "low",
                    },
                    client.name,
                    competitor.name,
                )
            )
        cleaned_rows = [r for r in cleaned_rows if r]

    deduped: list[dict] = []
    seen: set[str] = set()
    for row in cleaned_rows:
        key = _as_str(row["feature_name"]).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return {"competitor_name": competitor.name, "rows": deduped[:8]}


def _extract_features_from_markdown(markdown: str, limit: int = 12) -> list[dict]:
    features: list[dict] = []
    seen: set[str] = set()
    skip = {"contact us", "home", "about", "privacy", "terms", "blog", "careers", "login", "sign in"}
    for raw in (markdown or "").splitlines():
        line = raw.strip().lstrip("#*-• ").strip()
        if not line or len(line) < 3 or len(line) > 80:
            continue
        if "http" in line.lower() or line.startswith("["):
            continue
        words = line.split()
        if len(words) > 8:
            continue
        lowered = line.lower()
        if lowered in seen or lowered in skip:
            continue
        seen.add(lowered)
        features.append(
            {
                "name": line,
                "category": "Capability",
                "description": f"Publicly listed capability: {line}",
            }
        )
        if len(features) >= limit:
            break
    return features



# Hyperscalers, Big 4, mega SIs, and platform giants — not niche peer rivals.
_GLOBAL_RIVAL_BLOCKLIST = {
    "accenture", "ibm", "ibm watson", "watson", "microsoft", "microsoft ai", "microsoft azure",
    "azure", "google", "google cloud", "google cloud ai", "google ai", "dialogflow", "amazon",
    "aws", "amazon web services", "oracle", "oracle ai", "sap", "sap leonardo", "deloitte",
    "pwc", "ey", "ernst & young", "kpmg", "cognizant", "infosys", "capgemini", "tcs",
    "tata consultancy", "wipro", "meta", "openai", "anthropic", "salesforce", "adobe",
    "nvidia", "mckinsey", "bain", "bcg", "boston consulting", "slalom", "thoughtworks",
    "manychat", "converse.ai", "inbenta",
}

_GLOBAL_DOMAIN_BLOCKLIST = {
    "accenture.com", "ibm.com", "microsoft.com", "azure.microsoft.com", "google.com",
    "cloud.google.com", "dialogflow.cloud.google.com", "amazon.com", "aws.amazon.com",
    "oracle.com", "sap.com", "deloitte.com", "pwc.com", "ey.com", "kpmg.com",
    "cognizant.com", "infosys.com", "capgemini.com", "tcs.com", "wipro.com",
    "openai.com", "anthropic.com", "salesforce.com", "adobe.com", "nvidia.com",
    "mckinsey.com", "bain.com", "bcg.com", "slalom.com", "thoughtworks.com",
    "manychat.com", "converse.ai", "inbenta.com",
}


def _domain_of(url: str) -> str:
    raw = _as_str(url).strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:
        from urllib.parse import urlparse

        host = (urlparse(raw).hostname or "").lower()
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_global_megarival(name: str, website: str | None = None) -> bool:
    n = _as_str(name).strip().lower()
    if not n:
        return False
    if n in _GLOBAL_RIVAL_BLOCKLIST:
        return True
    for blocked in _GLOBAL_RIVAL_BLOCKLIST:
        if len(blocked) < 4:
            continue
        if blocked == n or n.startswith(blocked + " ") or n.endswith(" " + blocked) or f" {blocked} " in f" {n} ":
            return True
        if blocked in n and blocked not in {"ai", "aws", "ibm", "sap", "ey", "tcs", "bcg"}:
            return True
    host = _domain_of(website or "")
    if host:
        for blocked in _GLOBAL_DOMAIN_BLOCKLIST:
            if host == blocked or host.endswith("." + blocked):
                return True
    return False


def _market_area_from_client(client: ClientBrand) -> str:
    notes = _as_str(client.notes)
    for line in notes.splitlines():
        if line.lower().startswith("market:"):
            return line.split(":", 1)[1].strip()
    return ""


def _set_market_area(client: ClientBrand, market_area: str) -> None:
    market_area = _as_str(market_area).strip()
    notes = _as_str(client.notes)
    lines = [ln for ln in notes.splitlines() if not ln.lower().startswith("market:")]
    if market_area:
        lines.insert(0, f"Market: {market_area}")
    client.notes = "\n".join(lines).strip() or None


def _niche_competitor_queries(client: ClientBrand, market_area: str = "") -> list[str]:
    niche = _as_str(client.niche) or _as_str(client.industry) or "software"
    market = market_area or _market_area_from_client(client)
    queries = [
        f"{client.name} competitors {niche}",
        f"companies like {client.name} {niche}",
        f"{niche} agencies competitors {client.name}",
    ]
    if market:
        queries.extend(
            [
                f"{niche} companies in {market}",
                f"{niche} agencies {market} like {client.name}",
                f"{client.name} competitors {market}",
            ]
        )
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        key = q.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(q.strip())
    return out[:5]


def _filter_niche_competitors(
    items: list[dict],
    client_name: str,
    *,
    market_area: str = "",
    niche: str = "",
) -> list[dict]:
    deduped: list[dict] = []
    seen_names: set[str] = set()
    client_l = client_name.lower()
    niche_l = niche.lower()
    market_l = market_area.lower()

    for item in items:
        if not isinstance(item, dict):
            continue
        name = _as_str(item.get("name")).strip()
        website = _as_str(item.get("website")) or None
        if not name or name.lower() == client_l or name.lower() in seen_names:
            continue
        if _is_global_megarival(name, website):
            continue
        if item.get("same_niche") is False or item.get("is_global_platform") is True:
            continue
        why = _as_str(item.get("why_relevant") or item.get("description"))
        try:
            score = float(item.get("overlap_score") or item.get("niche_fit_score") or 60)
        except (TypeError, ValueError):
            score = 60.0
        blob = f"{name} {why} {website or ''}".lower()
        if niche_l and any(tok in blob for tok in niche_l.replace("/", " ").split() if len(tok) > 3):
            score += 8
        if market_l and any(tok in blob for tok in market_l.replace(",", " ").split() if len(tok) > 2):
            score += 10
        seen_names.add(name.lower())
        deduped.append(
            {
                **item,
                "name": name,
                "website": website,
                "why_relevant": why or item.get("why_relevant"),
                "overlap_score": min(score, 95),
                "threat_level": _as_str(item.get("threat_level"), "high").lower(),
            }
        )
        if len(deduped) >= 10:
            break
    return deduped


def _competitors_from_serp(organic: list[dict], client_name: str) -> list[dict]:
    rivals: list[dict] = []
    seen: set[str] = set()
    client_l = client_name.lower()
    skip_title_bits = ("vs ", " versus ", "alternative", "alternatives", "best ", "top ", "compared")
    for item in organic or []:
        title = _as_str(item.get("title"))
        link = _as_str(item.get("link"))
        snippet = _as_str(item.get("snippet"))
        if not title or not link:
            continue
        name = title.split("|")[0].split("-")[0].split("–")[0].strip()
        if not name or client_l in name.lower() or len(name) > 60:
            continue
        lowered = name.lower()
        if any(bit in lowered for bit in skip_title_bits):
            continue
        if _is_global_megarival(name, link):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        rivals.append(
            {
                "name": name,
                "website": link.split("?")[0],
                "why_relevant": snippet[:220] or f"Appears in niche search for {client_name} competitors",
                "threat_level": "high",
                "overlap_score": 68,
            }
        )
        if len(rivals) >= 10:
            break
    return rivals



async def enrich_client_profile(db: AsyncSession, agency: Agency, client: ClientBrand) -> dict:
    site = {}
    if client.website:
        site = await scrape_website(db, agency.id, client.website)
    site_md = (site.get("markdown") or "")[:4500]

    profile = await ai_service.structured_json(
        db,
        agency.id,
        (
            "Profile this company for competitive intelligence. "
            "Return JSON keys ONLY: industry, niche, market_area, business_model, tagline, description, "
            "goals (3-5 strings), features (8-12 objects with name, category, description). "
            "niche = specific category (not just 'AI' or 'Software'). "
            "market_area = concrete city/region/country they sell into "
            "(e.g. 'Pakistan', 'Karachi', 'UAE', 'MENA', 'US mid-market'). "
            "Never return only 'Global' or 'Worldwide' — use HQ or primary selling region from contact/address/phone clues. "
            "business_model = agency|product|saas|services|marketplace|other. "
            "Use the website excerpt. Be concrete. No filler."
        ),
        json.dumps(
            {
                "name": client.name,
                "website": client.website,
                "industry_hint": client.industry,
                "site_markdown": site_md,
            }
        )[:7000],
        temperature=0.2,
    )

    if not isinstance(profile.get("features"), list) or not profile.get("features"):
        profile = await ai_service.structured_json(
            db,
            agency.id,
            (
                "Return JSON: {industry, niche, market_area, business_model, tagline, description, "
                "goals:[], features:[{name, category, description}]}."
            ),
            json.dumps(
                {
                    "name": client.name,
                    "website": client.website,
                    "excerpt": site_md[:2500],
                }
            ),
            temperature=0.15,
        )

    client.industry = _as_str(profile.get("industry")) or client.industry or "Software"
    client.niche = _as_str(profile.get("niche")) or client.niche
    client.tagline = _as_str(profile.get("tagline")) or client.tagline
    market_area = _as_str(profile.get("market_area")) or _market_area_from_client(client)
    if market_area.strip().lower() in {"global", "worldwide", "international", "world"}:
        market_area = ""
    business_model = _as_str(profile.get("business_model")) or "services"
    description = _as_str(profile.get("description"))
    if description:
        client.notes = description
    _set_market_area(client, market_area)

    goals = profile.get("goals") if isinstance(profile.get("goals"), list) else []
    client.goals = [_as_str(g) for g in goals if _as_str(g)] or client.goals or [
        "Win more competitive deals",
        "Close product gaps vs leading rivals",
        "Improve category positioning",
    ]

    feature_items = profile.get("features") if isinstance(profile.get("features"), list) else []
    if not feature_items:
        feature_items = _extract_features_from_markdown(site_md)
    if not feature_items:
        feature_items = [
            {"name": "AI Strategy", "category": "Advisory", "description": "AI strategy and consultation offerings"},
            {"name": "Machine Learning", "category": "Capability", "description": "Machine learning solutions"},
            {"name": "AI Automation", "category": "Implementation", "description": "Automation and implementation services"},
            {"name": "Conversational AI", "category": "Product", "description": "Chatbot / conversational AI"},
            {"name": "Enterprise Delivery", "category": "Services", "description": "Enterprise-grade software delivery"},
        ]

    existing_features = (
        await db.execute(
            select(ProductFeature).where(ProductFeature.client_id == client.id, ProductFeature.agency_id == agency.id)
        )
    ).scalars().all()
    features_by_name = {_as_str(f.name).lower(): f for f in existing_features}
    feature_rows: list[ProductFeature] = list(existing_features)
    for item in feature_items[:14]:
        if not isinstance(item, dict):
            name = _as_str(item).strip()
            item = {"name": name, "category": "General", "description": name}
        name = _as_str(item.get("name"), "Feature").strip()
        if not name:
            continue
        key = name.lower()
        if key in features_by_name:
            feat = features_by_name[key]
            feat.category = _as_str(item.get("category") or feat.category, "General")
            if item.get("description"):
                feat.description = _as_str(item.get("description"))
        else:
            feature = ProductFeature(
                agency_id=agency.id,
                client_id=client.id,
                name=name,
                category=_as_str(item.get("category"), "General"),
                description=_as_str(item.get("description")),
            )
            db.add(feature)
            features_by_name[key] = feature
            feature_rows.append(feature)

    competitor_prompt = (
        "Find 8-10 REAL direct competitors for this company. "
        "They must be from the SAME niche, SAME business model, and preferably the SAME market/area "
        "(city, country, or region the client sells into). "
        "Return JSON: {competitors:[{name, website, why_relevant, threat_level, overlap_score, "
        "same_niche:true, market_overlap, is_global_platform:false}]}. "
        "Hard rules:\n"
        "1) Prefer local/regional peer agencies, boutiques, or product companies that chase the same buyers.\n"
        "2) EXCLUDE global hyperscalers and mega consultancies "
        "(Accenture, IBM, Microsoft, Google, Amazon/AWS, Oracle, SAP, Deloitte, PwC, EY, KPMG, Cognizant, Infosys, TCS, Wipro, OpenAI).\n"
        "3) EXCLUDE platforms that are tools/infrastructure rather than peer businesses "
        "(Dialogflow, Azure AI, Watson as platforms, ManyChat).\n"
        "4) If market_area is set, bias strongly toward rivals operating in that area or selling to that market.\n"
        "5) why_relevant must say how they overlap on niche + buyers + geography.\n"
        "6) Only include companies you believe actually exist."
    )
    competitor_pack = await ai_service.structured_json(
        db,
        agency.id,
        competitor_prompt,
        json.dumps(
            {
                "name": client.name,
                "website": client.website,
                "industry": client.industry,
                "niche": client.niche,
                "market_area": market_area,
                "business_model": business_model,
                "features": [f.name for f in feature_rows[:10]],
                "site_excerpt": site_md[:2000],
            }
        ),
        temperature=0.2,
    )
    competitor_items: list[dict] = []
    if isinstance(competitor_pack.get("competitors"), list):
        competitor_items = [c for c in competitor_pack["competitors"] if isinstance(c, dict)]

    competitor_items = _filter_niche_competitors(
        competitor_items, client.name, market_area=market_area, niche=_as_str(client.niche)
    )

    if len(competitor_items) < 4:
        for query in _niche_competitor_queries(client, market_area):
            serp = await serp_visibility(db, agency.id, query)
            competitor_items.extend(_competitors_from_serp(serp.get("organic") or [], client.name))
            competitor_items = _filter_niche_competitors(
                competitor_items, client.name, market_area=market_area, niche=_as_str(client.niche)
            )
            if len(competitor_items) >= 4:
                break

    if len(competitor_items) < 4:
        retry_pack = await ai_service.structured_json(
            db,
            agency.id,
            (
                "Propose niche peer competitors only (same category + similar company size/model). "
                "Return JSON {competitors:[{name, website, why_relevant, threat_level, overlap_score, same_niche:true}]}. "
                "No Fortune-500 tech giants. Prefer regional/local firms in the client's market_area."
            ),
            json.dumps(
                {
                    "name": client.name,
                    "niche": client.niche,
                    "industry": client.industry,
                    "market_area": market_area,
                    "business_model": business_model,
                    "already_have": [c.get("name") for c in competitor_items],
                }
            ),
            temperature=0.25,
        )
        if isinstance(retry_pack.get("competitors"), list):
            competitor_items.extend([c for c in retry_pack["competitors"] if isinstance(c, dict)])
        competitor_items = _filter_niche_competitors(
            competitor_items, client.name, market_area=market_area, niche=_as_str(client.niche)
        )

    deduped = competitor_items[:10]

    existing = (
        await db.execute(select(Competitor).where(Competitor.client_id == client.id, Competitor.agency_id == agency.id))
    ).scalars().all()
    by_name = {_as_str(c.name).lower(): c for c in existing}

    pruned_global = 0
    for competitor in existing:
        if competitor.is_pinned:
            continue
        if _is_global_megarival(competitor.name, competitor.website):
            competitor.is_tracking = False
            competitor.threat_level = "low"
            competitor.overlap_score = min(float(competitor.overlap_score or 0), 30)
            pruned_global += 1

    created_competitors = 0
    for item in deduped:
        name = _as_str(item.get("name")).strip()
        threat = _as_str(item.get("threat_level"), "high").lower()
        try:
            overlap = float(item.get("overlap_score") or 70)
        except (TypeError, ValueError):
            overlap = 70.0
        key = name.lower()
        why = _as_str(item.get("why_relevant"))
        if key in by_name:
            competitor = by_name[key]
            competitor.website = _as_str(item.get("website")) or competitor.website
            competitor.description = why or competitor.description
            competitor.why_dangerous = why or competitor.why_dangerous
            competitor.threat_level = threat if threat in {"medium", "high"} else "high"
            competitor.overlap_score = max(overlap, competitor.overlap_score or 0)
            competitor.is_tracking = True
        else:
            competitor = Competitor(
                agency_id=agency.id,
                client_id=client.id,
                name=name,
                website=_as_str(item.get("website")) or None,
                description=why or None,
                why_dangerous=why or None,
                threat_level=threat if threat in {"medium", "high"} else "high",
                overlap_score=overlap,
                is_tracking=True,
            )
            db.add(competitor)
            created_competitors += 1

    await db.flush()
    return {
        "features": len(feature_rows),
        "competitors_added": created_competitors,
        "competitors_pruned_global": pruned_global,
        "goals": len(client.goals or []),
        "industry": client.industry,
        "niche": client.niche,
        "market_area": market_area,
        "business_model": business_model,
    }



async def run_competitive_pack(db: AsyncSession, agency: Agency, client: ClientBrand) -> dict:
    features = (
        await db.execute(
            select(ProductFeature).where(
                ProductFeature.client_id == client.id,
                ProductFeature.agency_id == agency.id,
            )
        )
    ).scalars().all()
    competitors = (
        await db.execute(
            select(Competitor).where(
                Competitor.client_id == client.id,
                Competitor.agency_id == agency.id,
                Competitor.is_tracking.is_(True),
            )
        )
    ).scalars().all()

    globalish = sum(1 for c in competitors if _is_global_megarival(c.name, c.website))
    needs_refresh = (not features or not competitors) or (competitors and globalish >= max(2, len(competitors) // 2))
    if needs_refresh:
        await enrich_client_profile(db, agency, client)
        features = (
            await db.execute(
                select(ProductFeature).where(
                    ProductFeature.client_id == client.id,
                    ProductFeature.agency_id == agency.id,
                )
            )
        ).scalars().all()
        competitors = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                    Competitor.is_tracking.is_(True),
                )
            )
        ).scalars().all()

    if not features or not competitors:
        raise ValueError(
            "Could not build features/rivals for this client. Add a website, then run intel again "
            "or add features and competitors manually."
        )

    kept: list[Competitor] = []
    analyzed: list[Competitor] = []
    for competitor in competitors:
        site_data = {}
        if competitor.website:
            site_data = await scrape_website(db, agency.id, competitor.website)
        analysis = await ai_service.structured_json(
            db,
            agency.id,
            (
                "Enrich a competitor for competitive intelligence against THIS client only. "
                "Return JSON keys: tagline, description, headquarters, overlap_score (0-100), "
                "threat_level (low|medium|high), is_leading_rival (boolean), same_niche (boolean), "
                "same_market (boolean), is_global_platform (boolean), why_dangerous (1-2 sentences), "
                "evidence_snippet (short quote/paraphrase from site), "
                "features (array of {name, category, description}). "
                "Score overlap high only when niche + buyer + geography truly match. "
                "Global hyperscalers/platforms that are not peer businesses should be low threat, same_niche=false, is_global_platform=true."
            ),
            json.dumps(
                {
                    "client": client.name,
                    "client_industry": client.industry,
                    "client_niche": client.niche,
                    "client_market_area": _market_area_from_client(client),
                    "client_features": [
                        {"name": f.name, "category": f.category, "description": f.description} for f in features
                    ],
                    "competitor": {
                        "name": competitor.name,
                        "website": competitor.website,
                        "site_excerpt": (site_data.get("markdown") or "")[:3500],
                    },
                }
            )[:9000],
            temperature=0.2,
        )
        # If AI fallback text returned, keep prior competitor values and treat as trackable
        if "summary" in analysis and "features" not in analysis:
            analysis = {
                "overlap_score": competitor.overlap_score or 70,
                "threat_level": competitor.threat_level or "high",
                "is_leading_rival": True,
                "why_dangerous": competitor.why_dangerous or competitor.description or f"{competitor.name} competes for the same buyers.",
                "features": competitor.feature_list or [],
            }

        competitor.tagline = _as_str(analysis.get("tagline")) or competitor.tagline
        competitor.description = _as_str(analysis.get("description")) or competitor.description
        competitor.headquarters = _as_str(analysis.get("headquarters")) or competitor.headquarters
        try:
            competitor.overlap_score = float(analysis.get("overlap_score") or competitor.overlap_score or 65)
        except (TypeError, ValueError):
            competitor.overlap_score = competitor.overlap_score or 65
        competitor.threat_level = _as_str(analysis.get("threat_level") or competitor.threat_level or "medium").lower()
        if competitor.threat_level not in {"low", "medium", "high"}:
            competitor.threat_level = "medium"
        competitor.feature_list = analysis.get("features") if isinstance(analysis.get("features"), list) else (competitor.feature_list or [])
        competitor.why_dangerous = _as_str(analysis.get("why_dangerous")) or competitor.why_dangerous
        competitor.evidence_snippet = _as_str(analysis.get("evidence_snippet")) or competitor.evidence_snippet
        if site_data.get("markdown") and not competitor.evidence_snippet:
            competitor.evidence_snippet = (site_data.get("markdown") or "")[:280]
        competitor.last_scraped_at = datetime.utcnow()
        analyzed.append(competitor)

        off_niche = (
            not competitor.is_pinned
            and (
                _is_global_megarival(competitor.name, competitor.website)
                or analysis.get("is_global_platform") is True
                or analysis.get("same_niche") is False
                or (
                    competitor.threat_level == "low"
                    and (competitor.overlap_score or 0) < 45
                    and analysis.get("is_leading_rival") is False
                )
            )
        )
        if off_niche:
            competitor.is_tracking = False
            competitor.threat_level = "low"
            continue

        competitor.is_tracking = True
        if competitor.threat_level == "low":
            competitor.threat_level = "medium"
        kept.append(competitor)

    if not kept and analyzed:
        # Never leave a client with zero rivals after enrichment — keep strongest overlaps
        analyzed_sorted = sorted(analyzed, key=lambda c: float(c.overlap_score or 0), reverse=True)
        for competitor in analyzed_sorted[:6]:
            competitor.is_tracking = True
            if competitor.threat_level not in {"medium", "high"}:
                competitor.threat_level = "medium"
            kept.append(competitor)

    competitors = kept[:10]
    if not competitors:
        raise ValueError("No competitors available for this client after enrichment.")

    comparisons_payload: list[dict] = []
    for competitor in competitors:
        block = await _generate_competitor_comparisons(db, agency, client, list(features), competitor)
        comparisons_payload.append(block)

    pack = await ai_service.structured_json(
        db,
        agency.id,
        (
            "Build gap reports and goal-weighted alerts. "
            "Return JSON with keys: "
            "gap_reports (array of {competitor_name, summary, leading[], lagging[], opportunities[]}), "
            "goal_alerts (array of {goal, title, why_it_matters, impact, action, content_draft, estimated_cost, competitor_trigger, missing_feature}), "
            "highlights (string array of sharp executive takeaways). "
            "impact MUST be exactly one of: low | medium | high (never a sentence). "
            "ALERT RULE: only create alerts for features/specialties competitors have that the client does NOT have. "
            "Do not alert on features the client already owns. Be specific. No generic filler."
        ),
        json.dumps(
            {
                "client": {
                    "name": client.name,
                    "industry": client.industry,
                    "niche": client.niche,
                    "tagline": client.tagline,
                    "goals": client.goals or [],
                    "features": [
                        {"name": f.name, "category": f.category, "description": f.description} for f in features
                    ],
                },
                "competitors": [
                    {
                        "name": c.name,
                        "overlap_score": c.overlap_score,
                        "threat_level": c.threat_level,
                        "tagline": c.tagline,
                        "features": c.feature_list or [],
                    }
                    for c in competitors
                ],
                "comparison_snapshot": comparisons_payload,
            }
        )[:14000],
        temperature=0.35,
    )
    pack["comparisons"] = comparisons_payload

    await db.execute(delete(FeatureComparison).where(FeatureComparison.client_id == client.id))
    await db.execute(delete(GapReport).where(GapReport.client_id == client.id))
    await db.execute(delete(GoalAlert).where(GoalAlert.client_id == client.id))

    name_to_comp = {_as_str(c.name).lower(): c for c in competitors}
    comparison_count = 0
    for block in pack.get("comparisons", []):
        comp = name_to_comp.get(_as_str(block.get("competitor_name")).lower())
        if not comp:
            continue
        for row in block.get("rows", [])[:10]:
            cleaned = _normalize_comparison_row(row, client.name, comp.name)
            if not cleaned:
                continue
            db.add(
                FeatureComparison(
                    agency_id=agency.id,
                    client_id=client.id,
                    competitor_id=comp.id,
                    competitor_name=comp.name,
                    feature_name=cleaned["feature_name"],
                    category=cleaned["category"],
                    our_status=cleaned["our_status"],
                    competitor_status=cleaned["competitor_status"],
                    note=cleaned["note"],
                    how_competitor_leads=cleaned["how_competitor_leads"],
                    how_to_improve=cleaned["how_to_improve"],
                    citations=cleaned["citations"],
                    confidence_score=cleaned["confidence_score"],
                    evidence_strength=cleaned["evidence_strength"],
                    is_contested_move=cleaned["is_contested_move"],
                )
            )
            comparison_count += 1

    gap_count = 0
    for gap in pack.get("gap_reports", []) or []:
        comp = name_to_comp.get(_as_str(gap.get("competitor_name")).lower())
        if not comp:
            continue
        summary = _as_str(gap.get("summary")).strip()
        if not summary:
            continue
        leading = gap.get("leading") if isinstance(gap.get("leading"), list) else []
        lagging = gap.get("lagging") if isinstance(gap.get("lagging"), list) else []
        opportunities = gap.get("opportunities") if isinstance(gap.get("opportunities"), list) else []
        citations = gap.get("citations") if isinstance(gap.get("citations"), list) else []
        if not citations and comp.website:
            citations = [
                {
                    "url": comp.website or "",
                    "snippet": _as_str(comp.evidence_snippet or comp.description)[:300],
                    "source": "website",
                }
            ]
        try:
            conf = float(gap.get("confidence_score") or 0.6)
        except (TypeError, ValueError):
            conf = 0.6
        db.add(
            GapReport(
                agency_id=agency.id,
                client_id=client.id,
                competitor_id=comp.id,
                competitor_name=comp.name,
                summary=summary,
                leading=leading,
                lagging=lagging,
                opportunities=opportunities,
                citations=citations,
                confidence_score=conf,
                evidence_strength=_as_str(gap.get("evidence_strength"), "medium"),
            )
        )
        gap_count += 1

    if gap_count == 0:
        for comp in competitors:
            rival_rows = [b for b in comparisons_payload if _as_str(b.get("competitor_name")).lower() == comp.name.lower()]
            leading_feats = []
            opportunities = []
            for block in rival_rows:
                for row in block.get("rows") or []:
                    cleaned = row if "our_status" in row and "feature_name" in row else None
                    if not cleaned:
                        continue
                    if cleaned.get("competitor_status") == "leading":
                        leading_feats.append(cleaned["feature_name"])
                    if cleaned.get("our_status") == "lagging":
                        opportunities.append(cleaned.get("how_to_improve") or f"Improve {cleaned['feature_name']}")
            if not leading_feats and comp.feature_list:
                for f in comp.feature_list[:5]:
                    if isinstance(f, dict) and f.get("name"):
                        leading_feats.append(_as_str(f.get("name")))
            summary = (
                f"{comp.name} leads on {', '.join(leading_feats[:4])}."
                if leading_feats
                else f"{comp.name} remains a high-overlap rival ({int(comp.overlap_score or 0)}% overlap) that can pressure {client.name} in deals."
            )
            db.add(
                GapReport(
                    agency_id=agency.id,
                    client_id=client.id,
                    competitor_id=comp.id,
                    competitor_name=comp.name,
                    summary=summary,
                    leading=leading_feats[:8],
                    lagging=[],
                    opportunities=(opportunities or [f"Build a sharper counter-narrative vs {comp.name}"])[:8],
                    citations=[
                        {
                            "url": comp.website or "",
                            "snippet": _as_str(comp.evidence_snippet or comp.why_dangerous or comp.description)[:300],
                            "source": "website",
                        }
                    ]
                    if comp.website
                    else [],
                    confidence_score=0.62,
                    evidence_strength="medium",
                )
            )
            gap_count += 1

    alert_count = 0
    for alert in pack.get("goal_alerts", []) or []:
        title = _as_str(alert.get("title"), "Goal alert").strip()
        why = _as_str(alert.get("why_it_matters")).strip()
        action = _as_str(alert.get("action")).strip()
        if not title or not why:
            continue
        citations = alert.get("citations") if isinstance(alert.get("citations"), list) else []
        try:
            conf = float(alert.get("confidence_score") or 0.6)
        except (TypeError, ValueError):
            conf = 0.6
        db.add(
            GoalAlert(
                agency_id=agency.id,
                client_id=client.id,
                goal=_clip(_as_str(alert.get("goal") or ((client.goals or ["Grow market share"])[0])), 500),
                title=_clip(title, 500),
                why_it_matters=why,
                impact=_level_label(alert.get("impact"), "medium", max_len=255),
                action=action or f"Prioritize a response to {title}",
                content_draft=_as_str(alert.get("content_draft")),
                estimated_cost=_clip(_as_str(alert.get("estimated_cost")), 120),
                competitor_trigger=_clip(
                    _as_str(alert.get("competitor_trigger") or alert.get("missing_feature")), 255
                ),
                citations=citations,
                confidence_score=conf,
                evidence_strength=_level_label(alert.get("evidence_strength"), "medium"),
            )
        )
        alert_count += 1

    if alert_count == 0:
        seen_alert: set[str] = set()
        client_feat_names = {f.name.lower() for f in features}

        def _add_specialty_alert(
            *,
            feat: str,
            comp_name: str,
            why: str,
            action: str,
            citations: list | None = None,
            confidence: float = 0.6,
            evidence: str = "medium",
        ) -> None:
            nonlocal alert_count
            key = feat.lower()
            if not feat or key in seen_alert or alert_count >= 8:
                return
            if key in client_feat_names:
                return
            seen_alert.add(key)
            db.add(
                GoalAlert(
                    agency_id=agency.id,
                    client_id=client.id,
                    goal=_clip(((client.goals or ["Close competitive gaps"])[0]), 500),
                    title=_clip(f"Missing specialty: {feat}", 500),
                    why_it_matters=why,
                    impact="high",
                    action=action,
                    content_draft=f"Buyers comparing you to {comp_name} will ask about {feat}. Prepare a gap-close narrative this week.",
                    estimated_cost="1-2 sprints",
                    competitor_trigger=_clip(comp_name, 255),
                    citations=citations or [],
                    confidence_score=confidence,
                    evidence_strength=_level_label(evidence, "medium"),
                )
            )
            alert_count += 1

        for block in comparisons_payload:
            comp_name = _as_str(block.get("competitor_name"))
            for row in block.get("rows") or []:
                feat = _as_str(row.get("feature_name")).strip()
                our = _as_str(row.get("our_status")).lower()
                theirs = _as_str(row.get("competitor_status")).lower()
                if theirs != "leading" and our not in {"lagging", "missing", "weak", "none", "absent"}:
                    continue
                try:
                    conf = float(row.get("confidence_score") or 0.6)
                except (TypeError, ValueError):
                    conf = 0.6
                _add_specialty_alert(
                    feat=feat,
                    comp_name=comp_name,
                    why=_as_str(row.get("how_competitor_leads"))
                    or f"{comp_name} has {feat} as a specialty you lack or lag on.",
                    action=_as_str(row.get("how_to_improve")) or f"Add {feat} to wishlist and ship a development plan.",
                    citations=row.get("citations") if isinstance(row.get("citations"), list) else [],
                    confidence=conf,
                    evidence=_as_str(row.get("evidence_strength"), "medium"),
                )
                if alert_count >= 8:
                    break
            if alert_count >= 8:
                break

        if alert_count == 0:
            for comp in competitors:
                for f in comp.feature_list or []:
                    name = _as_str(f.get("name") if isinstance(f, dict) else f).strip()
                    _add_specialty_alert(
                        feat=name,
                        comp_name=comp.name,
                        why=f"{comp.name} lists {name} as a product specialty that {client.name} does not currently advertise.",
                        action=f"Add {name} to wishlist and draft a development plan this week.",
                        citations=[
                            {
                                "url": comp.website or "",
                                "snippet": _as_str(comp.evidence_snippet or comp.description or name)[:300],
                                "source": "website",
                            }
                        ]
                        if comp.website
                        else [],
                    )
                    if alert_count >= 8:
                        break
                if alert_count >= 8:
                    break

    await db.flush()
    return {
        "competitors": len(competitors),
        "comparisons": comparison_count,
        "gaps": gap_count,
        "alerts": alert_count,
        "highlights": pack.get("highlights") or [],
    }


async def love_feature_and_build_tickets(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    feature: ProductFeature,
) -> list[FeatureTicket]:
    feature.is_loved = True
    feature.is_wishlisted = True
    comparisons = (
        await db.execute(
            select(FeatureComparison).where(
                FeatureComparison.client_id == client.id,
                FeatureComparison.agency_id == agency.id,
                FeatureComparison.feature_name == feature.name,
            )
        )
    ).scalars().all()
    gaps = (
        await db.execute(
            select(GapReport).where(GapReport.client_id == client.id, GapReport.agency_id == agency.id)
        )
    ).scalars().all()
    alerts = (
        await db.execute(
            select(GoalAlert).where(GoalAlert.client_id == client.id, GoalAlert.agency_id == agency.id)
        )
    ).scalars().all()

    evidence = []
    for c in comparisons:
        for cite in c.citations or []:
            evidence.append(cite)
        evidence.append(
            {
                "url": "",
                "snippet": f"{c.competitor_name}: {c.how_competitor_leads}"[:350],
                "source": "comparison",
            }
        )

    # Keep AI under platform proxy limits (~60s). Fall back to templates on timeout/failure.
    payload: dict = {}
    try:
        payload = await asyncio.wait_for(
            ai_service.structured_json(
                db,
                agency.id,
                (
                    "The marketing agency / client manually selected this feature. Create BOARD-READY Jira work. "
                    "Return JSON {tickets:[{heading, body, acceptance_criteria[], priority, ticket_type, labels[], "
                    "estimated_effort, story_points, why_useful, competitor_context, evidence_links:[{url, snippet, source}]}]}. "
                    "Requirements:\n"
                    "- Exactly 1 epic first, then 4-6 stories/tasks under that theme.\n"
                    "- ticket_type must be epic|story|task.\n"
                    "- Each story must have 3-5 measurable acceptance criteria.\n"
                    "- Include effort estimate and story_points (1-8).\n"
                    "- Link competitor evidence in competitor_context and evidence_links.\n"
                    "- Cover: packaging, GTM, sales enablement, demo, analytics.\n"
                    "- No filler. Valid, shippable tickets only. Keep responses concise."
                ),
                json.dumps(
                    {
                        "feature": {
                            "name": feature.name,
                            "category": feature.category,
                            "description": feature.description,
                        },
                        "client": {"name": client.name, "goals": (client.goals or [])[:5]},
                        "feature_comparisons": [
                            {
                                "competitor": c.competitor_name,
                                "our_status": c.our_status,
                                "competitor_status": c.competitor_status,
                                "how_to_improve": (c.how_to_improve or "")[:280],
                                "how_competitor_leads": (c.how_competitor_leads or "")[:280],
                                "confidence_score": c.confidence_score,
                            }
                            for c in comparisons[:6]
                        ],
                        "related_gaps": [
                            {
                                "competitor": g.competitor_name,
                                "opportunities": (g.opportunities or [])[:3],
                                "summary": (g.summary or "")[:220],
                            }
                            for g in gaps[:4]
                        ],
                        "related_alerts": [
                            {"title": a.title, "action": (a.action or "")[:160]} for a in alerts[:4]
                        ],
                        "seed_evidence": evidence[:6],
                    }
                )[:7000],
                temperature=0.3,
            ),
            timeout=35,
        )
    except asyncio.TimeoutError:
        logger.warning("development-plan AI timed out for feature=%s — using templates", feature.id)
        payload = {}
    except Exception:
        logger.exception("development-plan AI failed for feature=%s — using templates", feature.id)
        payload = {}

    await db.execute(delete(FeatureTicket).where(FeatureTicket.feature_id == feature.id))
    tickets: list[FeatureTicket] = []
    epic_id: str | None = None
    items = payload.get("tickets") or []
    if not any(_as_str(i.get("ticket_type")).lower() == "epic" for i in items):
        items = [
            {
                "heading": f"[Epic] Ship {feature.name} competitive response",
                "body": f"Coordinate product, marketing, and sales work to close gaps around {feature.name}.",
                "acceptance_criteria": [
                    "All child stories completed or explicitly deferred",
                    "Weekly brief includes progress vs named rivals",
                    "Agency can demo the packaged narrative",
                ],
                "priority": "high",
                "ticket_type": "epic",
                "labels": [feature.category, "loved-feature", "epic"],
                "estimated_effort": "2-3 sprints",
                "story_points": 0,
                "why_useful": "Creates one parent workstream for the loved feature.",
                "competitor_context": "Derived from contested competitor comparisons.",
                "evidence_links": evidence[:4],
            }
        ] + items

    story_count = sum(1 for i in items if _as_str(i.get("ticket_type")).lower() != "epic")
    if story_count < 5:
        rival_names = sorted({c.competitor_name for c in comparisons}) or ["top rival"]
        templates = [
            ("story", f"Package {feature.name} as a sellable offer", "Rewrite offer page and sales one-pager with proof points.", ["Offer page live", "One-pager approved", "Proof points cited"], "3-5 days", 5),
            ("story", f"Build competitive battlecard vs {rival_names[0]}", f"Document how {feature.name} beats or matches {rival_names[0]}.", ["Battlecard in shared drive", "Sales team briefed", "Objection responses included"], "2-3 days", 3),
            ("story", f"Ship demo narrative for {feature.name}", "Create a 5-minute demo script with talk track and screens.", ["Script reviewed", "Demo recorded", "AE can run unassisted"], "3-4 days", 5),
            ("story", f"Create GTM messaging kit for {feature.name}", "Homepage module, email, LinkedIn, and paid ad variants.", ["4 assets drafted", "Brand review done", "UTM naming set"], "4-5 days", 5),
            ("story", f"Close product gap called out in contested moves", "Implement the highest-confidence gap tied to this feature.", ["Gap ticket scoped", "Acceptance tests pass", "Changelog published"], "1-2 weeks", 8),
            ("task", f"Collect evidence screenshots for {feature.name}", "Capture rival pages and client proof for citations.", ["At least 5 screenshots", "URLs logged", "Shared with report"], "1 day", 2),
        ]
        existing_heads = {_as_str(i.get("heading")).lower() for i in items}
        for ttype, heading, body, criteria, effort, points in templates:
            if heading.lower() in existing_heads:
                continue
            items.append(
                {
                    "heading": heading,
                    "body": body,
                    "acceptance_criteria": criteria,
                    "priority": "high" if ttype == "story" else "medium",
                    "ticket_type": ttype,
                    "labels": [feature.category, "loved-feature", ttype],
                    "estimated_effort": effort,
                    "story_points": points,
                    "why_useful": f"Board-ready work to commercialize {feature.name} against high-risk rivals.",
                    "competitor_context": f"Rivals in scope: {', '.join(rival_names[:4])}",
                    "evidence_links": evidence[:4],
                }
            )
            existing_heads.add(heading.lower())
            if sum(1 for i in items if _as_str(i.get("ticket_type")).lower() != "epic") >= 6:
                break

    for item in items[:8]:
        ttype = _as_str(item.get("ticket_type"), "story").lower()
        if ttype not in {"epic", "story", "task"}:
            ttype = "story"
        criteria = item.get("acceptance_criteria") or []
        if ttype != "epic" and len(criteria) < 3:
            criteria = list(criteria) + [
                "Definition of done reviewed with agency lead",
                "Competitor evidence linked",
                "Deliverable shared with client stakeholder",
            ]
            criteria = criteria[:6]
        ticket = FeatureTicket(
            agency_id=agency.id,
            client_id=client.id,
            feature_id=feature.id,
            heading=item.get("heading", f"Improve {feature.name}"),
            body=item.get("body", ""),
            acceptance_criteria=criteria,
            priority=item.get("priority", "medium"),
            ticket_type=ttype,
            labels=item.get("labels") or [feature.category, "loved-feature"],
            estimated_effort=item.get("estimated_effort", ""),
            story_points=item.get("story_points"),
            why_useful=item.get("why_useful", ""),
            competitor_context=item.get("competitor_context", ""),
            evidence_links=item.get("evidence_links") or evidence[:4],
            parent_ticket_id=None if ttype == "epic" else epic_id,
            status="draft",
        )
        db.add(ticket)
        await db.flush()
        if ttype == "epic" and epic_id is None:
            epic_id = ticket.id
        tickets.append(ticket)
    await db.flush()
    return tickets


async def create_all_feature_tickets_in_jira(
    db: AsyncSession,
    agency_id: str,
    client_id: str,
    feature_id: str,
) -> list[FeatureTicket]:
    connected = (
        await db.execute(
            select(Integration).where(
                Integration.agency_id == agency_id,
                Integration.provider == "jira",
                Integration.is_connected.is_(True),
            )
        )
    ).scalar_one_or_none()
    if not connected or not connected.encrypted_credentials:
        raise ValueError("Connect your Jira account first under Integrations.")

    tickets = (
        await db.execute(
            select(FeatureTicket)
            .where(
                FeatureTicket.feature_id == feature_id,
                FeatureTicket.client_id == client_id,
                FeatureTicket.agency_id == agency_id,
            )
            .order_by(FeatureTicket.created_at.asc())
        )
    ).scalars().all()
    if not tickets:
        raise ValueError("No feature tickets found. Generate a development plan first.")

    epic_jira_key = None
    for ticket in tickets:
        if ticket.jira_key and ticket.ticket_type == "epic":
            epic_jira_key = ticket.jira_key
            break

    errors: list[str] = []
    for ticket in tickets:
        if ticket.jira_key:
            if ticket.ticket_type == "epic" and not epic_jira_key:
                epic_jira_key = ticket.jira_key
            continue
        criteria = "\n".join(f"- {c}" for c in (ticket.acceptance_criteria or []))
        evidence = "\n".join(
            f"- {e.get('source', 'source')}: {e.get('url', '')} :: {(e.get('snippet') or '')[:180]}"
            for e in (ticket.evidence_links or [])
            if isinstance(e, dict)
        )
        description = (
            f"{ticket.body}\n\n"
            f"Why useful:\n{ticket.why_useful}\n\n"
            f"Competitor evidence:\n{ticket.competitor_context}\n\n"
            f"Evidence links:\n{evidence or '- n/a'}\n\n"
            f"Acceptance criteria:\n{criteria}\n\n"
            f"Type: {ticket.ticket_type} | Priority: {ticket.priority} | "
            f"Effort: {ticket.estimated_effort} | Points: {ticket.story_points}\n"
            f"Labels: {', '.join(ticket.labels or [])}"
        )
        try:
            created = await jira_service.create_jira_ticket(
                db,
                agency_id,
                client_id,
                ticket.heading,
                description,
                insight_id=ticket.id,
                issue_type="Epic" if ticket.ticket_type == "epic" else ("Story" if ticket.ticket_type == "story" else "Task"),
                parent_epic_key=None if ticket.ticket_type == "epic" else epic_jira_key,
            )
            ticket.jira_key = created.jira_key
            ticket.jira_url = created.jira_url
            ticket.jira_epic_key = epic_jira_key
            ticket.status = "created"
            if ticket.ticket_type == "epic":
                epic_jira_key = created.jira_key
                ticket.jira_epic_key = created.jira_key
            await db.flush()
        except Exception as exc:
            logger.warning("Jira push failed for ticket=%s: %s", ticket.id, exc)
            errors.append(f"{ticket.heading[:60]}: {exc}")
            # Don't abort the whole batch — continue with remaining tickets
            continue

    await db.flush()
    pushed = sum(1 for t in tickets if t.jira_key)
    if pushed == 0 and errors:
        raise ValueError(errors[0] if len(errors) == 1 else f"Jira push failed ({len(errors)} errors). First: {errors[0]}")
    return list(tickets)


async def run_full_ai_pipeline(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    *,
    push_jira: bool = True,
    generate_report: bool = True,
) -> dict:
    job = TrackingJob(
        agency_id=agency.id,
        client_id=client.id,
        job_type="full_ai_pipeline",
        status=JobStatus.running,
        started_at=datetime.utcnow(),
        detail="Autonomous AI pipeline running",
    )
    db.add(job)
    await db.flush()

    try:
        from app.services.embeddings import index_client_intel
        from app.services.intelligence import run_client_intelligence

        enrich = await enrich_client_profile(db, agency, client)
        pack = await run_competitive_pack(db, agency, client)
        radar = await run_client_intelligence(db, agency, client)

        report_id = None
        if generate_report:
            report = await generate_client_report(db, agency, client, period_label="AI Auto Brief")
            report_id = report.id

        jira_pushed = 0
        if push_jira:
            wishlisted = (
                await db.execute(
                    select(ProductFeature).where(
                        ProductFeature.client_id == client.id,
                        ProductFeature.agency_id == agency.id,
                        ProductFeature.is_wishlisted.is_(True),
                    )
                )
            ).scalars().all()
            for feature in wishlisted[:5]:
                try:
                    tickets = await love_feature_and_build_tickets(db, agency, client, feature)
                    pushed = await create_all_feature_tickets_in_jira(
                        db, agency.id, client.id, feature.id
                    )
                    jira_pushed += len(pushed or tickets or [])
                except Exception:
                    continue

        indexed = 0
        try:
            async with db.begin_nested():
                indexed = await index_client_intel(db, agency.id, client)
        except Exception as emb_exc:
            logger.warning("index_client_intel skipped: %s", emb_exc)
            indexed = 0

        result = {
            "enrich": enrich,
            "pack": pack,
            "radar_job_id": getattr(radar, "id", None),
            "report_id": report_id,
            "jira_tickets_pushed": jira_pushed,
            "embeddings_indexed": indexed,
            "note": "Wishlist items can auto-push to Jira when push_jira=True and Jira is connected.",
        }
        job.status = JobStatus.completed
        job.finished_at = datetime.utcnow()
        job.result_meta = result
        job.detail = "Autonomous AI pipeline completed"
        await db.flush()
        return result
    except Exception as exc:
        job.status = JobStatus.failed
        job.finished_at = datetime.utcnow()
        job.detail = str(exc)[:800]
        await db.flush()
        raise
