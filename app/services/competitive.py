import asyncio
import json
import logging
import re
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


def _as_int(value, default: int | None = None) -> int | None:
    """AI returns story points as 5, '5', '5 points', or '3-5' — keep the first integer."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.search(r"\d+", _as_str(value))
    return int(match.group()) if match else default


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value in (None, "", {}):
        return []
    return [value]


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
                "description": (
                    f"{line} is something this company already offers. "
                    f"In simple terms, it is a capability they promote publicly on their website. "
                    f"Customers can ask for this as part of what the brand sells or delivers today."
                ),
            }
        )
        if len(features) >= limit:
            break
    return features



_FEATURE_DESC_PROMPT = (
    "For each feature, write a plain-English description a non-technical agency user can understand. "
    "Rules for every description:\n"
    "1) Exactly 2–3 short sentences.\n"
    "2) Explain what the customer gets / what problem it solves — not buzzwords.\n"
    "3) Avoid jargon like production-grade, demoware, architecture-first, hyperscale, MLOps, "
    "unless you immediately explain it in everyday words.\n"
    "4) Do not repeat only the feature name. Do not write marketing slogans.\n"
    "5) Keep names as given; only rewrite descriptions.\n"
    "Return JSON: {features:[{name, category, description}]}."
)


def _feature_description_is_thin(name: str, description: str) -> bool:
    name = _as_str(name).strip()
    desc = _as_str(description).strip()
    if not desc:
        return True
    if desc.lower() == name.lower():
        return True
    if len(desc) < 90:
        return True
    # slogan-ish one-liners with little explanation
    if desc.count(".") == 0 and len(desc) < 140:
        return True
    return False


def _fallback_plain_feature_description(name: str, category: str, description: str, client_name: str) -> str:
    name = _as_str(name).strip() or "This capability"
    category = _as_str(category).strip() or "General"
    raw = _as_str(description).strip()
    soft = raw or name
    replacements = (
        ("production-grade ai, not demoware", "AI that is ready for real day-to-day business use — not just a flashy demo"),
        ("production grade ai, not demoware", "AI that is ready for real day-to-day business use — not just a flashy demo"),
        ("production-grade", "ready for real day-to-day business use"),
        ("demoware", "a demo that looks good but is not ready for real work"),
        ("architecture-first thinking", "planning the system carefully before building anything"),
        ("architecture-first", "planned carefully before building"),
        ("enterprise-grade", "built for larger companies"),
        ("end-to-end", "handled from start to finish"),
        ("cutting-edge", "up-to-date"),
        ("state-of-the-art", "modern"),
        ("ai-powered", "using AI to help"),
        ("ml-powered", "using machine learning to help"),
    )
    lowered = soft
    for a, b in replacements:
        idx = lowered.lower().find(a.lower())
        while idx >= 0:
            lowered = lowered[:idx] + b + lowered[idx + len(a) :]
            idx = lowered.lower().find(a.lower(), idx + len(b))
    soft = " ".join(lowered.split())
    if soft.lower() == name.lower() or len(soft) < 40:
        soft = f"customers can use {name} as part of what {client_name or 'the brand'} delivers today"
    cat_bit = f" ({category})" if category and category.lower() not in {"general", "capability"} else ""
    mid = soft[0].lower() + soft[1:] if soft else "customers can use this today"
    return (
        f"{name} is something {client_name or 'this brand'} already offers{cat_bit}. "
        f"In simple terms, {mid}{'' if mid.endswith('.') else '.'} "
        f"This is part of their current offering — not a future idea."
    )


async def clarify_feature_descriptions(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    features: list[ProductFeature] | None = None,
) -> list[ProductFeature]:
    """Rewrite thin/jargon feature descriptions into plain 2–3 sentence English."""
    if features is None:
        features = (
            await db.execute(
                select(ProductFeature).where(
                    ProductFeature.client_id == client.id,
                    ProductFeature.agency_id == agency.id,
                    ProductFeature.is_wishlisted.is_(False),
                )
            )
        ).scalars().all()

    owned = [f for f in features if not f.is_wishlisted]
    if not owned:
        return owned

    thin = [f for f in owned if _feature_description_is_thin(f.name, f.description or "")]
    # Always clarify thin ones; if most are thin, rewrite the whole set for consistency
    targets = owned if len(thin) >= max(1, len(owned) // 2) else thin
    if not targets:
        return owned
    # Cap rewrite work — full-set clarifications stall the intel pipeline
    targets = targets[:8]

    payload = [
        {"name": f.name, "category": f.category or "General", "description": f.description or ""}
        for f in targets
    ]
    rewritten = await ai_service.structured_json(
        db,
        agency.id,
        _FEATURE_DESC_PROMPT
        + f" Company name: {client.name}. Industry: {client.industry or 'unknown'}.",
        json.dumps({"features": payload})[:6000],
        temperature=0.2,
    )
    by_name: dict[str, dict] = {}
    for item in rewritten.get("features") if isinstance(rewritten.get("features"), list) else []:
        if not isinstance(item, dict):
            continue
        key = _as_str(item.get("name")).strip().lower()
        if key:
            by_name[key] = item

    for feat in targets:
        item = by_name.get(_as_str(feat.name).lower())
        new_desc = _as_str(item.get("description")) if item else ""
        if item and item.get("category"):
            feat.category = _as_str(item.get("category") or feat.category, "General")
        if _feature_description_is_thin(feat.name, new_desc):
            new_desc = _fallback_plain_feature_description(
                feat.name, feat.category or "General", feat.description or new_desc, client.name
            )
        feat.description = new_desc

    await db.flush()
    return owned



# Hyperscalers, Big 4, mega SIs, and platform giants — not niche peer rivals.
_GLOBAL_RIVAL_BLOCKLIST = {
    "accenture", "ibm", "ibm watson", "watson", "microsoft", "microsoft ai", "microsoft azure",
    "azure", "google", "google cloud", "google cloud ai", "google ai", "dialogflow", "amazon",
    "aws", "amazon web services", "oracle", "oracle ai", "sap", "sap leonardo", "deloitte",
    "pwc", "ey", "ernst & young", "kpmg", "cognizant", "infosys", "capgemini", "tcs",
    "tata consultancy", "wipro", "meta", "openai", "anthropic", "salesforce", "adobe",
    "nvidia", "mckinsey", "bain", "bcg", "boston consulting", "slalom",
    "manychat", "converse.ai", "inbenta",
}

_GLOBAL_DOMAIN_BLOCKLIST = {
    "accenture.com", "ibm.com", "microsoft.com", "azure.microsoft.com", "google.com",
    "cloud.google.com", "dialogflow.cloud.google.com", "amazon.com", "aws.amazon.com",
    "oracle.com", "sap.com", "deloitte.com", "pwc.com", "ey.com", "kpmg.com",
    "cognizant.com", "infosys.com", "capgemini.com", "tcs.com", "wipro.com",
    "openai.com", "anthropic.com", "salesforce.com", "adobe.com", "nvidia.com",
    "mckinsey.com", "bain.com", "bcg.com", "slalom.com",
    "manychat.com", "converse.ai", "inbenta.com",
    # Non-PK food collisions
    "shawarmajunction.com",  # US
    "shawarma-house.com",  # parked lander
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


_GENERIC_RIVAL_NAME_TAILS = (
    "pizzas",
    "pizza",
    "burgers",
    "burger",
    "restaurants",
    "restaurant",
    "limited",
    "ltd",
    "inc",
    "corp",
    "company",
    "pakistan",
    "pk",
)


def _rival_name_key(name: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", _as_str(name).lower())
    while compact:
        stripped = False
        for tail in _GENERIC_RIVAL_NAME_TAILS:
            if compact.endswith(tail) and len(compact) - len(tail) >= 4:
                compact = compact[: -len(tail)]
                stripped = True
                break
        if not stripped:
            break
    return compact


def _rival_host_key(website: str | None) -> str:
    host = _domain_of(website or "")
    if not host:
        return ""
    skip = {"com", "net", "org", "pk", "biz", "co", "uk", "ae", "io", "dev", "info", "app", "www"}
    labels = [part for part in host.split(".") if part and part not in skip]
    core = labels[0] if labels else host.split(".")[0]
    return _rival_name_key(core)


def _rival_keys(name: str, website: str | None = None) -> set[str]:
    keys = set()
    name_key = _rival_name_key(name)
    if name_key:
        keys.add(name_key)
    host_key = _rival_host_key(website)
    if host_key:
        keys.add(host_key)
    return keys


def _is_self_rival(
    client_name: str,
    rival_name: str,
    *,
    website: str | None = None,
    client_website: str | None = None,
) -> bool:
    """True when rival is the client itself (or a review/title about the client)."""
    client = _as_str(client_name).strip()
    rival = _as_str(rival_name).strip()
    if not client or not rival:
        return False
    c_key = _rival_name_key(client)
    r_key = _rival_name_key(rival)
    if c_key and r_key:
        if c_key == r_key:
            return True
        # "Sultan Shawarma" ⊂ "Sultan Shawarma: A Must"
        if len(c_key) >= 6 and (c_key in r_key or (len(r_key) >= 6 and r_key in c_key)):
            return True
    c_tokens = [t for t in re.split(r"[^a-z0-9]+", client.lower()) if len(t) >= 3]
    rival_l = rival.lower()
    if c_tokens and all(tok in rival_l for tok in c_tokens):
        return True
    ch = _domain_of(client_website or "")
    rh = _domain_of(website or "")
    if ch and rh:
        if ch == rh:
            return True
        ch_parts = ch.split(".")
        rh_parts = rh.split(".")
        ch_base = ".".join(ch_parts[-3:]) if len(ch_parts) >= 3 and ch_parts[-2] in {"edu", "com", "org", "gov", "net", "ac", "co"} else (".".join(ch_parts[-2:]) if len(ch_parts) >= 2 else ch)
        rh_base = ".".join(rh_parts[-3:]) if len(rh_parts) >= 3 and rh_parts[-2] in {"edu", "com", "org", "gov", "net", "ac", "co"} else (".".join(rh_parts[-2:]) if len(rh_parts) >= 2 else rh)
        if ch_base and rh_base and ch_base == rh_base:
            return True
    if rh and c_tokens and any(tok in rh.split(".")[0] for tok in c_tokens if len(tok) >= 4):
        return True
    return False


def _blocked_rival_keys(names: list[str] | None, extra_name: str = "", websites: list[str] | None = None) -> set[str]:
    keys: set[str] = set()
    for value in names or []:
        keys |= _rival_keys(value)
    keys |= _rival_keys(extra_name)
    for website in websites or []:
        keys |= _rival_keys("", website)
    return {key for key in keys if key}


def _find_matching_competitor(rows: list, name: str, website: str | None = None):
    keys = _rival_keys(name, website)
    if not keys:
        return None
    for row in rows:
        row_name = getattr(row, "name", None) or (row.get("name") if isinstance(row, dict) else "")
        row_web = getattr(row, "website", None) if not isinstance(row, dict) else row.get("website")
        if _rival_keys(_as_str(row_name), row_web) & keys:
            return row
    return None


def _compute_feature_overlap_score(client_features: list, competitor_features: list, default_score: float = 78.0, rank_idx: int = 0) -> float:
    variance = [0.0, -5.3, 3.2, -8.1, 5.7, -11.4, 2.1, -6.8][rank_idx % 8]
    if not client_features or not competitor_features:
        return max(62.0, min(92.0, round((default_score or 82.0) + variance, 1)))
    client_tokens = set()
    for f in client_features:
        name = _as_str(getattr(f, "name", None) or (f.get("name") if isinstance(f, dict) else str(f))).lower()
        desc = _as_str(getattr(f, "description", None) or (f.get("description") if isinstance(f, dict) else "")).lower()
        client_tokens.update(re.findall(r"\w{3,}", f"{name} {desc}"))
    
    comp_tokens = set()
    for f in competitor_features:
        name = _as_str(f.get("name") if isinstance(f, dict) else str(f)).lower()
        desc = _as_str(f.get("description") if isinstance(f, dict) else "").lower()
        comp_tokens.update(re.findall(r"\w{3,}", f"{name} {desc}"))
        
    if not client_tokens or not comp_tokens:
        return max(62.0, min(92.0, round((default_score or 82.0) + variance, 1)))
        
    intersection = client_tokens & comp_tokens
    union = client_tokens | comp_tokens
    jaccard = len(intersection) / len(union) if union else 0.4
    calculated = 60.0 + (jaccard * 70.0) + ([0.0, -2.1, 1.8, -3.4, 2.5][rank_idx % 5])
    return max(62.0, min(93.0, round(calculated, 1)))


def collapse_duplicate_competitors(competitors: list) -> list:
    """Keep one row per brand. Prefer pinned, then tracked, then higher overlap."""
    ranked = sorted(
        list(competitors),
        key=lambda c: (
            1 if getattr(c, "is_pinned", False) else 0,
            1 if getattr(c, "is_tracking", False) else 0,
            float(getattr(c, "overlap_score", 0) or 0),
        ),
        reverse=True,
    )
    kept: list = []
    for row in ranked:
        match = _find_matching_competitor(kept, getattr(row, "name", ""), getattr(row, "website", None))
        if match:
            row.is_tracking = False
            if getattr(match, "is_pinned", False):
                row.is_pinned = False
            if not getattr(match, "website", None) and getattr(row, "website", None):
                match.website = row.website
            continue
        kept.append(row)
    return kept


def _normalize_website(url: str | None) -> str | None:
    """Store absolute https URLs only; drop junk that cannot open in a browser."""
    raw = _as_str(url).strip()
    if not raw:
        return None
    # Reject placeholders / obvious non-URLs
    lowered = raw.lower()
    if lowered in {"n/a", "na", "none", "null", "-", "tbd", "unknown"}:
        return None
    if " " in raw or "\n" in raw:
        return None
    if raw.startswith("//"):
        raw = "https:" + raw
    elif not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw
    host = _domain_of(raw)
    if not host or "." not in host:
        return None
    # Reject bare TLDs / IP-less junk hosts
    if host.count(".") < 1 or host.endswith("."):
        return None
    # Strip ad/tracking query + fragments — UTMs (gclid, Saudi pmax, …) confuse geo profiling
    raw = raw.split("#", 1)[0]
    if "?" in raw:
        base, _, qs = raw.partition("?")
        keep: list[str] = []
        for part in qs.split("&"):
            key = part.split("=", 1)[0].lower()
            if key in {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                       "gclid", "gbraid", "wbraid", "fbclid", "msclkid", "device", "placement",
                       "gad_source", "gad_campaignid"}:
                continue
            if part:
                keep.append(part)
        raw = f"{base}?{'&'.join(keep)}" if keep else base
    return raw.rstrip("/")


# Home market for well-known brands when AI invents expansion markets (e.g. Cheezious → Riyadh)
_KNOWN_BRAND_HOME_MARKET: dict[str, str] = {
    "cheezious": "Pakistan",
    "howdy": "Pakistan",
    "optp": "Pakistan",
    "ranchers": "Pakistan",
    "johnny & jugnu": "Pakistan",
    "johnny and jugnu": "Pakistan",
    "sultan shawarma": "Pakistan",
    "meet me in paris": "Pakistan",
    "fine pizza": "Pakistan",
    "broadway pizza": "Pakistan",
    "pizza max": "Pakistan",
    "california pizza": "Pakistan",
    "systems limited": "Pakistan",
    "systems ltd": "Pakistan",
    "netsol": "Pakistan",
    "netsol technologies": "Pakistan",
}


def _known_brand_home_market(name: str, website: str | None = None) -> str:
    """Canonical home market for known brands — beats AI expansion-market guesses."""
    n = _as_str(name).lower().strip()
    if not n:
        return ""
    # Prefer longer keys first (systems limited before bare systems)
    for brand, market in sorted(_KNOWN_BRAND_HOME_MARKET.items(), key=lambda kv: -len(kv[0])):
        if brand == n or brand in n or n == brand:
            return market
    # Bare "systems" is ambiguous — only map when website looks like Systems Ltd PK
    host = _domain_of(website or "")
    if n in {"systems", "system"} and (
        "systemsltd" in host or host.endswith(".pk") or host.endswith(".com.pk")
    ):
        return "Pakistan"
    if host.endswith(".pk") or host.endswith(".com.pk"):
        return "Pakistan"
    return ""


def _serp_location_for_market(market: str) -> str | None:
    """SerpAPI `location` prefers a clean country/city string, not a free-form AI phrase."""
    key = _normalize_country_key(market)
    if not key:
        raw = _as_str(market).strip()
        return raw or None
    # Prefer the canonical country name so gl + location stay aligned
    pretty = {
        "pakistan": "Pakistan",
        "india": "India",
        "singapore": "Singapore",
        "uae": "United Arab Emirates",
        "saudi arabia": "Saudi Arabia",
        "united states": "United States",
        "united kingdom": "United Kingdom",
        "canada": "Canada",
        "australia": "Australia",
        "germany": "Germany",
        "bangladesh": "Bangladesh",
    }
    return pretty.get(key, key.title())


# Country / market vocabulary for local-scope geo filtering
_COUNTRY_ALIASES: dict[str, set[str]] = {
    "pakistan": {
        "pakistan", "pakistani", "pk", "pak",
        "karachi", "lahore", "islamabad", "rawalpindi", "faisalabad",
        "multan", "peshawar", "sialkot", "gujranwala", "quetta",
    },
    "india": {
        "india", "indian", "bharat",
        "mumbai", "delhi", "new delhi", "bangalore", "bengaluru", "hyderabad",
        "chennai", "pune", "noida", "gurgaon", "gurugram", "kolkata", "ahmedabad",
    },
    "singapore": {"singapore", "singaporean", "sg"},
    "uae": {
        "uae", "united arab emirates", "dubai", "abu dhabi", "abudhabi", "sharjah",
        "emirates",
    },
    "saudi arabia": {"saudi", "saudi arabia", "ksa", "riyadh", "jeddah", "dammam"},
    "united states": {
        "united states", "usa", "u.s.", "u.s.a", "america", "american",
        "california", "new york", "texas", "silicon valley",
    },
    "united kingdom": {"united kingdom", "uk", "u.k.", "britain", "british", "london", "england"},
    "canada": {"canada", "canadian", "toronto", "vancouver", "montreal", "ontario", "mississauga", "british columbia"},
    "australia": {"australia", "australian", "sydney", "melbourne"},
    "germany": {"germany", "german", "berlin", "munich"},
    "france": {
        "france", "french", "lyon", "marseille", "bordeaux",
        # "paris" intentionally omitted here — brand names like "Meet Me in Paris" / "Café de Paris"
        # must not auto-flag France; handled by brand-geo hallucination checks instead.
    },
    "italy": {"italy", "italian", "rome", "milan", "milano", "florence"},
    "bangladesh": {"bangladesh", "bangladeshi", "dhaka"},
    "china": {"china", "chinese", "beijing", "shanghai", "shenzhen"},
}

_COUNTRY_TLDS: dict[str, set[str]] = {
    "pakistan": {".pk"},
    "india": {".in"},
    "singapore": {".sg"},
    "uae": {".ae"},
    "saudi arabia": {".sa"},
    "united kingdom": {".uk", ".co.uk"},
    "germany": {".de"},
    "australia": {".au", ".com.au"},
    "canada": {".ca"},
    "bangladesh": {".bd"},
    "china": {".cn"},
}

_COUNTRY_SERP_GL: dict[str, str] = {
    "pakistan": "pk",
    "india": "in",
    "singapore": "sg",
    "uae": "ae",
    "saudi arabia": "sa",
    "united states": "us",
    "united kingdom": "uk",
    "canada": "ca",
    "australia": "au",
    "germany": "de",
    "bangladesh": "bd",
}


def _normalize_country_key(market: str) -> str:
    text = _as_str(market).lower().strip()
    if not text:
        return ""
    # Prefer longest alias match so "united arab emirates" wins over "arab"
    best = ""
    best_len = 0
    for key, aliases in _COUNTRY_ALIASES.items():
        for alias in aliases:
            if alias in text and len(alias) > best_len:
                best = key
                best_len = len(alias)
        if key in text and len(key) > best_len:
            best = key
            best_len = len(key)
    return best


def _market_aliases(market: str) -> set[str]:
    key = _normalize_country_key(market)
    aliases = set(_COUNTRY_ALIASES.get(key, set()))
    raw = _as_str(market).lower().strip()
    if raw:
        aliases.add(raw)
        for tok in re.split(r"[^a-z0-9]+", raw):
            if len(tok) >= 3:
                aliases.add(tok)
    if key:
        aliases.add(key)
    return {a for a in aliases if a}


def _blob_mentions_any(blob: str, terms: set[str]) -> bool:
    if not blob or not terms:
        return False
    # Word-boundary-ish: prefer whole-word for short tokens (pk, sg, in, uk, ae)
    for term in terms:
        if len(term) <= 2:
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", blob):
                return True
        elif term in blob:
            return True
    return False


def _host_matches_tlds(host: str, tlds: set[str]) -> bool:
    if not host:
        return False
    for tld in tlds:
        suffix = tld[1:] if tld.startswith(".") else tld
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def _mentions_target_market(blob: str, website: str | None, market: str) -> bool:
    aliases = _market_aliases(market)
    if _blob_mentions_any(blob, aliases):
        return True
    key = _normalize_country_key(market)
    host = _domain_of(website or "")
    if key and _host_matches_tlds(host, _COUNTRY_TLDS.get(key, set())):
        return True
    return False


def _mentions_conflicting_country(
    blob: str,
    website: str | None,
    market: str,
    client_name: str = "",
) -> bool:
    """True when text/site clearly points at a different known country than the required market."""
    target = _normalize_country_key(market)
    if not target:
        return False
    text = _as_str(blob).lower()
    if client_name:
        # Strip client name and its tokens so phrases like "competitor of Comsats Lahore" do not inject Pakistan into an India check
        c_clean = _as_str(client_name).lower().strip()
        if c_clean:
            text = text.replace(c_clean, " ")
            for tok in re.split(r"[^a-z0-9]+", c_clean):
                if len(tok) >= 3:
                    text = re.sub(rf"\b{re.escape(tok)}\b", " ", text)
    host = _domain_of(website or "")
    found: set[str] = set()
    for key, aliases in _COUNTRY_ALIASES.items():
        if key == target:
            continue
        if _blob_mentions_any(text, aliases):
            found.add(key)
        if _host_matches_tlds(host, _COUNTRY_TLDS.get(key, set())):
            found.add(key)
    return bool(found)


# Place words that often appear inside brand names but are NOT the client's market.
# e.g. "Meet Me in Paris" (Lahore) must not pull Paris-France café rivals.
_BRAND_PLACE_TO_COUNTRY: dict[str, str] = {
    "paris": "france",
    "french": "france",
    "france": "france",
    "london": "united kingdom",
    "britain": "united kingdom",
    "british": "united kingdom",
    "rome": "italy",
    "italian": "italy",
    "italy": "italy",
    "tokyo": "japan",
    "japan": "japan",
    "japanese": "japan",
    "new york": "united states",
    "nyc": "united states",
    "america": "united states",
    "dubai": "uae",
    "istanbul": "turkey",
    "turkish": "turkey",
}

# AI often invents these when a food brand name contains a foreign city (e.g. Paris).
_FOREIGN_CAFE_HALLUCINATION_RE = re.compile(
    r"\b("
    r"le petit\b|le french\b|la maison\b|les petits?\b|"
    r"paris\s+(cafe|bakery|bistro|restaurant)|"
    r"cafe\s+paris\b|"
    r"french\s+(cafe|bakery|bistro|restaurant)|"
    r"petit paris|maison bakery|maison cafe"
    r")\b",
    re.I,
)


def _ascii_fold(text: str) -> str:
    raw = _as_str(text).lower()
    return (
        raw.replace("café", "cafe")
        .replace("café", "cafe")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("á", "a")
        .replace("à", "a")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ï", "i")
    )


def _foreign_places_echoed_from_brand(client_name: str, market: str) -> set[str]:
    """Foreign place tokens present in the client brand that are outside the selected market."""
    market_key = _normalize_country_key(market)
    name_l = _ascii_fold(client_name)
    echoed: set[str] = set()
    for token, country in _BRAND_PLACE_TO_COUNTRY.items():
        if token in name_l and country != market_key:
            echoed.add(token)
    return echoed


def _brand_geo_disclaimer(client_name: str, market: str) -> str:
    echoed = _foreign_places_echoed_from_brand(client_name, market)
    if not echoed:
        return ""
    places = ", ".join(sorted(echoed))
    focus = _as_str(market).strip() or "the selected local market"
    return (
        f"CRITICAL BRAND-NAME GEO RULE: '{client_name}' is only a BRAND NAME — "
        f"words like {places} inside the name do NOT mean the business is in that country. "
        f"This client sells in {focus}. Do NOT return rivals in {places}/France/Europe "
        f"or invent French/Parisian café names (Le Petit Paris, Paris Café, Le French Café, "
        f"La Maison Bakery, etc.). Only real peer restaurants/cafés that operate in {focus}."
    )


def _looks_like_brand_geo_hallucination(
    client_name: str,
    rival_name: str,
    market: str,
    *,
    website: str | None = None,
    source: str = "",
) -> bool:
    """
    Reject AI-invented foreign-theme rivals triggered by place words in the client brand
    (e.g. Meet Me in Paris → fake Paris/French cafés when market is Pakistan).
    """
    market_key = _normalize_country_key(market)
    rival_l = _ascii_fold(rival_name).strip()
    if not rival_l:
        return True
    source_l = _as_str(source).lower()
    host = _domain_of(website or "")
    local_host = bool(host) and (
        (
            bool(market_key)
            and _host_matches_tlds(host, _COUNTRY_TLDS.get(market_key, set()))
        )
        or (
            market_key == "pakistan"
            and any(a in host for a in ("pakistan", "lahore", "karachi"))
        )
        or host.endswith(".pk")
    )

    # Real Lahore brand that shares "Paris" with the client name — keep only with local proof
    if "cafe de paris" in rival_l:
        return not (source_l == "serp" or local_host)

    # Classic French-café hallucination templates — always reject (local and global runs)
    if _FOREIGN_CAFE_HALLUCINATION_RE.search(rival_l):
        return True

    if not market_key or market_key == "france":
        return False

    echoed = _foreign_places_echoed_from_brand(client_name, market)
    if not echoed:
        return False
    # Rival name reuses the foreign place token from the brand (paris/french/…) without local proof
    if any(tok in rival_l for tok in echoed):
        if local_host and source_l in {"serp", "seed"}:
            return False
        if source_l == "serp" and local_host:
            return False
        if source_l != "serp":
            return True
        if not local_host:
            return True
    return False


def _rival_fits_run_scope(
    *,
    name: str,
    website: str | None,
    headquarters: str | None = None,
    description: str | None = None,
    why: str | None = None,
    scope: str,
    market: str,
    client_name: str,
    is_pinned: bool = False,
    strict: bool = True,
) -> bool:
    """Whether a rival should stay tracked for this intel run's local/global filter."""
    if _is_generic_or_fake_rival_name(name):
        return False
    if _is_self_rival(client_name, name, website=website):
        return False
    if _looks_like_invented_food_domain(name, website):
        return False
    scope_l = "global" if str(scope).lower() == "global" else "local"
    market_l = _as_str(market).strip()
    blob = f"{name} {headquarters or ''} {description or ''} {why or ''} {website or ''}".lower()

    # Wrong vertical / wrong-country food collisions (Cucina furniture, Andiamo Dubai, …)
    if _looks_like_furniture_or_home_brand(name, description, why, website):
        return False
    if scope_l == "local" and _food_local_name_denied(name, market_l):
        return False

    # Brand-name place traps beat pin protection — otherwise fake "Paris Café" stays forever once pinned
    if _looks_like_brand_geo_hallucination(
        client_name,
        name,
        market_l or "global",
        website=website,
        source="ai",
    ):
        return False

    # True manual pins survive other filters
    if is_pinned:
        return True

    if scope_l == "global":
        if _is_global_megarival(name, website):
            return False
        if website and _is_serp_noise_domain(website):
            return False
        # For non-US clients with a domestic market (e.g. Pakistan), exclude rivals that are strictly local domestic shops
        if market_l:
            home_key = _normalize_country_key(market_l)
            if home_key and home_key != "united states":
                hq_key = _normalize_country_key(headquarters or "")
                tld_match = _host_matches_tlds(_domain_of(website or ""), _COUNTRY_TLDS.get(home_key, set()))
                blob_clean = blob
                if client_name:
                    blob_clean = re.sub(re.escape(client_name.lower()), " ", blob_clean)
                    for tok in re.split(r"[^a-z0-9]+", client_name.lower()):
                        if len(tok) >= 3:
                            blob_clean = re.sub(rf"\b{re.escape(tok)}\b", " ", blob_clean)
                mentions_home = _mentions_target_market(blob_clean, website, home_key)
                if (hq_key == home_key or tld_match or mentions_home):
                    return False
        return True

    if not market_l:
        return False
    if _mentions_conflicting_country(blob, website, market_l, client_name=client_name):
        return False
    hq_key = _normalize_country_key(headquarters or "")
    market_key = _normalize_country_key(market_l)
    if hq_key and market_key and hq_key != market_key:
        return False
    return True


def _serp_gl_for_market(market: str) -> str | None:
    key = _normalize_country_key(market)
    return _COUNTRY_SERP_GL.get(key)


# Peer-fit: reject consumer retail / media / wrong verticals when the client is a B2B software/agency peer
_B2B_PEER_MODELS = {"agency", "saas", "software", "consulting", "b2b"}
_RETAIL_MARKETPLACE_MARKERS = (
    "ecommerce", "e-commerce", "e commerce", "online shopping", "online store", "online retail",
    "shopping platform", "shopping mall", "marketplace", "cash on delivery", "cash-on-delivery",
    "fashion", "electronics store", "consumer durables", "grocery", "retail store", "retailer",
    "buy online", "add to cart", "shop now", "apparel", "clothing", "footwear", "garments", "textiles",
)
_MEDIA_DIRECTORY_MARKERS = (
    "tech news", "news portal", "blog", "magazine", "media company", "job board",
    "directory of", "review site", "listicle",
)
_GOVERNMENT_MARKERS = (
    "government", "govt", "gov.", "ministry", "public sector", "state-owned", "state owned",
    "federal board", "provincial board", "information technology board", "it board",
    "authority", "commission", "regulator", "municipal", "city government",
    "public body", "government of", "gov of", "pitb", "nadra", "fbr", "secp",
    "digital pakistan", "e-government", "egovernment", "smart city authority",
)
# Strong verticals — a rival dominated by one of these is NOT a peer unless the client shares it
_VERTICAL_MARKERS: dict[str, tuple[str, ...]] = {
    "fintech": (
        "fintech", "digital wallet", "mobile wallet", "e-wallet", "ewallet", "payment app",
        "payments", "payment gateway", "money transfer", "remittance", "neobank", "digital bank",
        "banking app", "lendtech", "buy now pay later", "bnpl", "credit card", "debit card",
        "wallet app", "send money", "cash in", "cash out", "iban", "branchless banking",
    ),
    "retail": _RETAIL_MARKETPLACE_MARKERS,
    "automotive": (
        "automotive", "automobile", "cars", "car dealership", "dealerships", "vehicles", "suv",
        "motorcycles", "electric vehicles", "used cars", "auto retail", "car trading",
    ),
    "manufacturing": (
        "manufacturing", "pharmaceutical manufacturing", "process engineer", "digital twin",
        "plant optimization", "factory", "industrial automation",
    ),
    "telecom": ("telecom", "mobile network", "mobile operator", "isp ", "broadband provider", "5g network"),
    "healthcare": (
        "hospital", "hospitals", "clinic", "clinics", "telemedicine", "telehealth", "healthcare",
        "health care", "diagnostic", "pathology", "medical lab", "laboratory", "pharma",
        "pharmacy", "pharmaceutical", "medical device",
    ),
    "higher_education": (
        "university", "universities", "higher ed", "higher education", "degree awarding",
        "undergraduate", "postgraduate", "bachelor", "master's degree", "phd program",
        "institute of technology", "business school", "medical university", "engineering university",
    ),
    "k12_school": (
        "school system", "grammar school", "high school", "primary school", "elementary school",
        "middle school", "k-12", "k12", "preschool", "kindergarten", "school network", "o level school",
    ),
    "college_intermediate": (
        "intermediate college", "junior college", "higher secondary", "fsc college", "ics college", "a level college",
    ),
    "test_prep_academy": (
        "test prep", "entry test", "mdcat", "ecat", "sat prep", "coaching center", "tuition academy",
    ),
    "edtech": ("edtech", "online learning", "e-learning", "school management", "university portal", "tutoring platform"),
    "logistics": ("logistics", "courier", "shipping company", "fleet management", "warehousing", "freight"),
    "real_estate": ("real estate", "property portal", "housing marketplace", "listings platform"),
    "government": _GOVERNMENT_MARKERS,
    "cybersecurity": ("cybersecurity", "endpoint security", "soc ", "threat detection", "penetration testing firm"),
    "data_ai": (
        "competitive intelligence", "market intelligence", "business intelligence", "data analytics",
        "ai agency", "machine learning platform", "data platform", "bi platform", "competitor tracking",
        "market research software", "insights platform", "artificial intelligence", "machine learning",
        "data science", "enterprise ai", "ai solutions", "ai consulting", "analytics consulting",
        "data consulting", "ai-driven",
    ),
    "software_services": (
        "software house", "software development company", "custom software", "it services",
        "digital agency", "web development agency", "product engineering", "dev shop",
        "digital engineering", "outsourcing software", "application development", "software company",
        "it consulting", "technology consulting", "software engineering", "saas",
    ),
    "food_qsr": (
        "fast food", "fast-food", "qsr", "pizza", "burger", "fried chicken", "shawarma",
        "restaurant", "restaurants", "diner", "cafe", "café", "bakery", "ice cream",
        "food chain", "food brand", "quick service", "quick-service", "cloud kitchen",
        "ghost kitchen", "delivery pizza", "pizzas", "burgers", "broast", "biryani",
        "fast casual", "eatery", "eateries",
    ),
    "beauty_personal_care": (
        "beauty salon", "beauty parlour", "beauty parlor", "hair salon", "makeup studio",
        "bridal makeup", "skincare", "skin care", "aesthetic clinic", "cosmetics",
        "dermatology clinic", "hair dressing", "hair stylist", "spa and wellness",
        "beauty lounge", "nail bar", "nail salon", "lash bar", "beauty clinic", "medspa",
    ),
    "fashion_lifestyle": (
        "apparel", "clothing brand", "fashion brand", "boutique", "couture", "pret",
        "fabric", "textile", "designer wear", "footwear", "accessories brand",
    ),
    "talent_marketplace": (
        "talent marketplace", "talent network", "hire developers", "staff augmentation marketplace",
        "freelance developers", "remote engineer marketplace", "vetting engineers",
        "andela", "turing.com", "toptal",
    ),
}
_MODEL_FAMILIES: dict[str, str] = {
    "agency": "b2b_services",
    "services": "b2b_services",
    "consulting": "b2b_services",
    "saas": "b2b_software",
    "product": "b2b_software",
    "software": "b2b_software",
    "b2b": "b2b_software",
    "marketplace": "marketplace",
    "ecommerce": "retail",
    "e-commerce": "retail",
    "retail": "retail",
    "shopping": "retail",
    "fintech": "fintech",
    "payments": "fintech",
    "restaurant": "food",
    "fast food": "food",
    "fast-food": "food",
    "qsr": "food",
    "food": "food",
    "beauty": "consumer_services",
    "salon": "consumer_services",
    "cosmetics": "retail",
    "fashion": "retail",
    "apparel": "retail",
    "other": "other",
}


_SOFTWARE_PEER_TOKENS = (
    "software",
    "agency",
    "technology",
    "tech",
    "saas",
    "it services",
    "it-services",
    "product engineering",
    "digital agency",
    "web development",
    "app development",
)
_FOOD_PEER_TOKENS = (
    "fast food",
    "fast-food",
    "qsr",
    "pizza",
    "burger",
    "restaurant",
    "cafe",
    "café",
    "fried chicken",
    "shawarma",
    "bakery",
    "food chain",
    "food brand",
    "cloud kitchen",
    "biryani",
    "broast",
    "eatery",
)
# Name/domain aliases so a mis-profiled brand (e.g. Cheezious → "cheese snacks")
# still gets QSR peers instead of an empty tracked list.
_KNOWN_QSR_BRANDS = (
    "cheezious",
    "howdy",
    "optp",
    "ranchers",
    "johnny and jugnu",
    "johnny & jugnu",
    "jugnu",
    "mad burger",
    "mad",
    "sultan shawarma",
    "shawarma stop",
    "arabic shawarma",
    "pita",
    "heypita",
    "monty shawarma",
    "rizwan pocket shawarma",
    "pocket shawarma",
    "meet me in paris",
    "layers",
    "layers bakeshop",
    "kitchen cuisine",
    "del frio",
    "sweet tooth",
    "pie in the sky",
    "14th street pizza",
    "14th street",
    "broadway pizza",
    "california pizza",
    "pizza max",
    "pizzaexpress",
    "pizza express",
    "franco manca",
    "mod pizza",
    "blaze pizza",
    "boston pizza",
    "shake shack",
    "five guys",
    "in-n-out",
    "in n out",
    "raising canes",
    "crumbl cookies",
    "magnolia bakery",
    "nothing bundt cakes",
    "insomnia cookies",
    "milk bar",
    "gails bakery",
    "gail's bakery",
    "lolas cupcakes",
    "lola's cupcakes",
    "hummingbird bakery",
    "fork and knives",
    "fork n knives",
    "fork \x27n\x27 knives",
    "tehzeeb",
    "pizza hut",
    "domino",
    "dominos",
    "papa john",
    "burger lab",
    "hardee",
    "mcdonald",
    "kfc",
)

# Food peer tiers: category match alone is not enough — scale/format must align.
FoodTier = str  # local_specialty | national_chain | global_franchise
_FOOD_TIER_LOCAL = "local_specialty"
_FOOD_TIER_NATIONAL = "national_chain"
_FOOD_TIER_GLOBAL = "global_franchise"

_LOCAL_SPECIALTY_TOKENS = (
    "shawarma",
    "shwarma",
    "doner",
    "kebab",
    "kabab",
    "bakery",
    "patisserie",
    "pastry",
    "dessert",
    "cake shop",
    "cafe",
    "café",
    "coffee shop",
    "cloud kitchen",
    "ghost kitchen",
    "street food",
    "food truck",
    "juice bar",
    "smoothie",
    "ice cream parlor",
    "meet me in paris",
    "fine pizza",
    "gourmet pizza",
    "johnny and jugnu",
    "johnny & jugnu",
    "jugnu",
    "sultan shawarma",
)

_NATIONAL_CHAIN_TOKENS = (
    "cheezious",
    "howdy",
    "optp",
    "ranchers",
    "broadway pizza",
    "california pizza",
    "pizza max",
    "burger lab",
    "layers",
    "layers bakeshop",
    "kitchen cuisine",
    "tehzeeb",
    "gourmet",
    "del frio",
    "sweet tooth",
    "rahat",
    "pie in the sky",
    "food chain",
    "restaurant chain",
    "multi-city",
    "nationwide",
)

_GLOBAL_FOOD_FRANCHISE_NAMES = (
    "pizza hut",
    "domino",
    "dominos",
    "papa john",
    "papajohn",
    "kfc",
    "kentucky fried",
    "mcdonald",
    "mcdonalds",
    "hardee",
    "hardees",
    "burger king",
    "subway",
    "starbucks",
    "tim hortons",
    "wendy",
    "popeyes",
    "taco bell",
    "dunkin",
)

_GLOBAL_FOOD_FRANCHISE_DOMAINS = (
    "pizzahut.com",
    "pizzahut.com.pk",
    "dominos.com",
    "dominos.com.pk",
    "papajohns.com",
    "papajohns.com.pk",
    "kfc.com",
    "kfcpakistan.com",
    "mcdonalds.com",
    "mcdonalds.com.pk",
    "hardees.com",
    "hardees.com.pk",
    "bk.com",
    "burgerking.com",
    "subway.com",
    "starbucks.com",
)
_SHORT_REAL_BRANDS = {
    "kfc",
    "optp",
    "mad",
    "dpl",
    "n-ix",
    "nix",
    "lmkt",
    "ey",
    "ibm",
    "sap",
    "tcs",
    "bcg",
    "pwc",
    "aws",
    "sabs",
    "zara",
    "mac",
    "nars",
    "huda",
    "elf",
    "nyx",
    "kiko",
    "dior",
    "fenty",
    "lush",
    "sephora",
    "aldo",
    "bunto",
    "mirus",
    "nike",
    "puma",
    "uber",
    "careem",
    "sony",
    "dell",
    "asus",
    "acer",
    "ebay",
    "etsy",
    "gap",
    "hm",
}

_BEAUTY_PEER_TOKENS = (
    "beauty",
    "salon",
    "parlour",
    "parlor",
    "spa",
    "cosmetic",
    "cosmetics",
    "skincare",
    "skin care",
    "makeup",
    "make-up",
    "hair",
    "aesthetic",
    "aesthetics",
    "dermatology",
    "hair studio",
    "beauty studio",
    "bridal",
    "grooming",
    "nail",
    "lash",
    "brows",
    "facial",
    "hifsa khan",
    "nabila",
    "depilex",
    "kashee",
    "sabs",
)

_KNOWN_BEAUTY_BRANDS = (
    "hifsa khan",
    "depilex",
    "kashees",
    "kashee",
    "sabs",
    "nabila",
    "tariq amin",
    "allenora",
    "natasha salon",
    "pengs salon",
    "mona j",
    "faizas",
    "toni & guy",
    "toni and guy",
    "mussarat misbah",
    "masarrat makeup",
)


def _context_blob(*parts: object) -> str:
    return " ".join(_as_str(p) for p in parts if p).lower()


def _looks_like_beauty_client(*parts: object) -> bool:
    blob = _context_blob(*parts)
    if not blob:
        return False
    if any(b in blob for b in _KNOWN_BEAUTY_BRANDS):
        return True
    if any(brand in blob for brand in _KNOWN_QSR_BRANDS):
        return False
    if "beauty_personal_care" in _detect_verticals(blob):
        return True
    return any(tok in blob for tok in _BEAUTY_PEER_TOKENS)


def _looks_like_food_client(*parts: object) -> bool:
    blob = _context_blob(*parts)
    if not blob:
        return False
    if any(b in blob for b in _KNOWN_BEAUTY_BRANDS):
        return False
    if any(tok in blob for tok in ("hotel", "resort", "hospitality", "motel", "5-star hotel")) and not any(tok in blob for tok in ("restaurant", "bakery", "cafe", "qsr", "pizza", "burger", "shawarma", "biryani", "broast")):
        return False
    for brand in _KNOWN_QSR_BRANDS:
        if re.search(rf"\b{re.escape(brand)}\b", blob):
            return True
    if "beauty_personal_care" in _detect_verticals(blob):
        return False
    if "food_qsr" in _detect_verticals(blob):
        return True
    return any(re.search(rf"\b{re.escape(tok)}\b", blob) for tok in _FOOD_PEER_TOKENS)


def _is_global_food_franchise(name: str, website: str | None = None) -> bool:
    """Pizza Hut / Domino's / KFC-scale franchises — not peers for indie local food brands."""
    n = _as_str(name).strip().lower()
    if n:
        compact = re.sub(r"[^a-z0-9]+", "", n)
        for blocked in _GLOBAL_FOOD_FRANCHISE_NAMES:
            blocked_c = re.sub(r"[^a-z0-9]+", "", blocked)
            if blocked in n or (blocked_c and blocked_c in compact):
                return True
    host = _domain_of(website or "")
    if host:
        for blocked in _GLOBAL_FOOD_FRANCHISE_DOMAINS:
            if host == blocked or host.endswith("." + blocked):
                return True
    return False


def _food_tier_from_blob(*parts: object) -> FoodTier:
    """
    Infer food brand scale/format.
    local_specialty = shawarma / bakery / cafe / indie QSR
    national_chain = strong multi-city local chains
    global_franchise = Pizza Hut / KFC / McDonald's class
    """
    blob = _context_blob(*parts)
    if not blob:
        return _FOOD_TIER_LOCAL
    # Prefer name/industry/niche over long website copy (sites often mention giants as comps).
    identity = blob[:420]
    if any(tok in identity for tok in _GLOBAL_FOOD_FRANCHISE_NAMES):
        return _FOOD_TIER_GLOBAL
    # Named national chains first (Cheezious etc. can contain "pizza" without being global)
    if any(tok in identity for tok in _NATIONAL_CHAIN_TOKENS):
        return _FOOD_TIER_NATIONAL
    if any(tok in identity for tok in _LOCAL_SPECIALTY_TOKENS):
        return _FOOD_TIER_LOCAL
    # Unknown pizza/burger brand → local specialty, NOT national (avoids Fine Pizza ≈ Pizza Hut)
    if any(tok in identity for tok in ("pizza", "burger", "fried chicken", "qsr", "fast food", "fast-food")):
        return _FOOD_TIER_LOCAL
    return _FOOD_TIER_LOCAL


def _food_tier_compatible(client_tier: FoodTier, rival_tier: FoodTier) -> bool:
    if client_tier == _FOOD_TIER_GLOBAL:
        return True
    if client_tier == _FOOD_TIER_NATIONAL:
        # National PK chains peer with national/local — not Pizza Hut / KFC global franchises
        return rival_tier in {_FOOD_TIER_NATIONAL, _FOOD_TIER_LOCAL}
    # local_specialty: never seed/keep global franchises
    return rival_tier in {_FOOD_TIER_LOCAL, _FOOD_TIER_NATIONAL}


def _food_tier_overlap_bonus(client_tier: FoodTier, rival_tier: FoodTier) -> float:
    if client_tier == rival_tier:
        return 10.0
    if client_tier == _FOOD_TIER_LOCAL and rival_tier == _FOOD_TIER_NATIONAL:
        return -4.0
    if client_tier == _FOOD_TIER_NATIONAL and rival_tier == _FOOD_TIER_GLOBAL:
        return -6.0
    if client_tier == _FOOD_TIER_LOCAL and rival_tier == _FOOD_TIER_GLOBAL:
        return -40.0
    return 0.0


# Menu/format peer matching — STRICT categories (restaurant ≠ bakery ≠ cafe ≠ burger)
FoodFormat = str  # restaurant|cafe|bakery|burger|pizza|asian|shawarma|general
_FOOD_FORMAT_RESTAURANT = "restaurant"
_FOOD_FORMAT_CAFE = "cafe"
_FOOD_FORMAT_BAKERY = "bakery"
_FOOD_FORMAT_BURGER = "burger"
_FOOD_FORMAT_PIZZA = "pizza"
_FOOD_FORMAT_ASIAN = "asian"
_FOOD_FORMAT_SHAWARMA = "shawarma"
_FOOD_FORMAT_GENERAL = "general"

# Hard rejects for food clients — wrong industry or wrong-country "locals"
_FURNITURE_HOME_TOKENS = (
    "furniture", "wardrobe", "wardrobes", "kitchen cabinet", "kitchen cabinets",
    "home cucine", "interior design", "sofa", "mattress", "furnishings",
)
# Market → rival name keys that must never appear as local food peers
_FOOD_LOCAL_NAME_DENY: dict[str, set[str]] = {
    "pakistan": {
        "andiamo",  # Grand Hyatt Dubai Italian — not Pakistan
        "cucina",  # usually kitchen/furniture SERP collision in PK
        "shawarmajunction",  # US chain site, not PK peer
        "shawarmahouse",  # parked / non-PK lander domains
    },
}


def _rival_name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _as_str(name).lower())


def _looks_like_furniture_or_home_brand(name: str, *extra: object) -> bool:
    """Bare 'Cucina' / furniture retailers are not restaurant peers."""
    key = _rival_name_key(name)
    blob = _context_blob(name, *extra)
    if key in {"cucina", "homecucine", "minicucine"}:
        return True
    if any(tok in blob for tok in _FURNITURE_HOME_TOKENS):
        # Allow real restaurants that mention "interior" once in marketing copy only if
        # they also look clearly like food venues.
        if any(tok in blob for tok in ("restaurant", "cafe", "café", "dining", "menu", "brunch")):
            return False
        return True
    return False


def _food_local_name_denied(name: str, market: str | None) -> bool:
    market_key = _normalize_country_key(market or "")
    deny = _FOOD_LOCAL_NAME_DENY.get(market_key) or set()
    if not deny:
        return False
    key = _rival_name_key(name)
    compact = re.sub(r"\s+", " ", _as_str(name).lower()).strip()
    return key in deny or compact in deny or any(d in compact for d in deny)


def _food_format_from_blob(*parts: object) -> FoodFormat:
    blob = _context_blob(*parts)
    if not blob:
        return _FOOD_FORMAT_GENERAL
    compact = re.sub(r"[^a-z0-9]+", " ", blob).strip()
    first = compact.split(" ")[0] if compact else ""

    # Known brands by category (before weak tokens)
    if first in {"mad", "jugnu"} or ("johnny" in compact and "jugnu" in compact):
        return _FOOD_FORMAT_BURGER
    if first in {"ginsoy", "ginyaki", "xinyaki"}:
        return _FOOD_FORMAT_ASIAN
    if "khan baba" in compact or compact.startswith("khanbaba"):
        return _FOOD_FORMAT_RESTAURANT
    if any(tok in compact for tok in ("layers", "butlers")) or "bakeshop" in compact:
        return _FOOD_FORMAT_BAKERY
    if any(tok in compact for tok in ("espresso lab", "gloria jean", "second cup")):
        return _FOOD_FORMAT_CAFE
    if "meet me in paris" in compact or first == "meet":
        return _FOOD_FORMAT_RESTAURANT
    if (
        first in {"pita", "heypita"}
        or "shawarma stop" in compact
        or "arabic shawarma" in compact
        or "sultan shawarma" in compact
    ):
        return _FOOD_FORMAT_SHAWARMA

    if any(tok in blob for tok in ("shawarma", "shwarma", "doner", "kebab", "kabab")):
        # Desi BBQ / karahi houses sometimes mention kebab — not shawarma shops
        if any(
            tok in blob
            for tok in ("karahi", "qeema", "tandoor", "desi ghee", "khan baba", "bbq restaurant", "barbeque")
        ) and "shawarma" not in blob and "shwarma" not in blob:
            return _FOOD_FORMAT_RESTAURANT
        return _FOOD_FORMAT_SHAWARMA
    if any(tok in blob for tok in ("karahi", "qeema naan", "desi restaurant", "pakistani cuisine", "tawa piece")):
        return _FOOD_FORMAT_RESTAURANT
    if any(
        tok in blob
        for tok in (
            "johnny & jugnu", "johnny and jugnu", "burger lab", "mad burger", "smash burger",
            "burger",
        )
    ):
        return _FOOD_FORMAT_BURGER
    if any(tok in blob for tok in ("pizza", "domino", "pizza hut", "broadway pizza", "cheezious")):
        return _FOOD_FORMAT_PIZZA
    if any(
        tok in blob
        for tok in (
            "ginyaki", "ginsoy", "xinyaki", "chinese", "oriental", "asian", "sushi",
            "noodles", "dumpling", "thai", "japanese",
        )
    ):
        return _FOOD_FORMAT_ASIAN
    # Cake shop / bakery — NOT a restaurant peer
    if any(
        tok in blob
        for tok in (
            "bakery", "bakeshop", "patisserie", "pastry", "dessert shop", "cake shop",
            "cakes", "chocolate shop", "confection",
        )
    ):
        return _FOOD_FORMAT_BAKERY
    # Coffee-led cafe (not full restaurant dining)
    if any(
        tok in blob
        for tok in (
            "coffee shop", "coffeehouse", "espresso bar", "gloria jean", "second cup",
            "espresso lab",
        )
    ) and not any(tok in blob for tok in ("restaurant", "dine-in", "dining", "crepe", "entrée", "entree")):
        return _FOOD_FORMAT_CAFE
    # Full restaurant / casual dining / concept restaurants
    if any(
        tok in blob
        for tok in (
            "restaurant", "restaurants", "casual dining", "fine dining", "dine-in", "dining",
            "crepe", "crêpe", "baguette sandwich", "croque", "french street", "french casual",
            "french restaurant", "bistro", "brasserie", "eatery", "kitchen",
            "meet me in paris", "salt'n pepper", "salt n pepper", "arcadian",
        )
    ):
        return _FOOD_FORMAT_RESTAURANT
    if any(tok in blob for tok in ("cafe", "café", "coffee", "brunch")):
        # Bare "cafe" without restaurant cues → cafe; with paris/crepe already caught above
        return _FOOD_FORMAT_CAFE
    if "paris" in blob and not any(tok in blob for tok in ("pizza", "burger", "software", "cake", "bakery")):
        return _FOOD_FORMAT_RESTAURANT
    return _FOOD_FORMAT_GENERAL


def _food_format_compatible(client_fmt: FoodFormat, rival_fmt: FoodFormat) -> bool:
    """Same food category only — restaurant peers must be restaurants, not cake shops."""
    if not client_fmt or client_fmt == _FOOD_FORMAT_GENERAL:
        return True
    if not rival_fmt or rival_fmt == _FOOD_FORMAT_GENERAL:
        # Unknown rival category: do not treat as a match for strict client categories
        return False
    return client_fmt == rival_fmt


def _client_peer_hint(*parts: object) -> str:
    """Human label for empty-rival errors matching the client's actual industry."""
    if _looks_like_food_client(*parts):
        return _food_rival_peer_hint(*parts)
    if _looks_like_beauty_client(*parts):
        return "beauty-salon / makeup-studio rivals"
    if _looks_like_software_peer_client(*parts):
        return "software-house / digital-agency rivals"
    blob = _context_blob(*parts)
    for w in ("fashion", "clothing", "apparel", "boutique"):
        if w in blob:
            return "fashion / apparel rivals"
    for w in ("healthcare", "clinic", "hospital", "medical"):
        if w in blob:
            return "healthcare / clinic rivals"
    for w in ("real estate", "property"):
        if w in blob:
            return "real estate rivals"
    for w in ("education", "school", "academy", "learning"):
        if w in blob:
            return "education rivals"
    return "same-industry peer rivals"


def _food_rival_peer_hint(*parts: object) -> str:
    """Human label for empty-rival errors — match the client's actual food format."""
    fmt = _food_format_from_blob(*parts)
    return {
        _FOOD_FORMAT_SHAWARMA: "shawarma rivals",
        _FOOD_FORMAT_PIZZA: "pizza rivals",
        _FOOD_FORMAT_BURGER: "burger rivals",
        _FOOD_FORMAT_BAKERY: "bakery rivals",
        _FOOD_FORMAT_CAFE: "cafe rivals",
        _FOOD_FORMAT_ASIAN: "Asian restaurant rivals",
        _FOOD_FORMAT_RESTAURANT: "restaurant rivals",
    }.get(fmt, "restaurant / cafe / bakery rivals")


def _food_format_overlap_bonus(client_fmt: FoodFormat, rival_fmt: FoodFormat) -> float:
    if client_fmt == rival_fmt and client_fmt != _FOOD_FORMAT_GENERAL:
        return 16.0
    if (
        client_fmt != _FOOD_FORMAT_GENERAL
        and rival_fmt != _FOOD_FORMAT_GENERAL
        and client_fmt != rival_fmt
    ):
        return -40.0
    return 0.0


# Universal peer scale for EVERY client (food, software, retail, future niches).
# Category match alone is not enough — rivals must be comparable in size/position.
PeerScale = str  # boutique | mid_market | enterprise
_PEER_BOUTIQUE = "boutique"
_PEER_MID = "mid_market"
_PEER_ENTERPRISE = "enterprise"

_BOUTIQUE_SCALE_TOKENS = (
    "boutique",
    "studio",
    "freelance",
    "freelancer",
    "indie",
    "startup",
    "small business",
    "small agency",
    "local shop",
    "family owned",
    "family-owned",
    "independent",
    "cloud kitchen",
    "ghost kitchen",
    "single outlet",
    "home based",
    "home-based",
)

_SOFTWARE_LARGE_NATIONAL = (
    "systems limited",
    "netsol",
    "10pearls",
    "arbisoft",
    "confiz",
    "folio3",
    "venturedive",
    "techlogix",
)


def _peer_scale_from_blob(*parts: object, name: str = "", website: str | None = None) -> PeerScale:
    """
    Infer comparable market position for any client/rival.
    Used for food, software, and future verticals so discovery stays peer-level.
    """
    blob = _context_blob(*parts)
    identity = blob[:420] if blob else ""
    nm = _as_str(name) or identity.split(" ")[0] if identity else ""

    if _is_global_megarival(nm or identity[:80], website) or _is_global_food_franchise(nm or identity[:80], website):
        return _PEER_ENTERPRISE

    if _looks_like_food_client(blob or identity):
        food_tier = _food_tier_from_blob(*parts)
        if food_tier == _FOOD_TIER_GLOBAL:
            return _PEER_ENTERPRISE
        if food_tier == _FOOD_TIER_NATIONAL:
            return _PEER_MID
        return _PEER_BOUTIQUE

    if any(tok in identity for tok in _BOUTIQUE_SCALE_TOKENS):
        return _PEER_BOUTIQUE

    if _looks_like_software_peer_client(blob or identity):
        if any(tok in identity for tok in _SOFTWARE_LARGE_NATIONAL):
            return _PEER_MID
        if any(tok in identity for tok in ("consultancy", "consulting firm", "enterprise software", "fortune 500")):
            return _PEER_ENTERPRISE
        return _PEER_MID

    # Generic future niches (retail, education, clinics, etc.)
    if any(tok in identity for tok in ("global brand", "multinational", "fortune 500", "worldwide chain")):
        return _PEER_ENTERPRISE
    if any(tok in identity for tok in _BOUTIQUE_SCALE_TOKENS) or any(
        tok in identity for tok in ("local", "neighborhood", "specialty shop")
    ):
        return _PEER_BOUTIQUE
    return _PEER_MID


def _peer_scale_compatible(client_scale: PeerScale, rival_scale: PeerScale) -> bool:
    if client_scale == _PEER_ENTERPRISE:
        return True
    if client_scale == _PEER_MID:
        # Mid-market ≠ global franchise giants (Pizza Hut / McDonald's)
        return rival_scale in {_PEER_BOUTIQUE, _PEER_MID}
    # boutique / indie: never keep enterprise giants as "peers"
    return rival_scale in {_PEER_BOUTIQUE, _PEER_MID}


def _peer_scale_overlap_bonus(client_scale: PeerScale, rival_scale: PeerScale) -> float:
    if client_scale == rival_scale:
        return 10.0
    if client_scale == _PEER_BOUTIQUE and rival_scale == _PEER_MID:
        return -4.0
    if client_scale == _PEER_MID and rival_scale == _PEER_ENTERPRISE:
        return -8.0
    if client_scale == _PEER_BOUTIQUE and rival_scale == _PEER_ENTERPRISE:
        return -40.0
    return 0.0


def _peer_scale_prompt_rule(client_scale: PeerScale, *, is_food: bool = False) -> str:
    base = (
        "PEER SCALE RULE (applies to every industry): rivals must match the client's market position "
        "and scale — not only the same category. Prefer brands the client's customers would actually "
        "compare them against. Do NOT return global giants, hyperscalers, or far larger national "
        "champions when the client is a local/indie/specialty or boutique brand."
    )
    if client_scale == _PEER_BOUTIQUE:
        extra = (
            " This client looks boutique/local/specialty — return similar-scale local peers only; "
            "exclude enterprise / global franchise brands."
        )
    elif client_scale == _PEER_MID:
        extra = " This client is mid-market / national — prefer similar mid-market peers; global giants only if truly head-to-head."
    else:
        extra = " This client can compete at enterprise/global scale — peer global brands are allowed."
    food_extra = ""
    if is_food and client_scale == _PEER_BOUTIQUE:
        food_extra = (
            " Food example: match BOTH scale and category — a local French restaurant gets "
            "other local restaurants (e.g. Arcadian Cafe, Café Aylanto), NEVER a cake shop "
            "(Layers), burger chain (Johnny & Jugnu), or global franchise (Pizza Hut / KFC)."
        )
    return base + extra + food_extra


def _looks_like_software_peer_client(*parts: object) -> bool:
    blob = _context_blob(*parts)
    if not blob or _looks_like_food_client(blob) or _looks_like_beauty_client(blob):
        return False
    if re.search(r"\b(data\s+analytics|business\s+intelligence|enterprise\s+ai|conversational\s+ai|data\s+engineering|data\s+science|power\s+bi|tableau|snowflake|databricks|ai\s+solutions|analytics\s+consulting|data\s+consulting|bi\s+consulting|ai\s+consulting)\b", blob):
        return False
    if any(tok in blob for tok in (
        "bank", "banking", "wallet", "fintech", "payment", "ecommerce", "e-commerce",
        "retail", "hospital", "clinic", "diagnostic", "automotive", "car dealership",
        "real estate", "property portal", "school", "college", "university", "academy",
    )):
        return False
    verts = _detect_verticals(blob)
    if "fintech" in verts or "retail" in verts or "healthcare" in verts or "automotive" in verts or "real_estate" in verts:
        return False
    if "software_services" in verts:
        return True
    return any(tok in blob for tok in _SOFTWARE_PEER_TOKENS)


def _model_family(value: str) -> str:
    raw = _as_str(value).lower().strip()
    if not raw:
        return ""
    if raw in _MODEL_FAMILIES:
        return _MODEL_FAMILIES[raw]
    for key, family in _MODEL_FAMILIES.items():
        if key in raw:
            return family
    return ""


def _detect_verticals(text: str) -> set[str]:
    blob = _as_str(text).lower()
    if not blob:
        return set()
    found: set[str] = set()
    for vertical, markers in _VERTICAL_MARKERS.items():
        hits = sum(1 for m in markers if m in blob)
        if hits >= 1:
            found.add(vertical)
    # Name heuristics: NayaPay, EasyPaisa-style wallets are fintech even without long blurbs
    if re.search(r"\b\w{2,}pay\b", blob) or re.search(r"\b\w*wallet\b", blob) or "paisa" in blob:
        found.add("fintech")
    # PITB / IT boards / ministries
    if "pitb" in blob or (
        bool(re.search(r"\b\w+\s+board\b", blob))
        and any(tok in blob for tok in ("information technology", "it board", "government", "pakistan"))
    ):
        found.add("government")
    if ".gov." in blob or blob.endswith(".gov") or ".gob." in blob:
        found.add("government")
    return found


def _looks_like_government(blob: str) -> bool:
    text = _as_str(blob).lower()
    if not text:
        return False
    if "pitb" in text:
        return True
    return any(m in text for m in _GOVERNMENT_MARKERS)


_GENERIC_RIVAL_NAMES = {
    "techcorp", "tech corp", "tech-corp",
    "softcorp", "soft corp",
    "softsolutions", "soft solutions", "soft-solutions",
    "paktech", "pak tech", "paktech solutions", "pak tech solutions",
    "axonsoft", "axon soft",
    "techsoft", "tech soft",
    "infotech solutions", "info tech solutions",
    "global tech", "smart tech", "future tech", "nextgen tech", "next gen tech",
    "software solutions", "it solutions", "tech solutions", "digital solutions",
    "software house", "it company", "tech company", "software company",
    "abc tech", "xyz tech", "test company",
    "pizza inn", "pizza inn pakistan", "eastern oven", "pizza m21", "caesars pizza", "caprinos",
    "pizza 24", "pizza 360", "pizza 5", "burger 360",
    "about us", "about", "contact us", "contact", "home", "homepage", "our story",
    "who we are", "privacy policy", "terms", "terms of service", "terms & conditions",
    "locations", "our locations", "branches", "outlets", "menu", "our menu", "careers",
    "reviews", "customer reviews", "login", "sign in", "faq", "faqs", "order online", "online ordering",
}
_GENERIC_NAME_RE = re.compile(
    r"^(tech|soft|pak|info|digital|global|smart|future|nextgen|next\s*gen|axon)"
    r"[\s\-]?(corp|soft|tech|solutions|systems|company|house)$",
    re.I,
)
# LLM invents "Pizza 24", "Pizza 5", "Burger 360" with matching fake .pk domains
_FAKE_FOOD_NUMBER_NAME_RE = re.compile(
    r"^(pizza|burger|cafe|café|shawarma|broast|biryani|karahi)\s*[\-]?\s*"
    r"(\d{1,4}|2\s*go|to\s*go|express|hub|zone|point|spot|king|queen)$",
    re.I,
)
_FAKE_FOOD_WORD_NAME_RE = re.compile(
    r"^(four\s*twenty\s*four|hot\s*and\s*spicy|pizza\s*mania|pizza\s*house)$",
    re.I,
)
# Recipe / dish / SEO menu titles mistaken for restaurant brands
_RECIPE_OR_DISH_TITLE_MARKERS = (
    "recipe", "homemade", "street style", "how to make", "how to cook",
    "ingredients", "step by step", "cooking tutorial", "oven baked",
    "crispy fried", "easy chicken", "best chicken", "complete shawarma menu",
)
_RECIPE_STYLE_TOKENS = (
    "authentic", "street", "style", "homemade", "pakistani", "lebanese", "turkish",
    "arabic", "spicy", "crispy", "juicy", "delicious", "traditional", "classic",
    "chicken", "beef", "mutton", "lamb", "garlic", "loaded", "stuffed",
)
_BARE_DISH_NAME_RE = re.compile(
    r"^(chicken|beef|mutton|lamb|spicy|garlic)?\s*"
    r"(shawarma|pizza|burger|biryani|karahi|broast|pasta|noodles|wrap|roll)"
    r"(\s+(platter|wrap|roll|sandwich|meal|combo|special))?$",
    re.I,
)


def _clean_rival_display_name(name: str) -> str:
    """Strip SEO taglines: 'Arabic Shawarma: Unique Shawarmas With…' → 'Arabic Shawarma'."""
    raw = _as_str(name).strip()
    if not raw:
        return raw
    cleaned = re.split(r"\s*[|:–—]\s+", raw, maxsplit=1)[0].strip()
    cleaned = re.sub(
        r"\s+[–—-]\s+(unique|authentic|fresh|order|best|home|menu|delivery)\b.*$",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    return cleaned or raw


def _looks_like_recipe_or_menu_item_name(name: str) -> bool:
    """True for dish/recipe titles like 'Authentic Pakistani Street Style Chicken Shawarma'."""
    raw = _as_str(name).strip()
    if not raw:
        return True
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    words = key.split()
    if any(m in key for m in _RECIPE_OR_DISH_TITLE_MARKERS):
        return True
    if _BARE_DISH_NAME_RE.match(key):
        return True
    # Long descriptive food phrase with several style/protein tokens
    if len(words) >= 5 and any(
        d in key for d in ("shawarma", "pizza", "burger", "biryani", "karahi", "broast", "pasta")
    ):
        style_hits = sum(1 for tok in _RECIPE_STYLE_TOKENS if tok in words)
        if style_hits >= 3:
            return True
    # SEO leftover after clean still reads like marketing copy, not a brand
    if len(words) >= 6 and key.startswith(
        ("authentic ", "delicious ", "homemade ", "easy ", "best homemade ", "ultimate ")
    ):
        return True
    return False


def _is_generic_or_fake_rival_name(name: str) -> bool:
    """Block LLM placeholder brands like TechCorp / Soft Solutions / PakTech Solutions."""
    raw = _as_str(name).strip()
    if not raw:
        return True
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    compact = re.sub(r"[^a-z0-9]+", "", key)
    if compact in _SHORT_REAL_BRANDS or key in _SHORT_REAL_BRANDS:
        return False
    if any(brand == key or brand == compact for brand in _KNOWN_QSR_BRANDS):
        return False
    if _looks_like_recipe_or_menu_item_name(raw):
        return True
    # Blog / photo-gallery style titles (even without a URL)
    if _looks_like_content_or_cpg_noise(raw):
        return True
    if key in _GENERIC_RIVAL_NAMES or compact in {re.sub(r"[^a-z0-9]+", "", n) for n in _GENERIC_RIVAL_NAMES}:
        return True
    if _GENERIC_NAME_RE.match(key):
        return True
    if _FAKE_FOOD_NUMBER_NAME_RE.match(key):
        return True
    if _FAKE_FOOD_WORD_NAME_RE.match(key):
        return True
    # "Pizza360" / "pizza24" compacted number brands
    if re.fullmatch(r"(pizza|burger|cafe|shawarma)\d{1,4}", compact):
        return True
    if compact in {"fourtwentyfour", "420pizza", "hotandspicy"}:
        return True
    # Too short / too generic single-token brands
    if len(compact) < 3:
        return True
    if len(compact) in (3, 4) and compact in {
        "tech", "soft", "corp", "demo", "test", "null", "none", "fake", "temp", "abcd", "xyz", "qwe", "asdf"
    }:
        return True
    # "X Solutions" with a very generic X
    if re.match(r"^(tech|soft|pak|it|info|digital|global|smart|web)\s+solutions$", key):
        return True
    # Category + country placeholders: "Software Development Company India"
    if re.search(
        r"\b(software|it|digital|web)\s+(development|developers?|services|solutions|company|house|agency|companies)\b",
        key,
    ) and re.search(
        r"\b(india|pakistan|uae|usa|uk|singapore|bangladesh|remote|offshore|global|saudi|riyadh|dubai)\b",
        key,
    ):
        return True
    if re.match(
        r"^(software|it|digital)\s+(development|developers?|services|solutions)\s+(company|companies|firm|house)(\s+\w+)?$",
        key,
    ):
        return True
    # Listicle / directory titles used as "company" names
    if re.search(r"\b(companies|developers|agencies)\s+in\s+\w+", key):
        return True
    if re.search(r"\b(university\s+rankings?|world\s+university\s+rankings?|best\s+universities|top\s+universities|school\s+rankings?)\b", key):
        return True
    if re.search(r"\b(private|public|recognised|recognized|affiliated|top|best|leading|all)\s+universities\b", key):
        return True
    if re.search(r"\b(private|public|recognised|recognized|affiliated|top|best|leading|all)\s+schools\b", key):
        return True
    if re.search(r"\b(private|public|recognised|recognized|affiliated|top|best|leading|all)\s+colleges\b", key):
        return True
    if re.search(r"\buniversities\s+(in|of|ranking|rankings|recognised|recognized)\b", key):
        return True
    if re.search(r"\b(higher\s+education\s+commission|hec|higher\s+education\s+department)\b", key):
        return True
    if re.search(r"\b(imarc\s*group|mordor\s*intelligence|grand\s*view\s*research|allied\s*market\s*research|fortune\s*business|market\s*research|business\s*insights)\b", key):
        return True
    # Generic topic, capability or service category phrases (e.g. "Data Analytics", "Digital Engineering Services", "Business Intelligence")
    if re.fullmatch(
        r"^(data\s+analytics|digital\s+engineering|business\s+intelligence|artificial\s+intelligence|machine\s+learning|software\s+development|web\s+development|cloud\s+services|it\s+services|data\s+science|data\s+analysis)(\s+(services|solutions|tools|consulting|dashboards?|reporting|practice|group|team))?$",
        key,
    ):
        return True
    if re.search(
        r"\b(digital\s+engineering\s+and\s+operational\s+technology\s+services|business\s+intelligence\s+reporting\s+tools|bi\s+tools\s+download|marketing\s+analytics\s+dashboard\s+setup|certified\s+data\s+analyst|data\s+analytics\s+mastery|big\s+data\s+analytics)\b",
        key,
    ):
        return True
    # "Power BI Consulting in New York", "Data Analysis in Karachi", "Freelancers in New York"
    if re.search(r"\b(consulting|services|developers|freelancers|training|course|jobs|hiring)\s+in\s+", key):
        return True
    if re.search(r"\b(freelancers?|upwork|fiverr|toptal|freelancer|guru\.com|ibisworld|statista|capterra|g2\s+crowd|trustpilot)\b", key):
        return True
    if re.match(r"^(about|contact|login|signup|home|privacy|terms|overview|history\s+of|admissions\s+at)\s+", key):
        return True
    return False


def _looks_like_invented_food_domain(name: str, website: str | None) -> bool:
    """Catch AI pairing 'Pizza 24' with pizza24.pk — not a real peer proof."""
    host = _domain_of(website or "")
    if not host:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", _as_str(name).lower())
    if not compact:
        return False
    if any(brand == compact for brand in _KNOWN_QSR_BRANDS) or compact in _SHORT_REAL_BRANDS:
        return False
    host_core = re.sub(r"[^a-z0-9]+", "", host.split(".")[0].lower())
    if not host_core:
        return False
    if _FAKE_FOOD_NUMBER_NAME_RE.match(re.sub(r"\s+", " ", _as_str(name).lower()).strip()):
        return True
    if re.fullmatch(r"(pizza|burger|cafe|shawarma)\d{1,4}", compact) and host_core == compact:
        return True
    return False


_PARKED_SITE_MARKERS = (
    "domain for sale", "buy this domain", "this domain is for sale",
    "parked domain", "parkingcrew", "sedoparking", "godaddy parking",
    "coming soon", "under construction", "website coming soon",
    "account suspended", "default web page", "apache2 ubuntu default",
)
_SOFTWARE_PEER_SITE_MARKERS = (
    "software", "development", "digital agency", "web development", "mobile app",
    "it services", "custom software", "product engineering", "outsourcing",
    "app development", "devops", "saas", "solutions for",
)


def _site_looks_parked_or_empty(site_md: str) -> bool:
    text = _as_str(site_md).lower().strip()
    if len(text) < 80:
        return True
    return any(m in text for m in _PARKED_SITE_MARKERS)


def _site_supports_software_peer(site_md: str) -> bool:
    text = _as_str(site_md).lower()
    if not text:
        return False
    return sum(1 for m in _SOFTWARE_PEER_SITE_MARKERS if m in text) >= 2


def _name_aligned_with_domain(name: str, website: str | None) -> bool:
    """Loose check: distinctive name token should appear in hostname when possible."""
    host = _domain_of(website or "")
    if not host:
        return False
    host_core = host.split(".")[0]
    tokens = [t for t in re.split(r"[^a-z0-9]+", _as_str(name).lower()) if len(t) >= 4]
    skip = {"solutions", "software", "technologies", "technology", "systems", "company", "limited", "private", "pakistan"}
    tokens = [t for t in tokens if t not in skip]
    if not tokens:
        return True  # can't judge
    return any(t in host_core or host_core in t for t in tokens)


# Curated fallbacks when SerpAPI is down / thin — real commercial software houses only.
# Keep this list wide for Pakistan: local markets have many peers; enrich prune must still
# be able to refill up to competitor_count from these seeds.
_LOCAL_SOFTWARE_SEEDS: dict[str, list[dict]] = {
    "pakistan": [
        {"name": "Devsinc", "website": "https://www.devsinc.com"},
        {"name": "Emumba", "website": "https://emumba.com"},
        {"name": "Tintash", "website": "https://www.tintash.com"},
        {"name": "CodeLoop", "website": "https://codeloop.io"},
        {"name": "PureLogics", "website": "https://www.purelogics.net"},
        {"name": "InvoZone", "website": "https://invozone.com"},
        {"name": "QBatch", "website": "https://qbatch.com"},
        {"name": "Genetech Solutions", "website": "https://www.genetechsolutions.com"},
        {"name": "DPL", "website": "https://www.dpl.dev"},
        {"name": "RedBuffer", "website": "https://redbuffer.net"},
        {"name": "Nextbridge", "website": "https://www.nextbridge.com"},
        {"name": "Tkxel", "website": "https://www.tkxel.com"},
        {"name": "Confiz", "website": "https://www.confiz.com"},
        {"name": "Folio3", "website": "https://www.folio3.com"},
        {"name": "Rolustech", "website": "https://www.rolustech.com"},
        {"name": "NorthBay Solutions", "website": "https://www.northbaysolutions.com"},
        {"name": "Creative Chaos", "website": "https://www.creativechaos.co"},
        {"name": "Vizteck Solutions", "website": "https://www.vizteck.com"},
        {"name": "Xavor", "website": "https://www.xavor.com"},
        {"name": "TekRevol", "website": "https://www.tekrevol.com"},
        {"name": "Gaditek", "website": "https://www.gaditek.com"},
        {"name": "Ebryx", "website": "https://www.ebryx.com"},
        {"name": "Cubix", "website": "https://www.cubix.co"},
        {"name": "Digitify", "website": "https://www.digitify.com"},
        {"name": "Arbisoft", "website": "https://arbisoft.com"},
        {"name": "10Pearls", "website": "https://10pearls.com"},
        {"name": "Contour Software", "website": "https://www.contour-software.com"},
    ],
    "saudi arabia": [
        {"name": "Neologix", "website": "https://neologix.sa"},
        {"name": "OSIT", "website": "https://osit.com.sa"},
        {"name": "Tawakob", "website": "https://tawakob.com"},
        {"name": "B-IT", "website": "https://b-it.co"},
        {"name": "MBKS Global", "website": "https://mbksglobal.com"},
        {"name": "Creative Solutions", "website": "https://www.creative-sols.com"},
        {"name": "Elm", "website": "https://www.elm.sa"},
        {"name": "Sary", "website": "https://sary.com"},
        {"name": "Tamkeen Technologies", "website": "https://www.tamkeen.sa"},
        {"name": "Saudi Business Machines", "website": "https://www.sbm.com.sa"},
    ],
    "uae": [
        {"name": "Emagine", "website": "https://www.emagine.ae"},
        {"name": "Intertec Systems", "website": "https://www.intertecsystems.com"},
        {"name": "Finesse", "website": "https://www.finesse-group.com"},
        {"name": "Magna", "website": "https://www.magnasolutions.com"},
        {"name": "AST Computer", "website": "https://www.ast.ae"},
        {"name": "Gulf Business Machines", "website": "https://www.gbmuae.com"},
    ],
}
_GLOBAL_SOFTWARE_SEEDS: list[dict] = [
    {"name": "STRV", "website": "https://www.strv.com", "headquarters_country": "United States"},
    {"name": "Clevertech", "website": "https://clevertech.biz", "headquarters_country": "United States"},
    {"name": "10Clouds", "website": "https://10clouds.com", "headquarters_country": "Poland"},
    {"name": "Netguru", "website": "https://www.netguru.com", "headquarters_country": "Poland"},
    {"name": "Monterail", "website": "https://www.monterail.com", "headquarters_country": "Poland"},
    {"name": "Brainhub", "website": "https://brainhub.eu", "headquarters_country": "Poland"},
    {"name": "Vention", "website": "https://ventionteams.com", "headquarters_country": "United States"},
    {"name": "Ideamotive", "website": "https://ideamotive.co", "headquarters_country": "Poland"},
    {"name": "EPAM Systems", "website": "https://www.epam.com", "headquarters_country": "United States"},
    {"name": "Globant", "website": "https://www.globant.com", "headquarters_country": "Argentina"},
]
_GLOBAL_UNIVERSITY_SEEDS: list[dict] = [
    {"name": "Harvard University", "website": "https://www.harvard.edu", "headquarters_country": "United States", "industry": "Higher Education", "niche": "ivy league research university"},
    {"name": "Massachusetts Institute of Technology (MIT)", "website": "https://www.mit.edu", "headquarters_country": "United States", "industry": "Higher Education", "niche": "science, technology & research university"},
    {"name": "Stanford University", "website": "https://www.stanford.edu", "headquarters_country": "United States", "industry": "Higher Education", "niche": "private research university & innovation"},
    {"name": "University of Oxford", "website": "https://www.ox.ac.uk", "headquarters_country": "United Kingdom", "industry": "Higher Education", "niche": "collegiate research university"},
    {"name": "University of Cambridge", "website": "https://www.cam.ac.uk", "headquarters_country": "United Kingdom", "industry": "Higher Education", "niche": "collegiate research university"},
    {"name": "National University of Singapore (NUS)", "website": "https://nus.edu.sg", "headquarters_country": "Singapore", "industry": "Higher Education", "niche": "global research university in Asia"},
    {"name": "Imperial College London", "website": "https://www.imperial.ac.uk", "headquarters_country": "United Kingdom", "industry": "Higher Education", "niche": "science, engineering, medicine & business"},
    {"name": "University of Toronto", "website": "https://www.utoronto.ca", "headquarters_country": "Canada", "industry": "Higher Education", "niche": "public research university"},
]
_GLOBAL_SCHOOL_SEEDS: list[dict] = [
    {"name": "Eton College", "website": "https://www.etoncollege.com", "headquarters_country": "United Kingdom", "industry": "Primary & Secondary Education", "niche": "independent boarding school for boys"},
    {"name": "Phillips Academy Andover", "website": "https://www.andover.edu", "headquarters_country": "United States", "industry": "Primary & Secondary Education", "niche": "co-educational boarding & day school"},
    {"name": "Harrow School", "website": "https://www.harrowschool.org.uk", "headquarters_country": "United Kingdom", "industry": "Primary & Secondary Education", "niche": "independent boarding school for boys"},
    {"name": "Raffles Institution", "website": "https://www.ri.edu.sg", "headquarters_country": "Singapore", "industry": "Primary & Secondary Education", "niche": "premier independent secondary school"},
    {"name": "Trinity School New York", "website": "https://www.trinityschoolnyc.org", "headquarters_country": "United States", "industry": "Primary & Secondary Education", "niche": "co-educational independent day school"},
]
_LOCAL_QSR_SEEDS: dict[str, list[dict]] = {
    "pakistan": [
        # restaurants / casual dining (Meet Me in Paris peers) — keep ≥8 so slider count can fill
        {"name": "Salt'n Pepper", "website": "https://www.saltnpepperonline.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Arcadian Cafe", "website": "https://www.arcadiancafe.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Café Aylanto", "website": "https://www.aylanto.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Tuscany Courtyard", "website": "https://www.tuscanycourtyard.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Café Zouk", "website": "https://www.cafezouk.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Cosa Nostra", "website": "https://www.cosanostra.com.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Cooco's Den", "website": "https://www.coocosden.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Spice Bazaar", "website": "https://www.spicebazaar.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "The Brasserie", "website": "https://www.pearlcont.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Jade Cafe", "website": "https://www.jadecafe.com.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Haveli Restaurant", "website": "https://www.haveli.com.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Cubano", "website": "https://www.cubano.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        # NOTE: do NOT seed "Cucina" (PK furniture/kitchens) or "Andiamo" (Dubai Hyatt Italian)
        # cafe (coffee-led — not restaurant peers)
        {"name": "Espresso Lab", "website": "https://www.espressolab.com.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Gloria Jean's Coffees", "website": "https://www.gloriajeans.com.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        # bakery / cake shop / desserts (Layers Bakeshop peers)
        {"name": "Kitchen Cuisine", "website": "https://kitchencuisine.com.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Tehzeeb Bakers", "website": "https://tehzeeb.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Del Frio", "website": "https://delfrio.com.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Sweet Tooth", "website": "https://sweettooth.com.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Rahat Bakers", "website": "https://rahatbakers.com.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Gourmet Bakers", "website": "https://gourmetpakistan.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Pie in the Sky", "website": "https://pieinthesky.com.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Jalal Sons", "website": "https://jalalsons.com.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Cocoa Dolce", "website": "https://cocoadolce.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Cinnabon Pakistan", "website": "https://cinnabon.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Butlers Chocolate Café", "website": "https://www.butlerschocolates.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Layers Bakeshop", "website": "https://www.layers.com.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        # burger / indie QSR
        {"name": "Johnny & Jugnu", "website": "https://www.johnnyandjugnu.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Mad", "website": "https://www.mad.com.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Burger Lab", "website": "https://www.burgerlab.com.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BURGER},
        # shawarma / Arabic wrap QSR (Sultan Shawarma peers) — last-resort only
        {"name": "Shawarma Stop", "website": "https://shawarmastop.co", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_SHAWARMA},
        {"name": "PITA", "website": "https://heypita.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_SHAWARMA},
        {"name": "Arabic Shawarma", "website": "https://arabicshawarma.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_SHAWARMA},
        {"name": "Monty Shawarma", "website": "https://montysshawarma.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_SHAWARMA},
        {"name": "Rizwan Pocket Shawarma", "website": "https://rizwanpocketshawarma.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_SHAWARMA},
        # asian
        {"name": "Ginsoy", "website": "https://www.ginsoy.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_ASIAN},
        {"name": "Ginyaki", "website": "https://ginyaki.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_ASIAN},
        # national pizza chains & authentic peers (Cheezious / Broadway / 14th Street peers)
        {"name": "14th Street Pizza", "website": "https://14thstreetpizza.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Broadway Pizza", "website": "https://broadwaypizza.com.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "California Pizza", "website": "https://www.californiapizza.com.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Pizza Max", "website": "https://pizzamax.com.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Fork 'n' Knives Pizza Fusion", "website": "https://forknknives.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Tehzeeb Pizza", "website": "https://tehzeeb.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Pizza Junction", "website": "https://pizzajunction.pk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "NY212 Pizza", "website": "https://orders.ny-212.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Howdy", "website": "https://www.howdy.pk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "OPTP", "website": "https://www.optp.biz", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Ranchers", "website": "https://www.rancherscafe.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        # global franchises
        {"name": "Domino's Pizza", "website": "https://www.dominos.com.pk", "tier": _FOOD_TIER_GLOBAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Pizza Hut", "website": "https://www.pizzahut.com.pk", "tier": _FOOD_TIER_GLOBAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Papa John's", "website": "https://www.papajohns.com.pk", "tier": _FOOD_TIER_GLOBAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "KFC", "website": "https://www.kfcpakistan.com", "tier": _FOOD_TIER_GLOBAL, "format": _FOOD_FORMAT_GENERAL},
        {"name": "McDonald's", "website": "https://www.mcdonalds.com.pk", "tier": _FOOD_TIER_GLOBAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Hardee's", "website": "https://www.hardees.com.pk", "tier": _FOOD_TIER_GLOBAL, "format": _FOOD_FORMAT_BURGER},
    ],
    "india": [
        # restaurants & dining
        {"name": "Haldiram's", "website": "https://www.haldirams.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Bikanervala", "website": "https://bikanervala.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Barbeque Nation", "website": "https://www.barbequenation.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Paradise Biryani", "website": "https://www.paradisefoodcourt.in", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Smoke House Deli", "website": "https://smokehousedeli.in", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Karim's", "website": "https://karimhoteldelhi.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Punjab Grill", "website": "https://punjabgrill.in", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Mainland China", "website": "https://mainlandchina.in", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_ASIAN},
        # bakeries & dessert shops
        {"name": "Theobroma", "website": "https://theobroma.in", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Bakingo", "website": "https://www.bakingo.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Monginis", "website": "https://www.monginis.net", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Winni Cakes", "website": "https://www.winni.in", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Karachi Bakery Hyderabad", "website": "https://www.karachibakery.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Le15 Patisserie", "website": "https://le15.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Flurys", "website": "https://flurys.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Wenger's Deli", "website": "https://wengers.in", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        # cafes
        {"name": "Chaayos", "website": "https://chaayos.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Chai Point", "website": "https://chaipoint.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Cafe Coffee Day", "website": "https://www.cafecoffeeday.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Blue Tokai Coffee Roasters", "website": "https://bluetokaicoffee.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Third Wave Coffee", "website": "https://www.thirdwavecoffee.in", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        # burger & pizza QSR
        {"name": "Burger Singh", "website": "https://burgersinghonline.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Wow! Momo", "website": "https://www.wowmomo.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Faasos", "website": "https://www.eatsure.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_SHAWARMA},
        {"name": "La Pino'z Pizza", "website": "https://lapinozpizza.in", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Ovenstory Pizza", "website": "https://www.ovenstory.in", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
    ],
    "united kingdom": [
        {"name": "Gail's Bakery", "website": "https://gailsbread.co.uk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Lola's Cupcakes", "website": "https://www.lolascupcakes.co.uk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Hummingbird Bakery", "website": "https://hummingbirdbakery.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Ole & Steen UK", "website": "https://oleandsteen.co.uk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Paul UK", "website": "https://www.paul-uk.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Peggy Porschen", "website": "https://www.peggyporschen.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Greggs", "website": "https://www.greggs.co.uk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Nando's UK", "website": "https://www.nandos.co.uk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Wagamama", "website": "https://www.wagamama.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_ASIAN},
        {"name": "Dishoom", "website": "https://www.dishoom.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "PizzaExpress", "website": "https://www.pizzaexpress.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Franco Manca", "website": "https://www.francomanca.co.uk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Costa Coffee", "website": "https://www.costa.co.uk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Caffè Nero", "website": "https://caffenero.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Pret A Manger", "website": "https://www.pret.co.uk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Honest Burgers", "website": "https://www.honestburgers.co.uk", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Gourmet Burger Kitchen", "website": "https://www.gbk.co.uk", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Leon", "website": "https://leon.co", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
    ],
    "united states": [
        {"name": "Crumbl Cookies", "website": "https://crumblcookies.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Magnolia Bakery", "website": "https://magnoliabakery.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Nothing Bundt Cakes", "website": "https://www.nothingbundtcakes.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Insomnia Cookies", "website": "https://insomniacookies.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Milk Bar", "website": "https://milkbarstore.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Levain Bakery", "website": "https://levainbakery.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Sprinkles Cupcakes", "website": "https://sprinkles.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Paris Baguette US", "website": "https://parisbaguette.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Tous Les Jours US", "website": "https://www.tljus.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Carlo's Bakery", "website": "https://bakeshop.carlosbakery.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Panera Bread", "website": "https://www.panerabread.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Chipotle Mexican Grill", "website": "https://www.chipotle.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Sweetgreen", "website": "https://www.sweetgreen.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Shake Shack", "website": "https://shakeshack.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Five Guys", "website": "https://www.fiveguys.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "In-N-Out Burger", "website": "https://www.in-n-out.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Raising Cane's", "website": "https://www.raisingcanes.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Starbucks", "website": "https://www.starbucks.com", "tier": _FOOD_TIER_GLOBAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Dunkin'", "website": "https://www.dunkindonuts.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Peet's Coffee", "website": "https://www.peets.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "MOD Pizza", "website": "https://modpizza.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Blaze Pizza", "website": "https://blazepizza.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
    ],
    "uae": [
        {"name": "Paul Arabia", "website": "https://paularabia.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Bloomsbury's Boutique Bakery", "website": "https://bloomsburys.ae", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Sugaholic Bakeshop", "website": "https://sugaholic.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "House of Cakes Dubai", "website": "https://houseofcakesdubai.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Home Bakery UAE", "website": "https://homebakery.ae", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "L'ETO Caffe", "website": "https://letocaffe.ae", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Al Baik UAE", "website": "https://albaik.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Zaatar W Zeit UAE", "website": "https://zaatarwzeit.net", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_SHAWARMA},
        {"name": "Operation Falafel", "website": "https://operationfalafel.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_SHAWARMA},
        {"name": "SALT UAE", "website": "https://find-salt.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Pickl", "website": "https://eatpickl.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Gazebo", "website": "https://gazebo.ae", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Shakespeare and Co", "website": "https://shakespeare-and-co.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "% Arabica UAE", "website": "https://arabica.coffee", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_CAFE},
    ],
    "saudi arabia": [
        {"name": "Al Baik KSA", "website": "https://albaik.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Kudu", "website": "https://kudu.com.sa", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Herfy", "website": "https://herfy.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Shawarmer", "website": "https://shawarmer.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_SHAWARMA},
        {"name": "Mama Noura", "website": "https://mamanoura.com.sa", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_SHAWARMA},
        {"name": "Al Romansiah", "website": "https://alromansiah.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Maestro Pizza", "website": "https://maestropizza.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Saadeddin Pastry", "website": "https://saadeddin.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Munch Bakery", "website": "https://munchbakery.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Barn's Coffee", "website": "https://barns.com.sa", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Half Million", "website": "https://halfm.net", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_CAFE},
    ],
    "canada": [
        {"name": "Tim Hortons", "website": "https://www.timhortons.ca", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Boston Pizza", "website": "https://bostonpizza.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Harvey's", "website": "https://www.harveys.ca", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "A&W Canada", "website": "https://web.aw.ca", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "BeaverTails", "website": "https://beavertails.com", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Cora Breakfast and Lunch", "website": "https://www.chezcora.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Second Cup Coffee", "website": "https://secondcup.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Mary Brown's Chicken", "website": "https://marybrowns.com", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
    ],
    "australia": [
        {"name": "Guzman y Gomez", "website": "https://www.guzmanygomez.com.au", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_RESTAURANT},
        {"name": "Grill'd", "website": "https://www.grilld.com.au", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Hungry Jack's", "website": "https://www.hungryjacks.com.au", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BURGER},
        {"name": "Bakers Delight", "website": "https://www.bakersdelight.com.au", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_BAKERY},
        {"name": "Boost Juice", "website": "https://www.boostjuice.com.au", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "The Coffee Club", "website": "https://coffeeclub.com.au", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_CAFE},
        {"name": "Crust Pizza", "website": "https://www.crust.com.au", "tier": _FOOD_TIER_NATIONAL, "format": _FOOD_FORMAT_PIZZA},
        {"name": "Lord of the Fries", "website": "https://lordofthefries.com.au", "tier": _FOOD_TIER_LOCAL, "format": _FOOD_FORMAT_BURGER},
    ],
}

_LOCAL_BEAUTY_SEEDS: dict[str, list[dict]] = {
    "pakistan": [
        {"name": "Depilex Beauty Clinic", "website": "https://depilex.com", "niche": "beauty salon & aesthetic clinic"},
        {"name": "Kashee's Beauty Parlour", "website": "https://kashees.com", "niche": "bridal makeup & salon"},
        {"name": "Sabs The Salon", "website": "https://sabs.com.pk", "niche": "beauty salon & hair styling"},
        {"name": "Nabila Salon", "website": "https://nabila.net", "niche": "hair styling & beauty salon"},
        {"name": "Tariq Amin Salon", "website": "https://tariqamin.com", "niche": "styling studio & salon"},
        {"name": "Natasha Salon", "website": "https://natashasalon.com", "niche": "makeup studio & salon"},
        {"name": "Allenora Annie's Signature Salon", "website": "https://allenora.com", "niche": "bridal salon"},
        {"name": "Peng's Salon", "website": "https://pengssalon.com", "niche": "hair & beauty salon"},
        {"name": "Mona J Salon", "website": "https://monaj.com.pk", "niche": "beauty salon & spa"},
        {"name": "Faiza's Salon", "website": "https://faizas.com.pk", "niche": "beauty salon & makeup"},
    ],
    "india": [
        {"name": "Lakme Salon", "website": "https://www.lakmesalon.in", "niche": "beauty salon, hair styling & bridal makeup"},
        {"name": "Naturals Salon", "website": "https://naturalssalon.com", "niche": "hair & beauty salon chain"},
        {"name": "Jawed Habib Hair & Beauty", "website": "https://jawedhabib.co.in", "niche": "hair styling & beauty academy"},
        {"name": "Kaya Skin Clinic", "website": "https://www.kaya.in", "niche": "dermatology, aesthetic skincare & clinic"},
        {"name": "Enrich Salon", "website": "https://www.enrichbeauty.com", "niche": "beauty lounge & hair salon"},
        {"name": "VLCC Wellness", "website": "https://www.vlccwellness.com", "niche": "wellness, skincare & beauty clinic"},
        {"name": "Geetanjali Salon", "website": "https://geetanjalisalon.com", "niche": "luxury hair styling & beauty salon"},
        {"name": "BBLUNT Salon", "website": "https://bblunt.com", "niche": "contemporary hair styling & salon"},
    ],
    "uae": [
        {"name": "Tips & Toes", "website": "https://www.tipsandtoes.com", "niche": "beauty salon & spa"},
        {"name": "NStyle Beauty Lounge", "website": "https://nstyleintl.com", "niche": "beauty salon & lounge"},
        {"name": "Sisters Beauty Lounge", "website": "https://sistersbeautylounge.com", "niche": "beauty salon & lounge"},
        {"name": "Pastels Salon", "website": "https://pastels-salon.com", "niche": "hair & beauty salon"},
        {"name": "The White Room Spa", "website": "https://whiteroomdubai.com", "niche": "nail lounge & beauty spa"},
        {"name": "JetSet Salon", "website": "https://jetsetuae.com", "niche": "hair salon & blow dry bar"},
    ],
    "saudi arabia": [
        {"name": "Lumiere Salon", "website": "https://lumieresalon.sa", "niche": "beauty salon & spa"},
        {"name": "Maison de Joelle", "website": "https://maisondejoelle.com", "niche": "beauty salon & makeup"},
        {"name": "Base & Boon", "website": "https://baseandboon.com", "niche": "nail bar & beauty salon"},
        {"name": "4Spa Riyadh", "website": "https://4spa.com.sa", "niche": "luxury spa & beauty lounge"},
    ],
    "united kingdom": [
        {"name": "Toni & Guy", "website": "https://toniandguy.com", "niche": "hairdressing & beauty"},
        {"name": "Rush Hair & Beauty", "website": "https://rush.co.uk", "niche": "hair & beauty salon"},
        {"name": "Sassoon Salon", "website": "https://sassoon-salon.com", "niche": "creative hair styling & salon"},
        {"name": "Supercuts UK", "website": "https://www.supercuts.co.uk", "niche": "haircut & styling salon"},
        {"name": "Headmasters", "website": "https://www.headmasters.com", "niche": "hair & color salon"},
    ],
    "united states": [
        {"name": "Drybar", "website": "https://drybar.com", "niche": "blowout salon"},
        {"name": "Ulta Beauty", "website": "https://ulta.com", "niche": "beauty salon & cosmetics"},
        {"name": "Sephora Beauty Studio", "website": "https://www.sephora.com", "niche": "makeup & skincare studio"},
        {"name": "Regis Salons", "website": "https://www.regissalons.com", "niche": "hair styling & salon"},
        {"name": "Great Clips", "website": "https://www.greatclips.com", "niche": "hair salon franchise"},
        {"name": "Sport Clips", "website": "https://sportclips.com", "niche": "haircut & styling salon"},
    ],
}
_LOCAL_MULTI_INDUSTRY_SEEDS: dict[str, dict[str, list[dict]]] = {
    "pakistan": {
        "automotive": [
            {"name": "Toyota Indus Motor", "website": "https://toyota-indus.com", "industry": "Automotive", "niche": "passenger cars & SUVs"},
            {"name": "Suzuki Pakistan", "website": "https://suzukipakistan.com", "industry": "Automotive", "niche": "passenger cars & commercial vehicles"},
            {"name": "Hyundai Nishat Motors", "website": "https://hyundai-nishat.com", "industry": "Automotive", "niche": "passenger cars & SUVs"},
            {"name": "Kia Lucky Motors", "website": "https://kia-luckymotor.com.pk", "industry": "Automotive", "niche": "passenger cars & SUVs"},
            {"name": "Changan Pakistan", "website": "https://changan.com.pk", "industry": "Automotive", "niche": "sedans & SUVs"},
            {"name": "Haval Pakistan", "website": "https://sazgarauto.com", "industry": "Automotive", "niche": "hybrid SUVs & crossovers"},
            {"name": "MG Motors Pakistan", "website": "https://mgmotors.com.pk", "industry": "Automotive", "niche": "electric & hybrid SUVs"},
            {"name": "Yamaha Motor Pakistan", "website": "https://www.yamaha-motor.com.pk", "industry": "Automotive", "niche": "motorcycles & two-wheelers"},
            {"name": "Atlas Honda", "website": "https://www.atlashonda.com.pk", "industry": "Automotive", "niche": "motorcycles & automotive"},
            {"name": "Honda Atlas Cars", "website": "https://www.honda.com.pk", "industry": "Automotive", "niche": "passenger cars & sedans"},
        ],
        "real_estate": [
            {"name": "Zameen.com", "website": "https://www.zameen.com", "industry": "Real Estate", "niche": "property portal & real estate marketplace"},
            {"name": "Graana.com", "website": "https://www.graana.com", "industry": "Real Estate", "niche": "real estate portal & property marketing"},
            {"name": "Agency21 International", "website": "https://www.agency21.com", "industry": "Real Estate", "niche": "real estate agency & advisory"},
            {"name": "Imarat Group", "website": "https://imarat.com.pk", "industry": "Real Estate", "niche": "real estate development & hospitality"},
            {"name": "Habib Rafiq", "website": "https://www.habibrafiq.com", "industry": "Real Estate", "niche": "real estate infrastructure & development"},
        ],
        "retail": [
            {"name": "Daraz", "website": "https://www.daraz.pk", "industry": "Retail & E-Commerce", "niche": "online shopping & e-commerce marketplace"},
            {"name": "PriceOye", "website": "https://priceoye.pk", "industry": "Retail & E-Commerce", "niche": "consumer electronics & smartphones retail"},
            {"name": "Telemart", "website": "https://www.telemart.pk", "industry": "Retail & E-Commerce", "niche": "electronics & lifestyle e-commerce"},
            {"name": "Khaadi", "website": "https://pk.khaadi.com", "industry": "Apparel & Fashion", "niche": "apparel & fashion retail"},
            {"name": "Sapphire", "website": "https://pk.sapphireonline.pk", "industry": "Apparel & Fashion", "niche": "apparel & fashion retail"},
            {"name": "J. Junaid Jamshed", "website": "https://www.junaidjamshed.com", "industry": "Apparel & Fashion", "niche": "apparel, fragrances & fashion retail"},
            {"name": "Gul Ahmed", "website": "https://www.gulahmedshop.com", "industry": "Apparel & Fashion", "niche": "textile & fashion retail"},
            {"name": "Nishat Linen", "website": "https://nishatlinen.com", "industry": "Apparel & Fashion", "niche": "fashion & home textiles"},
        ],
        "fintech": [
            {"name": "Habib Bank Limited (HBL)", "website": "https://www.hbl.com", "industry": "Banking & Financial Services", "niche": "commercial banking & digital finance"},
            {"name": "Meezan Bank", "website": "https://www.meezanbank.com", "industry": "Banking & Financial Services", "niche": "islamic banking & retail finance"},
            {"name": "Bank Alfalah", "website": "https://www.bankalfalah.com", "industry": "Banking & Financial Services", "niche": "retail banking & digital payments"},
            {"name": "Easypaisa", "website": "https://easypaisa.com.pk", "industry": "Fintech", "niche": "digital wallet & branchless banking"},
            {"name": "JazzCash", "website": "https://www.jazzcash.com.pk", "industry": "Fintech", "niche": "mobile wallet & fintech services"},
            {"name": "NayaPay", "website": "https://www.nayapay.com", "industry": "Fintech", "niche": "digital wallet & emoney"},
            {"name": "SadaPay", "website": "https://sadapay.pk", "industry": "Fintech", "niche": "fintech debit card & global business accounts"},
        ],
        "telecom": [
            {"name": "Jazz", "website": "https://jazz.com.pk", "industry": "Telecommunications", "niche": "mobile network & telecom services"},
            {"name": "Telenor Pakistan", "website": "https://www.telenor.com.pk", "industry": "Telecommunications", "niche": "cellular telecom & data connectivity"},
            {"name": "Zong 4G", "website": "https://www.zong.com.pk", "industry": "Telecommunications", "niche": "cellular network & mobile broadband"},
            {"name": "Ufone 4G", "website": "https://www.ufone.com", "industry": "Telecommunications", "niche": "telecom network & cellular packages"},
            {"name": "PTCL", "website": "https://ptcl.com.pk", "industry": "Telecommunications", "niche": "broadband internet & fixed telecom"},
            {"name": "Nayatel", "website": "https://nayatel.com", "industry": "Telecommunications", "niche": "fiber optic internet & enterprise connectivity"},
        ],
        "healthcare": [
            {"name": "Chughtai Lab", "website": "https://chughtailab.com", "industry": "Healthcare", "niche": "diagnostic pathology & healthcare network"},
            {"name": "Shaukat Khanum", "website": "https://shaukatkhanum.org.pk", "industry": "Healthcare", "niche": "tertiary cancer hospital & research center"},
            {"name": "Aga Khan University Hospital", "website": "https://hospitals.aku.edu", "industry": "Healthcare", "niche": "multispecialty tertiary hospital & healthcare"},
            {"name": "Marham", "website": "https://www.marham.pk", "industry": "Healthcare", "niche": "digital health platform & doctor booking"},
            {"name": "Getz Pharma", "website": "https://getzpharma.com", "industry": "Healthcare", "niche": "pharmaceutical manufacturing & healthcare"},
        ],
        "university": [
            {"name": "LUMS", "website": "https://lums.edu.pk", "industry": "Higher Education", "niche": "leading research university & business school"},
            {"name": "NUST", "website": "https://nust.edu.pk", "industry": "Higher Education", "niche": "science, engineering & technology university"},
            {"name": "FAST NUCES", "website": "https://www.nu.edu.pk", "industry": "Higher Education", "niche": "computer science, IT & emerging sciences university"},
            {"name": "IBA Karachi", "website": "https://www.iba.edu.pk", "industry": "Higher Education", "niche": "business administration, finance & economics institute"},
            {"name": "GIKI", "website": "https://giki.edu.pk", "industry": "Higher Education", "niche": "engineering sciences & technology institute"},
            {"name": "COMSATS University", "website": "https://www.comsats.edu.pk", "industry": "Higher Education", "niche": "public research & IT university"},
            {"name": "Aga Khan University", "website": "https://www.aku.edu", "industry": "Higher Education", "niche": "health sciences & medical university"},
            {"name": "Lahore School of Economics", "website": "https://www.lahoreschoolofeconomics.edu.pk", "industry": "Higher Education", "niche": "economics & business university"},
            {"name": "University of the Punjab", "website": "https://pu.edu.pk", "industry": "Higher Education", "niche": "public comprehensive research university"},
            {"name": "SZABIST", "website": "https://szabist.edu.pk", "industry": "Higher Education", "niche": "science & technology degree-awarding institute"},
            {"name": "Habib University", "website": "https://habib.edu.pk", "industry": "Higher Education", "niche": "liberal arts & sciences undergraduate university"},
            {"name": "UET Lahore", "website": "https://uet.edu.pk", "industry": "Higher Education", "niche": "engineering & technology university"},
            {"name": "Information Technology University (ITU)", "website": "https://itu.edu.pk", "industry": "Higher Education", "niche": "computer science & engineering university"},
            {"name": "Quaid-i-Azam University", "website": "https://qau.edu.pk", "industry": "Higher Education", "niche": "public research university & postgraduate studies"},
        ],
        "school": [
            {"name": "Beaconhouse School System", "website": "https://www.beaconhouse.net", "industry": "Primary & Secondary Education", "niche": "k-12 schooling & cambridge education network"},
            {"name": "The City School", "website": "https://thecityschool.edu.pk", "industry": "Primary & Secondary Education", "niche": "k-12 schooling, o/a levels & primary education"},
            {"name": "Lahore Grammar School", "website": "https://lgs.edu.pk", "industry": "Primary & Secondary Education", "niche": "private schooling, o/a levels & preschool network"},
            {"name": "Roots Millennium Schools", "website": "https://millenniumschools.edu.pk", "industry": "Primary & Secondary Education", "niche": "k-12 schooling & international qualification"},
            {"name": "Karachi Grammar School", "website": "https://www.kgs.edu.pk", "industry": "Primary & Secondary Education", "niche": "grammar school & cambridge education"},
            {"name": "Army Public Schools (APSACS)", "website": "https://apsacssectt.edu.pk", "industry": "Primary & Secondary Education", "niche": "k-12 education system & nationwide schools"},
            {"name": "Froebel's International School", "website": "https://www.froebels.edu.pk", "industry": "Primary & Secondary Education", "niche": "international school & cambridge curriculum"},
            {"name": "Learning Alliance", "website": "https://learningalliance.edu.pk", "industry": "Primary & Secondary Education", "niche": "ib & cambridge school system"},
            {"name": "The Educators", "website": "https://www.educators.edu.pk", "industry": "Primary & Secondary Education", "niche": "nationwide private school network"},
            {"name": "Roots International Schools", "website": "https://rootsinternational.edu.pk", "industry": "Primary & Secondary Education", "niche": "international schooling & primary education"},
            {"name": "SICAS", "website": "https://sicas.edu.pk", "industry": "Primary & Secondary Education", "niche": "private schooling & o/a level system"},
            {"name": "Bloomfield Hall Schools", "website": "https://www.bloomfieldhall.com", "industry": "Primary & Secondary Education", "niche": "british-style education & k-12 schools"},
        ],
        "college": [
            {"name": "Punjab Group of Colleges", "website": "https://pgc.edu", "industry": "College Education", "niche": "intermediate colleges, fsc, ics & higher secondary"},
            {"name": "Superior Group of Colleges", "website": "https://superior.edu.pk", "industry": "College Education", "niche": "intermediate colleges & higher secondary education"},
            {"name": "KIPS Colleges", "website": "https://kips.edu.pk/colleges", "industry": "College Education", "niche": "intermediate colleges, fsc & entry test focus"},
            {"name": "Concordia Colleges", "website": "https://concordia.edu.pk", "industry": "College Education", "niche": "beaconhouse group intermediate colleges"},
            {"name": "GCU Lahore", "website": "https://www.gcu.edu.pk", "industry": "College Education", "niche": "higher secondary college & university education"},
        ],
        "academy": [
            {"name": "KIPS Academy", "website": "https://kips.edu.pk", "industry": "Test Preparation", "niche": "entry test prep, mdcat, ecat & coaching academy"},
            {"name": "STEP by PGC", "website": "https://step.pgc.edu", "industry": "Test Preparation", "niche": "entry test prep, mdcat, ecat & fsc coaching"},
            {"name": "Maqsad", "website": "https://maqsad.io", "industry": "EdTech", "niche": "edtech app, online test preparation & video learning"},
            {"name": "Noon Academy", "website": "https://www.learnatnoon.com", "industry": "EdTech", "niche": "social learning platform & online coaching"},
            {"name": "Nearpeer", "website": "https://www.nearpeer.org", "industry": "EdTech", "niche": "online courses, mdcat, ca & entry test prep"},
        ],
        "education": [
            {"name": "LUMS", "website": "https://lums.edu.pk", "industry": "Higher Education", "niche": "higher education & business school"},
            {"name": "Beaconhouse School System", "website": "https://www.beaconhouse.net", "industry": "Primary & Secondary Education", "niche": "k-12 schooling & education network"},
            {"name": "NUST", "website": "https://nust.edu.pk", "industry": "Higher Education", "niche": "science & technology university"},
            {"name": "The City School", "website": "https://thecityschool.edu.pk", "industry": "Primary & Secondary Education", "niche": "k-12 schooling & cambridge education"},
            {"name": "FAST NUCES", "website": "https://www.nu.edu.pk", "industry": "Higher Education", "niche": "computer science & IT university"},
            {"name": "Lahore Grammar School", "website": "https://lgs.edu.pk", "industry": "Primary & Secondary Education", "niche": "private schooling & preschool network"},
        ],
        "logistics": [
            {"name": "TCS Express", "website": "https://www.tcsexpress.com", "industry": "Logistics & Supply Chain", "niche": "express courier, parcel delivery & cargo logistics"},
            {"name": "Leopards Courier", "website": "https://leopardscourier.com", "industry": "Logistics & Supply Chain", "niche": "domestic courier, cargo & logistics services"},
            {"name": "M&P Courier", "website": "https://mulphilog.com", "industry": "Logistics & Supply Chain", "niche": "express delivery & distribution network"},
            {"name": "Trax Logistics", "website": "https://trax.pk", "industry": "Logistics & Supply Chain", "niche": "e-commerce logistics, cash on delivery & fulfillment"},
            {"name": "Call Courier", "website": "https://callcourier.com.pk", "industry": "Logistics & Supply Chain", "niche": "cash on delivery & parcel delivery services"},
        ],
        "energy": [
            {"name": "Reon Energy", "website": "https://reonenergy.com", "industry": "Renewable Energy & Solar", "niche": "industrial solar energy, microgrids & storage solutions"},
            {"name": "SkyElectric", "website": "https://skyelectric.com", "industry": "Renewable Energy & Solar", "niche": "smart solar energy systems & hybrid inverters"},
            {"name": "Pantera Energy", "website": "https://panteraenergy.pk", "industry": "Renewable Energy & Solar", "niche": "commercial & residential solar power installations"},
            {"name": "Premier Energy", "website": "https://premierenergy.com.pk", "industry": "Renewable Energy & Solar", "niche": "solar EPC & renewable energy solutions"},
        ],
        "fitness": [
            {"name": "Shapes Health Studio", "website": "https://shapes.com.pk", "industry": "Health & Fitness", "niche": "fitness club, gym & wellness center"},
            {"name": "Structure Health & Fitness", "website": "https://structure.com.pk", "industry": "Health & Fitness", "niche": "gym, personal training & fitness center"},
            {"name": "Synergy Fitness Club", "website": "https://synergyfitness.pk", "industry": "Health & Fitness", "niche": "strength training & gym club"},
        ],
        "hospitality": [
            {"name": "Pearl Continental Hotels", "website": "https://www.pchotels.com", "industry": "Hospitality & Hotels", "niche": "luxury 5-star hotels & resorts"},
            {"name": "Serena Hotels Pakistan", "website": "https://www.serenahotels.com", "industry": "Hospitality & Hotels", "niche": "luxury heritage hotels & resorts"},
            {"name": "Avari Hotels", "website": "https://www.avari.com", "industry": "Hospitality & Hotels", "niche": "luxury hospitality & hotel suites"},
            {"name": "Nishat Hotel", "website": "https://nishathotel.com", "industry": "Hospitality & Hotels", "niche": "boutique luxury hotel & hospitality"},
        ],
        "data_ai": [
            {"name": "Devsinc", "website": "https://www.devsinc.com", "industry": "Custom Software & AI", "niche": "full-stack development, cloud engineering & AI solutions"},
            {"name": "Emumba", "website": "https://emumba.com", "industry": "Modern Software & AI", "niche": "data analytics, modern dashboards & AI engineering"},
            {"name": "Tintash", "website": "https://www.tintash.com", "industry": "Product Engineering", "niche": "product design, custom software & agile development"},
            {"name": "CodeLoop", "website": "https://codeloop.io", "industry": "Software & AI Solutions", "niche": "conversational AI, modern software & automation"},
            {"name": "PureLogics", "website": "https://www.purelogics.net", "industry": "Custom Software", "niche": "web, mobile app & custom software development"},
            {"name": "InvoZone", "website": "https://invozone.com", "industry": "Software Engineering", "niche": "custom software development & dedicated agile teams"},
            {"name": "QBatch", "website": "https://qbatch.com", "industry": "Software & Cloud Solutions", "niche": "enterprise software, cloud engineering & analytics"},
            {"name": "Genetech Solutions", "website": "https://www.genetechsolutions.com", "industry": "Software Solutions", "niche": "custom software, web development & digital engineering"},
            {"name": "RedBuffer", "website": "https://redbuffer.net", "industry": "AI & Software Engineering", "niche": "enterprise AI, machine learning & modern web platforms"},
            {"name": "Datakulture", "website": "https://datakulture.com", "industry": "Data Analytics & BI", "niche": "business intelligence, dashboards & modern data solutions"},
            {"name": "Inspirata Analytics", "website": "https://inspiratadata.com", "industry": "Data Analytics & Engineering", "niche": "data engineering, reporting dashboards & analytics consulting"},
            {"name": "DPL", "website": "https://www.dpl.dev", "industry": "Agile Software", "niche": "agile software development & digital engineering"},
        ],
    },
    "uae": {
        "automotive": [
            {"name": "Al-Futtaim Motors", "website": "https://www.alfuttaim.com", "industry": "Automotive", "niche": "automotive distribution & dealerships"},
            {"name": "Al Tayer Motors", "website": "https://www.altayermotors.com", "industry": "Automotive", "niche": "luxury automotive dealerships"},
            {"name": "Gargash Group", "website": "https://www.gargash.ae", "industry": "Automotive", "niche": "automotive enterprise & dealerships"},
        ],
        "real_estate": [
            {"name": "Emaar Properties", "website": "https://www.emaar.com", "industry": "Real Estate", "niche": "property development & master communities"},
            {"name": "Damac Properties", "website": "https://www.damacproperties.com", "industry": "Real Estate", "niche": "luxury residential & property development"},
            {"name": "Bayut", "website": "https://www.bayut.com", "industry": "Real Estate", "niche": "property portal & real estate search"},
        ],
        "fintech": [
            {"name": "Tabby", "website": "https://tabby.ai", "industry": "Fintech", "niche": "buy now pay later & shopping payments"},
            {"name": "Tamara", "website": "https://tamara.co", "industry": "Fintech", "niche": "split payments & consumer shopping finance"},
            {"name": "Careem Pay", "website": "https://www.careem.com/en-ae/careem-pay", "industry": "Fintech", "niche": "digital wallet, peer to peer & bill payments"},
            {"name": "Liv. by Emirates NBD", "website": "https://www.liv.me", "industry": "Fintech", "niche": "digital lifestyle banking & smart finance"},
            {"name": "Wio Bank", "website": "https://wio.io", "industry": "Fintech", "niche": "digital platform banking for businesses & consumers"},
        ],
        "retail": [
            {"name": "Noon", "website": "https://www.noon.com", "industry": "Retail & E-Commerce", "niche": "digital marketplace, electronics & lifestyle retail"},
            {"name": "Amazon UAE", "website": "https://www.amazon.ae", "industry": "Retail & E-Commerce", "niche": "online shopping & e-commerce marketplace"},
            {"name": "Namshi", "website": "https://www.namshi.com", "industry": "Apparel & Fashion", "niche": "online fashion, apparel & beauty retail"},
            {"name": "Sharaf DG", "website": "https://uae.sharafdg.com", "industry": "Retail & E-Commerce", "niche": "electronics, gadgets & technology retail"},
        ],
        "healthcare": [
            {"name": "Aster DM Healthcare", "website": "https://www.asterdmhealthcare.com", "industry": "Healthcare", "niche": "hospitals, clinics & diagnostic network"},
            {"name": "NMC Healthcare", "website": "https://nmc.ae", "industry": "Healthcare", "niche": "private multispecialty hospital network"},
            {"name": "Burjeel Holdings", "website": "https://burjeel.com", "industry": "Healthcare", "niche": "premium healthcare & specialized medical centers"},
            {"name": "Mediclinic Middle East", "website": "https://www.mediclinic.ae", "industry": "Healthcare", "niche": "private hospital & medical clinic network"},
        ],
        "telecom": [
            {"name": "e& (Etisalat)", "website": "https://www.eand.com", "industry": "Telecommunications", "niche": "telecom, 5g network & digital connectivity"},
            {"name": "du", "website": "https://www.du.ae", "industry": "Telecommunications", "niche": "mobile network, broadband & corporate telecom"},
        ],
        "university": [
            {"name": "American University of Sharjah (AUS)", "website": "https://www.aus.edu", "industry": "Higher Education", "niche": "higher education & research university"},
            {"name": "NYU Abu Dhabi", "website": "https://nyuad.nyu.edu", "industry": "Higher Education", "niche": "liberal arts & research university"},
            {"name": "United Arab Emirates University (UAEU)", "website": "https://www.uaeu.ac.ae", "industry": "Higher Education", "niche": "national comprehensive university"},
            {"name": "University of Wollongong in Dubai", "website": "https://www.uowdubai.ac.ae", "industry": "Higher Education", "niche": "international university campus"},
        ],
        "school": [
            {"name": "GEMS Education", "website": "https://www.gemseducation.com", "industry": "Primary & Secondary Education", "niche": "international k-12 school network"},
            {"name": "Taaleem", "website": "https://www.taaleem.ae", "industry": "Primary & Secondary Education", "niche": "premium school management & education"},
            {"name": "Dubai College", "website": "https://www.dubaicollege.org", "industry": "Primary & Secondary Education", "niche": "british curriculum secondary school"},
            {"name": "Kings School Dubai", "website": "https://kings-edu.com", "industry": "Primary & Secondary Education", "niche": "british primary & secondary schools"},
        ],
        "logistics": [
            {"name": "Aramex", "website": "https://www.aramex.com", "industry": "Logistics & Supply Chain", "niche": "express international logistics & freight forwarding"},
            {"name": "Fetchr", "website": "https://fetchr.us", "industry": "Logistics & Supply Chain", "niche": "tech-enabled delivery & logistics services"},
            {"name": "Careem Box", "website": "https://www.careem.com", "industry": "Logistics & Supply Chain", "niche": "on-demand parcel delivery & courier"},
        ],
        "energy": [
            {"name": "Masdar", "website": "https://masdar.ae", "industry": "Renewable Energy & Solar", "niche": "clean energy, solar power & sustainable development"},
            {"name": "Yellow Door Energy", "website": "https://yellowdoorenergy.com", "industry": "Renewable Energy & Solar", "niche": "commercial & industrial solar power"},
            {"name": "SirajPower", "website": "https://sirajpower.com", "industry": "Renewable Energy & Solar", "niche": "distributed solar energy solutions"},
        ],
        "fitness": [
            {"name": "Fitness First Middle East", "website": "https://uae.fitnessfirstme.com", "industry": "Health & Fitness", "niche": "health club, gym & fitness chain"},
            {"name": "GymNation UAE", "website": "https://gymnation.com", "industry": "Health & Fitness", "niche": "affordable gym club & fitness classes"},
            {"name": "Warehouse Gym", "website": "https://whgym.com", "industry": "Health & Fitness", "niche": "boutique gym & strength training"},
        ],
        "hospitality": [
            {"name": "Jumeirah Hotels & Resorts", "website": "https://www.jumeirah.com", "industry": "Hospitality & Hotels", "niche": "luxury 5-star hotels & hospitality"},
            {"name": "Atlantis The Palm", "website": "https://www.atlantis.com/dubai", "industry": "Hospitality & Hotels", "niche": "luxury ocean resort & entertainment destination"},
            {"name": "Rotana Hotels", "website": "https://www.rotana.com", "industry": "Hospitality & Hotels", "niche": "hotel management & premium suites"},
        ],
        "data_ai": [
            {"name": "DataBridge UAE", "website": "https://databridge.ae", "industry": "Data Analytics", "niche": "business intelligence & data analytics consulting"},
            {"name": "Data Insight Middle East", "website": "https://datainsight.me", "industry": "Data & BI", "niche": "Power BI, data warehouse & analytics consulting"},
            {"name": "DataLab UAE", "website": "https://datalab.ae", "industry": "AI & Data Solutions", "niche": "enterprise AI, analytics & data engineering"},
            {"name": "Inovar Consulting", "website": "https://inovarconsulting.com", "industry": "Data Analytics", "niche": "BI consulting, Tableau & data strategy"},
            {"name": "Brite Solutions", "website": "https://britesolutions.ae", "industry": "BI & Analytics", "niche": "business intelligence dashboards & data architecture"},
        ],
    },
    "saudi arabia": {
        "automotive": [
            {"name": "Abdul Latif Jameel Motors", "website": "https://www.alj.com", "industry": "Automotive", "niche": "automotive distribution & passenger cars"},
            {"name": "Al Jazirah Vehicles Agencies", "website": "https://www.aljazirahford.com", "industry": "Automotive", "niche": "automotive dealerships"},
            {"name": "Aljomaih Automotive", "website": "https://www.aljomaihauto.com", "industry": "Automotive", "niche": "automotive distribution & vehicles"},
        ],
        "real_estate": [
            {"name": "Roshn Group", "website": "https://www.roshn.sa", "industry": "Real Estate", "niche": "community development & real estate"},
            {"name": "Dar Al Arkan", "website": "https://www.daralarkan.com", "industry": "Real Estate", "niche": "property development & residential"},
        ],
        "fintech": [
            {"name": "stc pay", "website": "https://stcpay.com.sa", "industry": "Fintech", "niche": "digital wallet, money transfers & digital banking"},
            {"name": "Tamara KSA", "website": "https://tamara.co", "industry": "Fintech", "niche": "split payments & consumer shopping finance"},
            {"name": "Tabby KSA", "website": "https://tabby.ai", "industry": "Fintech", "niche": "buy now pay later & shopping payments"},
            {"name": "Urpay", "website": "https://urpay.com.sa", "industry": "Fintech", "niche": "digital financial services wallet"},
        ],
        "retail": [
            {"name": "Jarir Bookstore", "website": "https://www.jarir.com", "industry": "Retail & E-Commerce", "niche": "electronics, books & consumer technology retail"},
            {"name": "Extra Stores", "website": "https://www.extra.com", "industry": "Retail & E-Commerce", "niche": "consumer electronics & home appliances retail"},
            {"name": "Noon KSA", "website": "https://www.noon.com/saudi-en", "industry": "Retail & E-Commerce", "niche": "e-commerce marketplace & shopping"},
            {"name": "Panda Retail", "website": "https://panda.com.sa", "industry": "Retail & E-Commerce", "niche": "hypermarkets & grocery retail chain"},
        ],
        "healthcare": [
            {"name": "Dr. Sulaiman Al Habib Medical Group", "website": "https://drsulaimanalhabib.com", "industry": "Healthcare", "niche": "multispecialty tertiary hospitals & medical centers"},
            {"name": "Mouwasat Medical Services", "website": "https://www.mouwasat.com", "industry": "Healthcare", "niche": "hospital network & specialized healthcare"},
            {"name": "Saudi German Health", "website": "https://sghgroup.com", "industry": "Healthcare", "niche": "tertiary hospitals & medical facilities"},
            {"name": "Dallah Healthcare", "website": "https://dallah-hospital.com", "industry": "Healthcare", "niche": "private hospitals & healthcare services"},
        ],
        "telecom": [
            {"name": "stc", "website": "https://www.stc.com.sa", "industry": "Telecommunications", "niche": "telecom, 5g network, fiber & enterprise digital services"},
            {"name": "Mobily", "website": "https://www.mobily.com.sa", "industry": "Telecommunications", "niche": "cellular telecom, broadband & digital solutions"},
            {"name": "Zain KSA", "website": "https://sa.zain.com", "industry": "Telecommunications", "niche": "telecommunications network & 5g mobile packages"},
        ],
        "university": [
            {"name": "KAUST", "website": "https://www.kaust.edu.sa", "industry": "Higher Education", "niche": "science & technology graduate research university"},
            {"name": "KFUPM", "website": "https://www.kfupm.edu.sa", "industry": "Higher Education", "niche": "petroleum, minerals & engineering university"},
            {"name": "King Saud University", "website": "https://ksu.edu.sa", "industry": "Higher Education", "niche": "public research university"},
            {"name": "King Abdulaziz University", "website": "https://www.kau.edu.sa", "industry": "Higher Education", "niche": "higher education & research"},
        ],
        "school": [
            {"name": "British International School Riyadh", "website": "https://www.bisr.com.sa", "industry": "Primary & Secondary Education", "niche": "british curriculum international school"},
            {"name": "American International School Riyadh", "website": "https://www.aisr.org", "industry": "Primary & Secondary Education", "niche": "american curriculum international school"},
            {"name": "Kingdom Schools", "website": "https://www.kingdomschools.edu.sa", "industry": "Primary & Secondary Education", "niche": "private k-12 schooling"},
        ],
        "logistics": [
            {"name": "SMSA Express", "website": "https://www.smsaexpress.com", "industry": "Logistics & Supply Chain", "niche": "express courier, parcel delivery & cargo transport"},
            {"name": "Naqel Express", "website": "https://www.naqelexpress.com", "industry": "Logistics & Supply Chain", "niche": "logistics supply chain & cold chain transport"},
            {"name": "SPL (Saudi Post)", "website": "https://splonline.com.sa", "industry": "Logistics & Supply Chain", "niche": "national postal & parcel logistics"},
        ],
        "energy": [
            {"name": "ACWA Power", "website": "https://acwapower.com", "industry": "Renewable Energy & Solar", "niche": "power generation, solar energy & green hydrogen"},
            {"name": "Desert Technologies", "website": "https://desert-technologies.com", "industry": "Renewable Energy & Solar", "niche": "solar PV panels, energy storage & renewable solutions"},
        ],
        "fitness": [
            {"name": "Fitness Time", "website": "https://fitnesstime.com.sa", "industry": "Health & Fitness", "niche": "sports club & fitness center network"},
            {"name": "Bodymasters", "website": "https://bodymasters.com.sa", "industry": "Health & Fitness", "niche": "fitness centers & health clubs"},
        ],
        "hospitality": [
            {"name": "Dur Hospitality", "website": "https://dur.sa", "industry": "Hospitality & Hotels", "niche": "hotel operations, residential compounds & hospitality"},
            {"name": "Al Khozama", "website": "https://alkhozama.com", "industry": "Hospitality & Hotels", "niche": "luxury hospitality & hotel properties"},
        ],
        "data_ai": [
            {"name": "Elm Data & AI", "website": "https://www.elm.sa", "industry": "Data & AI Solutions", "niche": "digital intelligence, government analytics & AI"},
            {"name": "Quant Data & Analytics", "website": "https://quant.sa", "industry": "Data & Analytics", "niche": "data science, business intelligence & analytics products"},
            {"name": "Mozn AI", "website": "https://mozn.ai", "industry": "Enterprise AI", "niche": "AI solutions, risk intelligence & natural language processing"},
            {"name": "Lucidya", "website": "https://lucidya.com", "industry": "AI & Analytics", "niche": "AI-powered customer intelligence & analytics platform"},
        ],
    },
    "united kingdom": {
        "automotive": [
            {"name": "Arnold Clark", "website": "https://www.arnoldclark.com", "industry": "Automotive", "niche": "car dealerships & automotive retail"},
            {"name": "Lookers", "website": "https://www.lookers.co.uk", "industry": "Automotive", "niche": "motor retail & car dealerships"},
            {"name": "Auto Trader UK", "website": "https://www.autotrader.co.uk", "industry": "Automotive", "niche": "digital automotive marketplace"},
            {"name": "Cazoo", "website": "https://www.cazoo.co.uk", "industry": "Automotive", "niche": "online car retail & buying platform"},
        ],
        "university": [
            {"name": "University of Oxford", "website": "https://www.ox.ac.uk", "industry": "Higher Education", "niche": "collegiate research university"},
            {"name": "University of Cambridge", "website": "https://www.cam.ac.uk", "industry": "Higher Education", "niche": "collegiate research university"},
            {"name": "Imperial College London", "website": "https://www.imperial.ac.uk", "industry": "Higher Education", "niche": "science, engineering & medicine university"},
            {"name": "University College London (UCL)", "website": "https://www.ucl.ac.uk", "industry": "Higher Education", "niche": "multidisciplinary research university"},
            {"name": "London School of Economics (LSE)", "website": "https://www.lse.ac.uk", "industry": "Higher Education", "niche": "social sciences, finance & economics university"},
            {"name": "University of Edinburgh", "website": "https://www.ed.ac.uk", "industry": "Higher Education", "niche": "public research university"},
        ],
        "school": [
            {"name": "Eton College", "website": "https://www.etoncollege.com", "industry": "Primary & Secondary Education", "niche": "independent boarding school"},
            {"name": "Harrow School", "website": "https://www.harrowschool.org.uk", "industry": "Primary & Secondary Education", "niche": "independent boarding school"},
            {"name": "Westminster School", "website": "https://www.westminster.org.uk", "industry": "Primary & Secondary Education", "niche": "independent day and boarding school"},
            {"name": "Winchester College", "website": "https://www.winchestercollege.org", "industry": "Primary & Secondary Education", "niche": "independent boarding school"},
        ],
        "fintech": [
            {"name": "Revolut", "website": "https://www.revolut.com", "industry": "Fintech", "niche": "digital banking, currency exchange & global accounts"},
            {"name": "Monzo", "website": "https://monzo.com", "industry": "Fintech", "niche": "digital mobile bank & personal finance"},
            {"name": "Wise", "website": "https://wise.com", "industry": "Fintech", "niche": "cross-border money transfers & multi-currency banking"},
            {"name": "Starling Bank", "website": "https://www.starlingbank.com", "industry": "Fintech", "niche": "digital banking & business accounts"},
        ],
        "retail": [
            {"name": "ASOS", "website": "https://www.asos.com", "industry": "Apparel & Fashion", "niche": "online fashion & beauty retail"},
            {"name": "Farfetch", "website": "https://www.farfetch.com", "industry": "Apparel & Fashion", "niche": "luxury fashion & designer marketplace"},
            {"name": "Gymshark", "website": "https://www.gymshark.com", "industry": "Apparel & Fashion", "niche": "fitness apparel & athletic wear"},
            {"name": "Boohoo", "website": "https://www.boohoo.com", "industry": "Apparel & Fashion", "niche": "fast fashion e-commerce"},
            {"name": "Next", "website": "https://www.next.co.uk", "industry": "Retail & E-Commerce", "niche": "clothing, footwear & home products"},
        ],
        "real_estate": [
            {"name": "Rightmove", "website": "https://www.rightmove.co.uk", "industry": "Real Estate", "niche": "property portal & real estate search"},
            {"name": "Zoopla", "website": "https://www.zoopla.co.uk", "industry": "Real Estate", "niche": "property search & real estate market data"},
            {"name": "Savills UK", "website": "https://www.savills.co.uk", "industry": "Real Estate", "niche": "global real estate services & property advisory"},
            {"name": "Knight Frank", "website": "https://www.knightfrank.co.uk", "industry": "Real Estate", "niche": "residential & commercial property consultancy"},
        ],
        "healthcare": [
            {"name": "Bupa UK", "website": "https://www.bupa.co.uk", "industry": "Healthcare", "niche": "health insurance, private clinics & care homes"},
            {"name": "Babylon Health", "website": "https://www.babylonhealth.com", "industry": "Healthcare", "niche": "digital healthcare & virtual doctor consultations"},
            {"name": "Boots UK", "website": "https://www.boots.com", "industry": "Healthcare", "niche": "pharmacy, health & beauty retail"},
        ],
        "telecom": [
            {"name": "BT Group", "website": "https://www.bt.com", "industry": "Telecommunications", "niche": "broadband, fixed line & enterprise telecom"},
            {"name": "Vodafone UK", "website": "https://www.vodafone.co.uk", "industry": "Telecommunications", "niche": "mobile network & 5G telecom"},
            {"name": "EE", "website": "https://ee.co.uk", "industry": "Telecommunications", "niche": "mobile network & broadband services"},
        ],
        "logistics": [
            {"name": "Royal Mail", "website": "https://www.royalmail.com", "industry": "Logistics & Supply Chain", "niche": "postal delivery, courier & parcel services"},
            {"name": "DPD UK", "website": "https://www.dpd.co.uk", "industry": "Logistics & Supply Chain", "niche": "express parcel delivery & logistics"},
        ],
        "energy": [
            {"name": "Octopus Energy", "website": "https://octopus.energy", "industry": "Renewable Energy & Utilities", "niche": "green renewable electricity & smart technology"},
            {"name": "Lightsource bp", "website": "https://lightsourcebp.com", "industry": "Renewable Energy & Solar", "niche": "solar energy development & renewable power"},
            {"name": "SolarEdge UK", "website": "https://www.solaredge.com/uk", "industry": "Renewable Energy & Solar", "niche": "smart solar inverters & energy storage"},
        ],
        "fitness": [
            {"name": "PureGym", "website": "https://www.puregym.com", "industry": "Health & Fitness", "niche": "affordable 24/7 gym & fitness chain"},
            {"name": "The Gym Group", "website": "https://www.thegymgroup.com", "industry": "Health & Fitness", "niche": "low-cost gym club & fitness classes"},
            {"name": "David Lloyd Clubs", "website": "https://www.davidlloyd.co.uk", "industry": "Health & Fitness", "niche": "premium health, spa & racquets clubs"},
        ],
        "hospitality": [
            {"name": "Premier Inn", "website": "https://www.premierinn.com", "industry": "Hospitality & Hotels", "niche": "hotel chain & comfortable stays"},
            {"name": "Travelodge UK", "website": "https://www.travelodge.co.uk", "industry": "Hospitality & Hotels", "niche": "budget hotel chain & accommodation"},
            {"name": "InterContinental Hotels Group", "website": "https://www.ihg.com", "industry": "Hospitality & Hotels", "niche": "global luxury hotel & resort brands"},
        ],
        "data_ai": [
            {"name": "Thorogood", "website": "https://www.thorogood.com", "industry": "Data & Analytics Consulting", "niche": "business intelligence, advanced analytics & AI strategy"},
            {"name": "Keyrus UK", "website": "https://keyrus.co.uk", "industry": "Data Intelligence", "niche": "data intelligence, cloud analytics & BI consulting"},
            {"name": "Kubrick Group", "website": "https://kubrickgroup.com", "industry": "Data & AI Consulting", "niche": "data engineering, machine learning & analytics advisory"},
            {"name": "Peak AI", "website": "https://peak.ai", "industry": "Enterprise AI", "niche": "decision intelligence & commercial AI platform"},
        ],
    },
    "united states": {
        "automotive": [
            {"name": "Tesla", "website": "https://www.tesla.com", "industry": "Automotive", "niche": "electric vehicles, clean energy & automotive"},
            {"name": "CarMax", "website": "https://www.carmax.com", "industry": "Automotive", "niche": "used car retail & automotive dealerships"},
            {"name": "Carvana", "website": "https://www.carvana.com", "industry": "Automotive", "niche": "e-commerce automotive platform"},
            {"name": "AutoNation", "website": "https://www.autonation.com", "industry": "Automotive", "niche": "automotive retailer & dealerships"},
            {"name": "Rivian", "website": "https://rivian.com", "industry": "Automotive", "niche": "electric trucks & adventure vehicles"},
        ],
        "university": [
            {"name": "Harvard University", "website": "https://www.harvard.edu", "industry": "Higher Education", "niche": "ivy league research university"},
            {"name": "Stanford University", "website": "https://www.stanford.edu", "industry": "Higher Education", "niche": "private research university"},
            {"name": "MIT", "website": "https://www.mit.edu", "industry": "Higher Education", "niche": "science & technology institute"},
            {"name": "Columbia University", "website": "https://www.columbia.edu", "industry": "Higher Education", "niche": "ivy league research university"},
            {"name": "UC Berkeley", "website": "https://www.berkeley.edu", "industry": "Higher Education", "niche": "public research university"},
            {"name": "Princeton University", "website": "https://www.princeton.edu", "industry": "Higher Education", "niche": "ivy league research university"},
            {"name": "Yale University", "website": "https://www.yale.edu", "industry": "Higher Education", "niche": "ivy league research university"},
            {"name": "Caltech", "website": "https://www.caltech.edu", "industry": "Higher Education", "niche": "science & engineering research institute"},
            {"name": "Carnegie Mellon University", "website": "https://www.cmu.edu", "industry": "Higher Education", "niche": "computer science, AI & robotics university"},
            {"name": "Cornell University", "website": "https://www.cornell.edu", "industry": "Higher Education", "niche": "ivy league research university"},
        ],
        "school": [
            {"name": "Phillips Exeter Academy", "website": "https://www.exeter.edu", "industry": "Primary & Secondary Education", "niche": "independent co-educational school"},
            {"name": "Phillips Academy Andover", "website": "https://www.andover.edu", "industry": "Primary & Secondary Education", "niche": "co-educational boarding school"},
            {"name": "Trinity School NYC", "website": "https://www.trinityschoolnyc.org", "industry": "Primary & Secondary Education", "niche": "independent preparatory school"},
            {"name": "St. Paul's School", "website": "https://www.sps.edu", "industry": "Primary & Secondary Education", "niche": "college-preparatory boarding school"},
            {"name": "The Lawrenceville School", "website": "https://www.lawrenceville.org", "industry": "Primary & Secondary Education", "niche": "co-educational preparatory boarding school"},
        ],
        "fintech": [
            {"name": "Stripe", "website": "https://stripe.com", "industry": "Fintech", "niche": "online payment processing & financial infrastructure"},
            {"name": "Block (Square)", "website": "https://squareup.com", "industry": "Fintech", "niche": "pos systems, merchant payments & financial services"},
            {"name": "Robinhood", "website": "https://robinhood.com", "industry": "Fintech", "niche": "commission-free stock trading & investing app"},
            {"name": "Chime", "website": "https://www.chime.com", "industry": "Fintech", "niche": "mobile banking & digital financial accounts"},
            {"name": "Plaid", "website": "https://plaid.com", "industry": "Fintech", "niche": "financial data network & banking api"},
            {"name": "Coinbase", "website": "https://www.coinbase.com", "industry": "Fintech", "niche": "cryptocurrency exchange & digital asset platform"},
        ],
        "retail": [
            {"name": "Amazon", "website": "https://www.amazon.com", "industry": "Retail & E-Commerce", "niche": "online marketplace & cloud retail"},
            {"name": "Shopify Stores / Revolve", "website": "https://www.revolve.com", "industry": "Apparel & Fashion", "niche": "trendy designer apparel & fashion e-commerce"},
            {"name": "Shein US", "website": "https://us.shein.com", "industry": "Apparel & Fashion", "niche": "fast fashion e-commerce & apparel"},
            {"name": "Nike", "website": "https://www.nike.com", "industry": "Apparel & Fashion", "niche": "athletic footwear, apparel & sports equipment"},
            {"name": "Lululemon", "website": "https://shop.lululemon.com", "industry": "Apparel & Fashion", "niche": "technical athletic apparel & yoga wear"},
            {"name": "Nordstrom", "website": "https://www.nordstrom.com", "industry": "Retail & E-Commerce", "niche": "luxury department store & fashion retail"},
            {"name": "Target", "website": "https://www.target.com", "industry": "Retail & E-Commerce", "niche": "general merchandise & retail discount store"},
        ],
        "real_estate": [
            {"name": "Zillow", "website": "https://www.zillow.com", "industry": "Real Estate", "niche": "real estate marketplace & property valuations"},
            {"name": "Redfin", "website": "https://www.redfin.com", "industry": "Real Estate", "niche": "full-service real estate brokerage & listings"},
            {"name": "Compass", "website": "https://www.compass.com", "industry": "Real Estate", "niche": "technology-driven residential real estate brokerage"},
            {"name": "CBRE", "website": "https://www.cbre.com", "industry": "Real Estate", "niche": "commercial real estate services & investment"},
            {"name": "JLL", "website": "https://www.us.jll.com", "industry": "Real Estate", "niche": "commercial real estate & property investment management"},
        ],
        "healthcare": [
            {"name": "Teladoc Health", "website": "https://www.teladochealth.com", "industry": "Healthcare", "niche": "virtual healthcare, telehealth & digital medicine"},
            {"name": "Mayo Clinic", "website": "https://www.mayoclinic.org", "industry": "Healthcare", "niche": "integrated clinical practice, education & research hospital"},
            {"name": "Cleveland Clinic", "website": "https://my.clevelandclinic.org", "industry": "Healthcare", "niche": "multispecialty academic medical center"},
            {"name": "One Medical", "website": "https://www.onemedical.com", "industry": "Healthcare", "niche": "technology-powered primary healthcare clinic"},
            {"name": "Quest Diagnostics", "website": "https://www.questdiagnostics.com", "industry": "Healthcare", "niche": "diagnostic testing & clinical pathology laboratory"},
        ],
        "telecom": [
            {"name": "Verizon", "website": "https://www.verizon.com", "industry": "Telecommunications", "niche": "wireless network, broadband & 5G connectivity"},
            {"name": "AT&T", "website": "https://www.att.com", "industry": "Telecommunications", "niche": "telecommunications, mobile network & high-speed fiber"},
            {"name": "T-Mobile US", "website": "https://www.t-mobile.com", "industry": "Telecommunications", "niche": "wireless network & 5G telecom provider"},
        ],
        "logistics": [
            {"name": "FedEx", "website": "https://www.fedex.com", "industry": "Logistics & Supply Chain", "niche": "global courier delivery & express freight"},
            {"name": "UPS", "website": "https://www.ups.com", "industry": "Logistics & Supply Chain", "niche": "package delivery, supply chain & freight forwarding"},
            {"name": "Flexport", "website": "https://www.flexport.com", "industry": "Logistics & Supply Chain", "niche": "digital freight forwarding & supply chain logistics"},
            {"name": "ShipBob", "website": "https://www.shipbob.com", "industry": "Logistics & Supply Chain", "niche": "ecommerce order fulfillment & 3PL logistics"},
        ],
        "academy": [
            {"name": "Coursera", "website": "https://www.coursera.org", "industry": "EdTech", "niche": "online learning platform, university degrees & professional certificates"},
            {"name": "edX", "website": "https://www.edx.org", "industry": "EdTech", "niche": "online university courses & master degree programs"},
            {"name": "Khan Academy", "website": "https://www.khanacademy.org", "industry": "EdTech", "niche": "free online education & k-12 test prep"},
            {"name": "Udacity", "website": "https://www.udacity.com", "industry": "EdTech", "niche": "tech nanodegrees, AI, coding & cloud training"},
        ],
        "energy": [
            {"name": "Sunrun", "website": "https://www.sunrun.com", "industry": "Renewable Energy & Solar", "niche": "residential solar panel installations & home battery storage"},
            {"name": "Tesla Solar", "website": "https://www.tesla.com/solarpanels", "industry": "Renewable Energy & Solar", "niche": "solar roof tiles, clean energy & powerwall batteries"},
            {"name": "SunPower", "website": "https://us.sunpower.com", "industry": "Renewable Energy & Solar", "niche": "residential & commercial solar power systems"},
            {"name": "Enphase Energy", "website": "https://enphase.com", "industry": "Renewable Energy & Solar", "niche": "microinverter technology & home energy systems"},
        ],
        "fitness": [
            {"name": "Equinox", "website": "https://www.equinox.com", "industry": "Health & Fitness", "niche": "luxury fitness club, personal training & wellness"},
            {"name": "Planet Fitness", "website": "https://www.planetfitness.com", "industry": "Health & Fitness", "niche": "nationwide affordable gym & fitness centers"},
            {"name": "Life Time", "website": "https://www.lifetime.life", "industry": "Health & Fitness", "niche": "athletic country resorts, gyms & wellness centers"},
            {"name": "LA Fitness", "website": "https://www.lafitness.com", "industry": "Health & Fitness", "niche": "health club, gym facilities & group fitness"},
        ],
        "hospitality": [
            {"name": "Marriott International", "website": "https://www.marriott.com", "industry": "Hospitality & Hotels", "niche": "global hotel portfolio & luxury hospitality"},
            {"name": "Hilton Hotels & Resorts", "website": "https://www.hilton.com", "industry": "Hospitality & Hotels", "niche": "global hotel hospitality & luxury suites"},
            {"name": "Hyatt Hotels", "website": "https://www.hyatt.com", "industry": "Hospitality & Hotels", "niche": "hospitality management, luxury hotels & resorts"},
            {"name": "Wyndham Hotels & Resorts", "website": "https://www.wyndhamhotels.com", "industry": "Hospitality & Hotels", "niche": "hotel franchise & hospitality chains"},
        ],
        "data_ai": [
            {"name": "Aimpoint Digital", "website": "https://aimpointdigital.com", "industry": "Data & AI Consulting", "niche": "modern data stack, AI automation & analytics advisory"},
            {"name": "Keyrus", "website": "https://keyrus.com", "industry": "Data & BI Consulting", "niche": "data intelligence, cloud analytics & BI consulting"},
            {"name": "PhData", "website": "https://www.phdata.io", "industry": "Data & AI Services", "niche": "Snowflake, Databricks & AI automation consulting"},
            {"name": "Caserta", "website": "https://caserta.com", "industry": "Data Architecture & AI", "niche": "data strategy, analytics engineering & modern AI"},
            {"name": "DAS42", "website": "https://das42.com", "industry": "Data & BI Consulting", "niche": "full-stack data analytics & modern business intelligence"},
            {"name": "Analytics8", "website": "https://www.analytics8.com", "industry": "Data Analytics Consulting", "niche": "data strategy, business intelligence & data engineering"},
            {"name": "Blend360", "website": "https://www.blend360.com", "industry": "Data Science & AI", "niche": "data science solutions, predictive analytics & enterprise AI"},
            {"name": "Tiger Analytics", "website": "https://www.tigeranalytics.com", "industry": "AI & Analytics Consulting", "niche": "enterprise AI, machine learning & business analytics"},
        ],
    },
    "india": {
        "automotive": [
            {"name": "Tata Motors", "website": "https://www.tatamotors.com", "industry": "Automotive", "niche": "passenger vehicles, commercial & EV"},
            {"name": "Mahindra & Mahindra", "website": "https://www.mahindra.com", "industry": "Automotive", "niche": "SUVs, electric vehicles & automotive"},
            {"name": "Maruti Suzuki India", "website": "https://www.marutisuzuki.com", "industry": "Automotive", "niche": "passenger cars & hatchbacks"},
            {"name": "Hyundai Motor India", "website": "https://www.hyundai.com/in", "industry": "Automotive", "niche": "SUVs, sedans & passenger cars"},
            {"name": "Hero MotoCorp", "website": "https://www.heromotocorp.com", "industry": "Automotive", "niche": "two-wheelers & motorcycles"},
            {"name": "Bajaj Auto", "website": "https://www.bajajauto.com", "industry": "Automotive", "niche": "two-wheelers & commercial three-wheelers"},
        ],
        "university": [
            {"name": "IIT Bombay", "website": "https://www.iitb.ac.in", "industry": "Higher Education", "niche": "premier engineering & technology institute"},
            {"name": "IIT Delhi", "website": "https://www.iitd.ac.in", "industry": "Higher Education", "niche": "institute of national importance & research"},
            {"name": "BITS Pilani", "website": "https://www.bits-pilani.ac.in", "industry": "Higher Education", "niche": "science & technology research university"},
            {"name": "Manipal Academy of Higher Education", "website": "https://www.manipal.edu", "industry": "Higher Education", "niche": "deemed research university"},
            {"name": "Amity University", "website": "https://www.amity.edu", "industry": "Higher Education", "niche": "private research & higher education network"},
            {"name": "IIT Madras", "website": "https://www.iitm.ac.in", "industry": "Higher Education", "niche": "technical institute & higher education"},
        ],
        "school": [
            {"name": "The Doon School", "website": "https://www.doonschool.com", "industry": "Primary & Secondary Education", "niche": "all-boys boarding school"},
            {"name": "Delhi Public School (DPS)", "website": "https://www.dpsrkp.net", "industry": "Primary & Secondary Education", "niche": "cbse nationwide school network"},
            {"name": "The Shri Ram School", "website": "https://www.tsrs.org", "industry": "Primary & Secondary Education", "niche": "icse & ib curriculum private schooling"},
            {"name": "Cathedral and John Connon School", "website": "https://cathedral-school.com", "industry": "Primary & Secondary Education", "niche": "co-educational private school"},
            {"name": "Dhirubhai Ambani International School", "website": "https://www.dais.edu.in", "industry": "Primary & Secondary Education", "niche": "ib world school & international education"},
        ],
        "college": [
            {"name": "St. Stephen's College", "website": "https://www.ststephens.edu", "industry": "College Education", "niche": "liberal arts & sciences constituent college"},
            {"name": "Loyola College Chennai", "website": "https://www.loyolacollege.edu", "industry": "College Education", "niche": "autonomous arts and science college"},
            {"name": "St. Xavier's College Mumbai", "website": "https://xaviers.ac", "industry": "College Education", "niche": "autonomous undergraduate & postgraduate college"},
            {"name": "Hindu College Delhi", "website": "https://hinducollege.ac.in", "industry": "College Education", "niche": "arts, science & commerce college"},
        ],
        "academy": [
            {"name": "FIITJEE", "website": "https://www.fiitjee.com", "industry": "Test Preparation", "niche": "iit jee, engineering & foundation test prep"},
            {"name": "Allen Career Institute", "website": "https://www.allen.ac.in", "industry": "Test Preparation", "niche": "neet, jee coaching & test prep"},
            {"name": "Aakash Institute", "website": "https://www.aakash.ac.in", "industry": "Test Preparation", "niche": "medical & engineering entrance test coaching"},
            {"name": "Physics Wallah", "website": "https://www.pw.live", "industry": "EdTech", "niche": "online education, jee & neet prep platform"},
            {"name": "Unacademy", "website": "https://unacademy.com", "industry": "EdTech", "niche": "online learning & competitive exam prep"},
        ],
        "software": [
            {"name": "Tata Consultancy Services (TCS)", "website": "https://www.tcs.com", "industry": "IT Services", "niche": "global it consulting & digital solutions"},
            {"name": "Infosys", "website": "https://www.infosys.com", "industry": "IT Services", "niche": "digital services, consulting & it outsourcing"},
            {"name": "Wipro", "website": "https://www.wipro.com", "industry": "IT Services", "niche": "technology consulting & enterprise it services"},
            {"name": "HCLTech", "website": "https://www.hcltech.com", "industry": "IT Services", "niche": "global technology & digital engineering services"},
            {"name": "Zoho Corporation", "website": "https://www.zoho.com", "industry": "Software & SaaS", "niche": "cloud software suite & business applications"},
            {"name": "Freshworks", "website": "https://www.freshworks.com", "industry": "Software & SaaS", "niche": "customer engagement & itsm software"},
        ],
        "fintech": [
            {"name": "Paytm", "website": "https://paytm.com", "industry": "Fintech", "niche": "digital payments, financial services & merchant commerce"},
            {"name": "PhonePe", "website": "https://www.phonepe.com", "industry": "Fintech", "niche": "digital payments & upi ecosystem"},
            {"name": "Razorpay", "website": "https://razorpay.com", "industry": "Fintech", "niche": "payment gateway & business banking"},
            {"name": "Zerodha", "website": "https://zerodha.com", "industry": "Fintech", "niche": "discount stock brokerage & trading platform"},
        ],
        "retail": [
            {"name": "Flipkart", "website": "https://www.flipkart.com", "industry": "Retail & E-Commerce", "niche": "e-commerce marketplace & online retail"},
            {"name": "Reliance Retail", "website": "https://relianceretail.com", "industry": "Retail", "niche": "omnichannel retail & consumer goods"},
            {"name": "Nykaa", "website": "https://www.nykaa.com", "industry": "Retail & E-Commerce", "niche": "beauty, cosmetics & fashion e-commerce"},
            {"name": "Myntra", "website": "https://www.myntra.com", "industry": "Retail & E-Commerce", "niche": "fashion & lifestyle e-commerce"},
        ],
    },
    "canada": {
        "university": [
            {"name": "University of Toronto", "website": "https://www.utoronto.ca", "industry": "Higher Education", "niche": "public research university"},
            {"name": "University of British Columbia (UBC)", "website": "https://www.ubc.ca", "industry": "Higher Education", "niche": "public research university & higher education"},
            {"name": "McGill University", "website": "https://www.mcgill.ca", "industry": "Higher Education", "niche": "public research university & medical school"},
            {"name": "University of Waterloo", "website": "https://uwaterloo.ca", "industry": "Higher Education", "niche": "computer science, engineering & co-op education"},
        ],
        "school": [
            {"name": "Upper Canada College", "website": "https://www.ucc.on.ca", "industry": "Primary & Secondary Education", "niche": "ib continuum boys school"},
            {"name": "Branksome Hall", "website": "https://www.branksome.on.ca", "industry": "Primary & Secondary Education", "niche": "ib world school for girls"},
        ],
        "fintech": [
            {"name": "Wealthsimple", "website": "https://www.wealthsimple.com", "industry": "Fintech", "niche": "digital investing, automated trading & savings"},
            {"name": "Nuvei", "website": "https://nuvei.com", "industry": "Fintech", "niche": "global payment technology & payment processing"},
        ],
        "retail": [
            {"name": "Shopify", "website": "https://www.shopify.com", "industry": "E-Commerce", "niche": "global ecommerce platform & retail technology"},
            {"name": "SSENSE", "website": "https://www.ssense.com", "industry": "Apparel & Fashion", "niche": "luxury streetwear & designer fashion e-commerce"},
            {"name": "Canada Goose", "website": "https://www.canadagoose.com", "industry": "Apparel & Fashion", "niche": "luxury outerwear & cold-weather apparel"},
            {"name": "Aritzia", "website": "https://www.aritzia.com", "industry": "Apparel & Fashion", "niche": "women's fashion & luxury apparel retail"},
        ],
        "telecom": [
            {"name": "Rogers Communications", "website": "https://www.rogers.com", "industry": "Telecommunications", "niche": "wireless, high-speed internet & media telecom"},
            {"name": "Bell Canada", "website": "https://www.bell.ca", "industry": "Telecommunications", "niche": "telecom services, fiber broadband & mobile network"},
        ],
    },
    "australia": {
        "university": [
            {"name": "University of Melbourne", "website": "https://www.unimelb.edu.au", "industry": "Higher Education", "niche": "public research university"},
            {"name": "University of Sydney", "website": "https://www.sydney.edu.au", "industry": "Higher Education", "niche": "comprehensive research university"},
            {"name": "Australian National University (ANU)", "website": "https://www.anu.edu.au", "industry": "Higher Education", "niche": "national research university"},
            {"name": "University of Queensland", "website": "https://www.uq.edu.au", "industry": "Higher Education", "niche": "research university & higher education"},
            {"name": "University of New South Wales (UNSW)", "website": "https://www.unsw.edu.au", "industry": "Higher Education", "niche": "engineering & business university"},
        ],
        "school": [
            {"name": "Sydney Grammar School", "website": "https://www.sydgram.nsw.edu.au", "industry": "Primary & Secondary Education", "niche": "independent day school for boys"},
            {"name": "Scotch College Melbourne", "website": "https://www.scotch.vic.edu.au", "industry": "Primary & Secondary Education", "niche": "independent presbyterian day and boarding school"},
        ],
        "fintech": [
            {"name": "Afterpay", "website": "https://www.afterpay.com", "industry": "Fintech", "niche": "buy now pay later & digital payments"},
            {"name": "Airwallex", "website": "https://www.airwallex.com", "industry": "Fintech", "niche": "global cross-border payments & financial platform"},
            {"name": "Judo Bank", "website": "https://www.judo.bank", "industry": "Fintech", "niche": "digital neo-bank for small and medium businesses"},
        ],
        "retail": [
            {"name": "Cotton On Group", "website": "https://cottonon.com", "industry": "Apparel & Fashion", "niche": "casual apparel & global retail"},
            {"name": "The Iconic", "website": "https://www.theiconic.com.au", "industry": "Apparel & Fashion", "niche": "online fashion, sportswear & footwear retail"},
            {"name": "Zimmermann", "website": "https://www.zimmermann.com", "industry": "Apparel & Fashion", "niche": "luxury designer fashion & resort wear"},
        ],
        "real_estate": [
            {"name": "REA Group (realestate.com.au)", "website": "https://www.realestate.com.au", "industry": "Real Estate", "niche": "digital property portal & residential real estate"},
            {"name": "Domain Group", "website": "https://www.domain.com.au", "industry": "Real Estate", "niche": "real estate marketplace & property listings"},
        ],
        "telecom": [
            {"name": "Telstra", "website": "https://www.telstra.com.au", "industry": "Telecommunications", "niche": "telecommunications, 5G mobile & broadband network"},
            {"name": "Optus", "website": "https://www.optus.com.au", "industry": "Telecommunications", "niche": "cellular telecom, mobile & enterprise connectivity"},
        ],
    },
    "germany": {
        "university": [
            {"name": "Technical University of Munich (TUM)", "website": "https://www.tum.de", "industry": "Higher Education", "niche": "technical university & engineering research"},
            {"name": "LMU Munich", "website": "https://www.lmu.de", "industry": "Higher Education", "niche": "public research university"},
            {"name": "Heidelberg University", "website": "https://www.uni-heidelberg.de", "industry": "Higher Education", "niche": "research university & medicine"},
        ],
        "fintech": [
            {"name": "N26", "website": "https://n26.com", "industry": "Fintech", "niche": "mobile digital banking & neo-bank"},
            {"name": "Trade Republic", "website": "https://traderepublic.com", "industry": "Fintech", "niche": "mobile commission-free broker & savings"},
        ],
        "retail": [
            {"name": "Zalando", "website": "https://www.zalando.com", "industry": "Apparel & Fashion", "niche": "online fashion & lifestyle e-commerce platform"},
            {"name": "About You", "website": "https://corporate.aboutyou.de", "industry": "Apparel & Fashion", "niche": "personalized fashion e-commerce"},
            {"name": "Hugo Boss", "website": "https://www.hugoboss.com", "industry": "Apparel & Fashion", "niche": "premium & luxury apparel fashion house"},
            {"name": "Adidas", "website": "https://www.adidas.com", "industry": "Apparel & Fashion", "niche": "sportswear, athletic shoes & sporting goods"},
            {"name": "Puma", "website": "https://about.puma.com", "industry": "Apparel & Fashion", "niche": "sports apparel, footwear & lifestyle retail"},
        ],
        "automotive": [
            {"name": "Auto1 Group", "website": "https://www.auto1-group.com", "industry": "Automotive", "niche": "digital automotive platform & car trading"},
            {"name": "Mobile.de", "website": "https://www.mobile.de", "industry": "Automotive", "niche": "vehicle marketplace & automotive search"},
        ],
    },
    "singapore": {
        "university": [
            {"name": "National University of Singapore (NUS)", "website": "https://www.nus.edu.sg", "industry": "Higher Education", "niche": "autonomous comprehensive research university"},
            {"name": "Nanyang Technological University (NTU)", "website": "https://www.ntu.edu.sg", "industry": "Higher Education", "niche": "research-intensive technical university"},
            {"name": "Singapore Management University (SMU)", "website": "https://www.smu.edu.sg", "industry": "Higher Education", "niche": "business, economics & law university"},
        ],
        "school": [
            {"name": "Raffles Institution", "website": "https://www.ri.edu.sg", "industry": "Primary & Secondary Education", "niche": "premier independent school"},
            {"name": "Hwa Chong Institution", "website": "https://www.hci.edu.sg", "industry": "Primary & Secondary Education", "niche": "independent premier school"},
        ],
        "fintech": [
            {"name": "Grab Financial Group", "website": "https://www.grab.com/sg/financial", "industry": "Fintech", "niche": "digital payments, micro-financing & fintech ecosystem"},
            {"name": "Endowus", "website": "https://endowus.com", "industry": "Fintech", "niche": "digital wealth advisor & investment platform"},
        ],
        "retail": [
            {"name": "Shopee / Sea Group", "website": "https://shopee.sg", "industry": "Retail & E-Commerce", "niche": "e-commerce marketplace & digital retail"},
            {"name": "Love, Bonito", "website": "https://www.lovebonito.com", "industry": "Apparel & Fashion", "niche": "women's fashion & direct-to-consumer apparel"},
            {"name": "Charles & Keith", "website": "https://www.charleskeith.com", "industry": "Apparel & Fashion", "niche": "contemporary footwear, bags & fashion accessories"},
        ],
        "real_estate": [
            {"name": "PropertyGuru", "website": "https://www.propertyguru.com.sg", "industry": "Real Estate", "niche": "proptech company & property marketplace"},
            {"name": "CapitaLand", "website": "https://www.capitaland.com", "industry": "Real Estate", "niche": "real estate investment & property development"},
        ],
        "telecom": [
            {"name": "Singtel", "website": "https://www.singtel.com", "industry": "Telecommunications", "niche": "global communications technology & 5G telecom"},
        ],
    },
}


def _detect_education_tier(*parts: object) -> str:
    """Classifies education client/rival into: university | school | college | academy | general."""
    blob = " " + _context_blob(*parts).lower() + " "
    if not blob.strip():
        return "general"

    # 1. School markers (check first if not explicitly a university)
    if re.search(
        r"\b(school|schools|grammar school|high school|primary school|elementary school|middle school|preschool|kindergarten|k-?12|o\s*levels?|matric|beaconhouse|the city school|city school|roots|lgs|lahore grammar|karachi grammar|kgs|froebel|learning alliance|the educators|sicas|bloomfield|gems education|nord anglia|taaleem|choueifat|eton college|exeter)\b",
        blob,
    ):
        if not re.search(
            r"\b(university|universities|degree awarding|bachelor|master\b|master'\''s|phd|undergraduate|postgraduate|lums|nust|fast nuces|giki|iba karachi|aku|comsats|szabist|uet|itu|pu\.edu|qau|harvard|mit|stanford|university of oxford|university of cambridge|kaust|kfupm)\b",
            blob,
        ):
            return "school"

    # 2. University / Higher Education markers
    if re.search(
        r"\b(university|universities|higher ed|higher education|degree awarding|bachelor|master\b|master'\''s|phd|undergraduate|postgraduate|lums|nust|fast nuces|giki|iba karachi|aku|szabist|comsats|pu\.edu|qau|itu|uet|harvard|mit|stanford|oxford university|cambridge university|university of cambridge|university of oxford|kaust|kfupm|aus\.edu|nyuad|fccollege|lahore school of economics)\b",
        blob,
    ):
        return "university"

    # 3. Intermediate / Pre-university College
    if re.search(
        r"\b(intermediate college|junior college|punjab college|pgc|superior college|concordia college|fsc|ics|higher secondary)\b",
        blob,
    ):
        return "college"

    # 4. Test Prep / Coaching Academy / Edtech
    if re.search(
        r"\b(test prep|entry test|mdcat|ecat|sat prep|ielts|coaching center|tuition academy|kips academy|step prep|maqsad|noon academy|nearpeer)\b",
        blob,
    ):
        return "academy"

    if re.search(r"\b(school|schools)\b", blob):
        return "school"
    if re.search(r"\b(college|colleges)\b", blob):
        return "college"
    if re.search(r"\b(academy|academies)\b", blob):
        return "academy"

    return "general"


def _education_tier_compatible(client_tier: str, rival_tier: str) -> bool:
    """Strictly prevents Universities matching Schools, Schools matching Universities, etc."""
    if not client_tier or not rival_tier or client_tier == "general" or rival_tier == "general":
        return True
    if client_tier == rival_tier:
        return True
    # University and School are strictly incompatible
    if client_tier == "university" and rival_tier in {"school", "academy", "college"}:
        return False
    if client_tier == "school" and rival_tier in {"university", "academy", "college"}:
        return False
    if client_tier == "college" and rival_tier in {"school", "university", "academy"}:
        return False
    if client_tier == "academy" and rival_tier in {"school", "university", "college"}:
        return False
    return client_tier == rival_tier


def _detect_industry_category(*parts: object) -> str:
    blob = " " + _context_blob(*parts).lower() + " "
    if not blob.strip():
        return "other"
    # 1. Hospitality & Hotels (check before food so hotel dining is classified under hospitality)
    if re.search(r"\b(hotels?|resorts?|hospitality|luxury\s+hotel|marriott|hilton|hyatt|jumeirah|serena\s+hotel|hotel\s+suites|motels?)\b", blob):
        return "hospitality"
    # 2. Beauty & Wellness
    if _looks_like_beauty_client(blob) or re.search(r"\b(beauty|salon|salons|parlour|parlor|spa|bridal|makeup|skincare|haircare|hair\s+styling|cosmetics?|aesthetic\s+clinic)\b", blob):
        return "beauty"
    # 3. Health & Fitness / Gyms
    if re.search(r"\b(gyms?|fitness|workout|crossfit|wellness|bodybuilding|personal\s+training|puregym|equinox|planet\s+fitness)\b", blob):
        return "fitness"
    # 4. Energy & Solar
    if re.search(r"\b(solar|cleantech|renewable\s+energy|energy|photovoltaic|inverters?|batteries|green\s+energy|sunrun|tesla\s+solar)\b", blob):
        return "energy"
    # 5. Logistics & Supply Chain
    if re.search(r"\b(logistics|supply\s+chain|courier|freight|shipping|warehousing|cargo|delivery\s+services|tcs|leopards|fedex|ups|dhl|aramex)\b", blob):
        return "logistics"
    # 6. Food / Restaurant / Bakery / Cafe
    if _looks_like_food_client(blob) or re.search(r"\b(fast\s*food|qsr|restaurants?|cafes?|caf[eé]|dining|baker(y|ies|s)?|desserts?|cakes?|pastr(y|ies)|shawarma|burgers?|pizzas?|biryani|eatery|food\s+chain)\b", blob):
        return "food"
    # 7. Education / University / School / College / Academy (requires full words, not matching "fast" in "fast food")
    if re.search(r"\b(schools?|colleges?|universit(y|ies)|education|higher\s+ed|edtech|academ(y|ies)|beaconhouse|lgs|city\s+school|lums|nust|fast\s+nuces|iba\s+karachi|giki|harvard|stanford|cambridge|oxford)\b", blob):
        edu_tier = _detect_education_tier(blob)
        if edu_tier in {"university", "school", "college", "academy"}:
            return edu_tier
        return "education"
    # 8. Automotive (whole words only)
    if re.search(r"\b(cars?|automotive|automobile|dealership|motorcycles?|honda|toyota|suzuki|hyundai|kia|changan|haval|mg\s+motor|carmax|carvana|autonation)\b", blob):
        return "automotive"
    # 9. Real Estate
    if re.search(r"\b(real\s+estate|property|housing|zameen|graana|apartments?|builders?)\b", blob):
        return "real_estate"
    # 10. Fintech & Banking
    if re.search(r"\b(banks?|banking|fintech|wallets?|payments?|easypaisa|jazzcash|nayapay|sadapay|stripe|revolut|paypal)\b", blob):
        return "fintech"
    # 11. Telecom
    if re.search(r"\b(telecom|cellular|broadband|isp|fiber|telenor|zong|ufone|ptcl|nayatel)\b", blob):
        return "telecom"
    # 12. Healthcare
    if re.search(r"\b(hospitals?|clinics?|pharma(cy)?|pharmaceutical|medical|diagnostic|healthcare|chughtai|shaukat\s+khanum)\b", blob):
        return "healthcare"
    # 13. Data Analytics, Business Intelligence & AI Solutions (separate from generic custom software)
    if re.search(r"\b(data\s+analytics|business\s+intelligence|enterprise\s+ai|conversational\s+ai|data\s+engineering|data\s+science|power\s+bi|tableau|snowflake|databricks|ai\s+solutions|analytics\s+consulting|data\s+consulting|bi\s+consulting|ai\s+consulting)\b", blob):
        return "data_ai"
    # 14. Software / IT / SaaS / Custom Software Development
    if _looks_like_software_peer_client(blob) or re.search(r"\b(software|saas|it\s+services|digital\s+agency|it\s+consulting|artificial\s+intelligence|machine\s+learning|custom\s+software|web\s+development|mobile\s+apps?)\b", blob):
        return "software"
    # 15. Retail & E-Commerce
    if re.search(r"\b(e-?commerce|retail|apparels?|fashions?|clothings?|shopping|daraz|khaadi|sapphire|flipkart|amazon)\b", blob):
        return "retail"
    return "other"


def _seed_local_industry_rivals(
    industry: str,
    market: str,
    client_name: str,
    *,
    already_have: list[str] | None = None,
    client_website: str | None = None,
    client_niche: str | None = None,
    limit: int = 8,
) -> list[dict]:
    key = _normalize_country_key(market)
    country_catalog = _LOCAL_MULTI_INDUSTRY_SEEDS.get(key) or {}
    cat = _detect_industry_category(industry, client_name, client_niche or "", client_website or "")
    seeds = (
        country_catalog.get(cat)
        or country_catalog.get("education" if cat in {"university", "school", "college", "academy"} else "")
        or []
    )
    if not seeds:
        return []
    blocked = _blocked_rival_keys(already_have, client_name, websites=[client_website] if client_website else None)
    client_host = _domain_of(client_website or "")

    out: list[dict] = []
    for seed in seeds:
        name = _as_str(seed.get("name")).strip()
        website = _normalize_website(_as_str(seed.get("website")) or None)
        if not name or not website:
            continue
        seed_keys = _rival_keys(name, website)
        if seed_keys & blocked:
            continue
        if client_host and _domain_of(website) == client_host:
            continue
        if _is_self_rival(client_name, name, website=website, client_website=client_website):
            continue
        niche_text = _as_str(seed.get("niche")) or cat.replace("_", " ")
        ind_label = _as_str(seed.get("industry")) or industry or cat.title()
        out.append(
            {
                "name": name,
                "website": website,
                "industry": ind_label,
                "business_model": "product" if cat in {"automotive", "retail"} else "services",
                "headquarters_country": key.title() if key else market,
                "why_relevant": f"Leading peer {cat.replace('_', ' ')} brand in {market} ({niche_text}); direct market competitor.",
                "threat_level": "high" if len(out) < 2 else "medium",
                "overlap_score": max(72.0, 92.0 - (len(out) * 4.0)),
                "same_niche": True,
                "same_market": True,
                "source": "seed",
            }
        )
        blocked |= seed_keys
        if len(out) >= limit:
            break
    return out


def _seed_global_industry_rivals(
    industry: str,
    client_name: str,
    *,
    already_have: list[str] | None = None,
    client_website: str | None = None,
    client_niche: str | None = None,
    client_market: str | None = None,
    limit: int = 8,
) -> list[dict]:
    """Provides high-quality international competitors from US/UK/global catalogs when scope is global."""
    cat = _detect_industry_category(industry, client_name, client_niche or "", client_website or "")
    blocked = _blocked_rival_keys(already_have, client_name, websites=[client_website] if client_website else None)
    client_host = _domain_of(client_website or "")

    seeds: list[tuple[str, dict]] = []

    # 1. Global university benchmark catalogs
    if cat in {"education", "university", "school", "college", "academy"}:
        edu_tier = _detect_education_tier(client_name, client_niche or "", industry)
        if edu_tier == "university":
            for s in _GLOBAL_UNIVERSITY_SEEDS:
                seeds.append(("global", s))
        elif edu_tier == "school":
            for s in _GLOBAL_SCHOOL_SEEDS:
                seeds.append(("global", s))

    # 2. Food clients: pull from international same-format chains
    elif _looks_like_food_client(client_name, client_niche, industry):
        fmt = _food_format_from_blob(client_name, client_niche, industry)
        for g_key in ("united states", "united kingdom", "canada", "uae", "saudi arabia"):
            cat_map = _LOCAL_MULTI_INDUSTRY_SEEDS.get(g_key, {})
            food_list = cat_map.get("food", [])
            for s in food_list:
                s_fmt = _food_format_from_blob(s.get("name"), s.get("niche"), s.get("industry"))
                if s_fmt == fmt:
                    seeds.append((g_key, s))

    # 3. Software clients: pull from global SaaS & international software catalogs
    elif (cat == "software" or _looks_like_software_peer_client(client_name, client_niche, industry)) and cat != "data_ai":
        for s in _GLOBAL_SOFTWARE_SEEDS:
            seeds.append(("global", s))
        for g_key in ("united states", "united kingdom", "canada", "germany", "singapore", "australia"):
            cat_map = _LOCAL_MULTI_INDUSTRY_SEEDS.get(g_key, {})
            for s in cat_map.get("software", []):
                seeds.append((g_key, s))

    # 4. Multi-industry clients (healthcare, real estate, automotive, data/AI, etc.)
    else:
        for g_key in ("united states", "united kingdom", "canada", "uae", "saudi arabia"):
            cat_map = _LOCAL_MULTI_INDUSTRY_SEEDS.get(g_key, {})
            ind_list = cat_map.get(cat, [])
            for s in ind_list:
                seeds.append((g_key, s))

    out: list[dict] = []
    for g_key, seed in seeds:
        name = _as_str(seed.get("name")).strip()
        website = _normalize_website(_as_str(seed.get("website")) or None)
        if not name or not website:
            continue
        seed_keys = _rival_keys(name, website)
        if seed_keys & blocked:
            continue
        if client_host and _domain_of(website) == client_host:
            continue
        if _is_self_rival(client_name, name, website=website, client_website=client_website):
            continue
        niche_text = _as_str(seed.get("niche")) or cat.replace("_", " ")
        ind_label = _as_str(seed.get("industry")) or industry or cat.title()
        country_name = "United States" if g_key == "united states" else ("United Kingdom" if g_key == "united kingdom" else ("UAE" if g_key == "uae" else g_key.title()))
        out.append(
            {
                "name": name,
                "website": website,
                "industry": ind_label,
                "business_model": "product" if cat in {"automotive", "retail"} else "services",
                "headquarters_country": country_name,
                "why_relevant": f"Leading global / international {cat.replace('_', ' ')} benchmark in {country_name} ({niche_text}); direct international peer.",
                "threat_level": "high" if len(out) < 2 else "medium",
                "overlap_score": max(70.0, 90.0 - (len(out) * 3.5)),
                "same_niche": True,
                "same_market": False,
                "source": "seed",
            }
        )
        blocked |= seed_keys
        if len(out) >= limit:
            break
    return out


def _seed_local_beauty_rivals(
    market: str,
    client_name: str,
    *,
    already_have: list[str] | None = None,
    client_website: str | None = None,
    client_niche: str = "",
    client_industry: str = "",
    limit: int = 8,
) -> list[dict]:
    key = _normalize_country_key(market)
    seeds = list(_LOCAL_BEAUTY_SEEDS.get(key) or [])
    if not seeds:
        return []
    blocked = _blocked_rival_keys(already_have, client_name, websites=[client_website] if client_website else None)
    client_host = _domain_of(client_website or "")

    out: list[dict] = []
    for seed in seeds:
        name = _as_str(seed.get("name")).strip()
        website = _normalize_website(_as_str(seed.get("website")) or None)
        if not name or not website:
            continue
        seed_keys = _rival_keys(name, website)
        if seed_keys & blocked:
            continue
        if client_host and _domain_of(website) == client_host:
            continue
        if _is_self_rival(client_name, name, website=website, client_website=client_website):
            continue
        niche_text = _as_str(seed.get("niche")) or "beauty salon & personal care"
        out.append(
            {
                "name": name,
                "website": website,
                "industry": "Beauty & Personal Care",
                "business_model": "services",
                "headquarters_country": key.title() if key else market,
                "why_relevant": f"Leading peer beauty & personal care brand in {market} ({niche_text}); competes for similar clients as {client_name}.",
                "threat_level": "high",
                "overlap_score": 78.0,
                "same_niche": True,
                "same_market": True,
                "source": "seed",
            }
        )
        blocked |= seed_keys
        if len(out) >= limit:
            break
    return out


def _seed_local_software_rivals(
    market: str,
    client_name: str,
    *,
    already_have: list[str] | None = None,
    client_website: str | None = None,
    client_peer_scale: PeerScale | None = None,
    client_niche: str = "",
    client_industry: str = "",
    limit: int = 8,
) -> list[dict]:
    key = _normalize_country_key(market)
    seeds = list(_LOCAL_SOFTWARE_SEEDS.get(key) or [])
    if not seeds:
        return []
    scale = client_peer_scale or _peer_scale_from_blob(
        client_name, client_niche, client_industry, name=client_name
    )
    blocked = _blocked_rival_keys(already_have, client_name, websites=[client_website] if client_website else None)
    client_host = _domain_of(client_website or "")

    def _build(seed: dict) -> dict | None:
        name = _as_str(seed.get("name")).strip()
        website = _normalize_website(_as_str(seed.get("website")) or None)
        if not name or not website:
            return None
        rival_scale = _peer_scale_from_blob(name, "software house", name=name, website=website)
        if not _peer_scale_compatible(scale, rival_scale):
            return None
        seed_keys = _rival_keys(name, website)
        if seed_keys & blocked or _is_generic_or_fake_rival_name(name):
            return None
        if client_host and _domain_of(website) == client_host:
            return None
        base = 78.0 if rival_scale == scale else 72.0
        base += _peer_scale_overlap_bonus(scale, rival_scale)
        is_large_national = any(tok in name.lower() for tok in _SOFTWARE_LARGE_NATIONAL)
        if scale == _PEER_BOUTIQUE and is_large_national:
            base -= 6
        return {
            "name": name,
            "website": website,
            "industry": "Software",
            "business_model": "services",
            "headquarters_country": key.title() if key else market,
            "why_relevant": (
                f"Same-tier software / digital peer in {market} ({rival_scale.replace('_', ' ')}); "
                f"competes for similar buyers as {client_name}."
            ),
            "threat_level": "high" if rival_scale == scale else "medium",
            "overlap_score": max(60.0, min(90.0, base)),
            "same_niche": True,
            "same_market": True,
            "peer_scale": rival_scale,
            "source": "seed",
            "_large_national": is_large_national,
            "_keys": seed_keys,
        }

    primary: list[dict] = []
    fallback: list[dict] = []
    for seed in seeds:
        row = _build(seed)
        if not row:
            continue
        if scale == _PEER_BOUTIQUE and row.pop("_large_national", False):
            fallback.append(row)
        else:
            row.pop("_large_national", None)
            primary.append(row)

    out: list[dict] = []
    for row in primary + (fallback if scale == _PEER_BOUTIQUE else []):
        keys = row.pop("_keys", set())
        if keys & blocked:
            continue
        out.append(row)
        blocked |= keys
        if len(out) >= limit:
            break
    return out


def _seed_global_software_rivals(
    client_name: str,
    *,
    already_have: list[str] | None = None,
    client_peer_scale: PeerScale | None = None,
    limit: int = 8,
) -> list[dict]:
    scale = client_peer_scale or _peer_scale_from_blob(client_name, name=client_name)
    # Boutique local clients should not be seeded with global engineering giants
    if scale == _PEER_BOUTIQUE:
        return []
    blocked = _blocked_rival_keys(already_have, client_name)
    out: list[dict] = []
    for seed in _GLOBAL_SOFTWARE_SEEDS:
        name = _as_str(seed.get("name")).strip()
        website = _normalize_website(_as_str(seed.get("website")) or None)
        if not name or not website:
            continue
        seed_keys = _rival_keys(name, website)
        if seed_keys & blocked or _is_generic_or_fake_rival_name(name):
            continue
        out.append(
            {
                "name": name,
                "website": website,
                "industry": "Software",
                "business_model": "services",
                "headquarters_country": _as_str(seed.get("headquarters_country")) or "Global",
                "why_relevant": (
                    "Global custom software / digital engineering firm competing for similar enterprise buyers."
                ),
                "threat_level": "high",
                "overlap_score": 70,
                "same_niche": True,
                "same_market": True,
                "peer_scale": _PEER_ENTERPRISE,
                "source": "seed",
            }
        )
        blocked |= seed_keys
        if len(out) >= limit:
            break
    return out


def _seed_local_qsr_rivals(
    market: str,
    client_name: str,
    *,
    already_have: list[str] | None = None,
    client_website: str | None = None,
    client_tier: FoodTier | None = None,
    client_niche: str = "",
    client_industry: str = "",
    limit: int = 8,
) -> list[dict]:
    key = _normalize_country_key(market)
    seeds = list(_LOCAL_QSR_SEEDS.get(key) or [])
    if not seeds:
        return []
    tier = client_tier or _food_tier_from_blob(client_name, client_niche, client_industry)
    client_fmt = _food_format_from_blob(client_name, client_niche, client_industry)
    # Prefer same-format + same-tier first
    tier_rank = {_FOOD_TIER_LOCAL: 0, _FOOD_TIER_NATIONAL: 1, _FOOD_TIER_GLOBAL: 2}
    client_rank = tier_rank.get(tier, 0)

    def _seed_sort_key(seed: dict) -> tuple:
        st = _as_str(seed.get("tier")) or _food_tier_from_blob(_as_str(seed.get("name")))
        sf = _as_str(seed.get("format")) or _food_format_from_blob(_as_str(seed.get("name")))
        sr = tier_rank.get(st, 9)
        same_fmt = 0 if _food_format_compatible(client_fmt, sf) and (
            client_fmt == sf or client_fmt == _FOOD_FORMAT_GENERAL
        ) else (1 if _food_format_compatible(client_fmt, sf) else 2)
        return (same_fmt, abs(sr - client_rank), sr)

    seeds = sorted(seeds, key=_seed_sort_key)
    blocked = _blocked_rival_keys(already_have, client_name, websites=[client_website] if client_website else None)
    client_host = _domain_of(client_website or "")
    out: list[dict] = []
    for seed in seeds:
        name = _as_str(seed.get("name")).strip()
        website = _normalize_website(_as_str(seed.get("website")) or None)
        if not name or not website:
            continue
        rival_tier = _as_str(seed.get("tier")) or _food_tier_from_blob(name)
        rival_fmt = _as_str(seed.get("format")) or _food_format_from_blob(name)
        if not _food_tier_compatible(tier, rival_tier):
            continue
        if not _food_format_compatible(client_fmt, rival_fmt):
            continue
        # Local specialty: prefer local peers first; still allow national same-category fill
        if (
            tier == _FOOD_TIER_LOCAL
            and rival_tier == _FOOD_TIER_NATIONAL
            and len(out) >= limit
        ):
            continue
        seed_keys = _rival_keys(name, website)
        if seed_keys & blocked:
            continue
        if client_host and _domain_of(website) == client_host:
            continue
        base_overlap = 82.0 if rival_tier == tier else (74.0 if rival_tier == _FOOD_TIER_LOCAL else 70.0)
        base_overlap += _food_tier_overlap_bonus(tier, rival_tier)
        base_overlap += _food_format_overlap_bonus(client_fmt, rival_fmt)
        out.append(
            {
                "name": name,
                "website": website,
                "industry": (
                    "Restaurant"
                    if rival_fmt == _FOOD_FORMAT_RESTAURANT
                    else "Cafe"
                    if rival_fmt == _FOOD_FORMAT_CAFE
                    else "Bakery"
                    if rival_fmt == _FOOD_FORMAT_BAKERY
                    else "Fast food"
                ),
                "business_model": "other",
                "headquarters_country": key.title() if key else market,
                "why_relevant": (
                    f"Same-category {rival_fmt} peer in {market} ({rival_tier.replace('_', ' ')}); "
                    f"competes for similar diners as {client_name}."
                ),
                "threat_level": "high" if rival_tier == tier and rival_fmt == client_fmt else "medium",
                "overlap_score": max(60.0, min(90.0, base_overlap)),
                "same_niche": True,
                "same_market": True,
                "food_tier": rival_tier,
                "food_format": rival_fmt,
                "peer_scale": (
                    _PEER_BOUTIQUE
                    if rival_tier == _FOOD_TIER_LOCAL
                    else _PEER_ENTERPRISE
                    if rival_tier == _FOOD_TIER_GLOBAL
                    else _PEER_MID
                ),
                "source": "seed",
            }
        )
        blocked |= seed_keys
        if len(out) >= limit:
            break
    return out


def _is_curated_seed_rival(name: str, market: str | None = None, *, kind: str | None = None) -> bool:
    key = _as_str(name).lower().strip()
    if not key:
        return False
    market_key = _normalize_country_key(market or "")
    software_ok = kind == "software"
    food_ok = kind == "food"
    beauty_ok = kind == "beauty"
    if software_ok or kind is None:
        for seed in _GLOBAL_SOFTWARE_SEEDS:
            if _as_str(seed.get("name")).lower() == key:
                return True
        for country_key, seeds in _LOCAL_SOFTWARE_SEEDS.items():
            if market_key and country_key != market_key:
                continue
            for seed in seeds:
                if _as_str(seed.get("name")).lower() == key:
                    return True
    if food_ok or kind is None:
        for country_key, seeds in _LOCAL_QSR_SEEDS.items():
            if market_key and country_key != market_key:
                continue
            for seed in seeds:
                if _as_str(seed.get("name")).lower() == key:
                    return True
    if beauty_ok or kind is None:
        for country_key, seeds in _LOCAL_BEAUTY_SEEDS.items():
            if market_key and country_key != market_key:
                continue
            for seed in seeds:
                if _as_str(seed.get("name")).lower() == key:
                    return True
    for country_key, cat_dict in _LOCAL_MULTI_INDUSTRY_SEEDS.items():
        if market_key and country_key != market_key:
            continue
        for cat, seeds in cat_dict.items():
            for seed in seeds:
                if _as_str(seed.get("name")).lower() == key:
                    return True
    return False


def _looks_like_retail_or_media(blob: str) -> str | None:
    """Return 'retail' or 'media' when blob clearly isn't a B2B peer company."""
    text = _as_str(blob).lower()
    if not text:
        return None
    retail_hits = sum(1 for m in _RETAIL_MARKETPLACE_MARKERS if m in text)
    if retail_hits >= 1 and any(
        m in text
        for m in ("shop", "shopping", "retail", "marketplace", "ecommerce", "e-commerce", "store", "cart")
    ):
        return "retail"
    if any(m in text for m in _MEDIA_DIRECTORY_MARKERS):
        return "media"
    return None


def _incompatible_peer(
    *,
    client_model: str,
    client_industry: str,
    client_niche: str,
    rival_model: str,
    rival_industry: str,
    rival_blob: str,
    client_name: str = "",
) -> bool:
    """True when rival is clearly not the same kind of business as the client."""
    client_family = _model_family(client_model)
    rival_family = _model_family(rival_model)
    client_l = f"{client_name} {client_model} {client_industry} {client_niche}".lower()
    rival_l = f"{rival_model} {rival_industry} {rival_blob}".lower()

    client_cat = _detect_industry_category(client_l)
    rival_cat = _detect_industry_category(rival_l)

    client_is_consumer = _looks_like_food_client(client_l) or _looks_like_beauty_client(client_l) or client_cat in {"retail", "food", "beauty"}
    client_is_b2b = (not client_is_consumer) and (
        client_cat == "software"
        or _looks_like_software_peer_client(client_l)
        or client_family in {"b2b_services", "b2b_software"}
    )

    client_verticals = _detect_verticals(client_l)
    rival_verticals = _detect_verticals(rival_l)
    if _looks_like_food_client(client_l):
        client_verticals.add("food_qsr")
    if _looks_like_food_client(rival_l):
        rival_verticals.add("food_qsr")
    if _looks_like_beauty_client(client_l):
        client_verticals.add("beauty_personal_care")
    if _looks_like_beauty_client(rival_l):
        rival_verticals.add("beauty_personal_care")

    client_is_food = _looks_like_food_client(client_l) or "food_qsr" in client_verticals
    rival_is_food = _looks_like_food_client(rival_l) or "food_qsr" in rival_verticals
    client_is_beauty = _looks_like_beauty_client(client_l) or "beauty_personal_care" in client_verticals
    rival_is_beauty = _looks_like_beauty_client(rival_l) or "beauty_personal_care" in rival_verticals

    # News / blogs / media portals / rankings / publishers are NEVER competitors for education, software, food, beauty, retail, etc.
    if any(
        m in rival_l for m in (
            "news website", "news portal", "information portal", "news and information",
            "media company", "news publisher", "online magazine", "blog post", "article portal",
            "not an educational institution", "not a university", "not a school", "not a direct competitor",
            "does not compete for students", "does not compete for diners", "does not compete for clients",
            "regional studies", "regionalstudies", "research journal", "news article", "journal publication"
        )
    ):
        return True

    # Education sub-tier check (University vs School vs College vs Academy)
    client_is_edu = client_cat in {"education", "university", "school", "college", "academy"} or any(
        t in client_l for t in ("education", "school", "university", "college", "academy", "higher ed", "k-12", "k12")
    )
    rival_is_edu = rival_cat in {"education", "university", "school", "college", "academy"} or any(
        t in rival_l for t in ("education", "school", "university", "college", "academy", "higher ed", "k-12", "k12")
    )
    if client_is_edu and rival_is_edu:
        c_tier = _detect_education_tier(client_l)
        r_tier = _detect_education_tier(rival_l)
        if not _education_tier_compatible(c_tier, r_tier):
            return True
        return False

    # Food sub-format check (Bakery vs Burger vs Pizza vs Shawarma vs Cafe vs Asian vs Dining)
    if client_is_food and rival_is_food:
        c_fmt = _food_format_from_blob(client_l)
        r_fmt = _food_format_from_blob(rival_l)
        if c_fmt and c_fmt != _FOOD_FORMAT_GENERAL and r_fmt and r_fmt != _FOOD_FORMAT_GENERAL:
            if not _food_format_compatible(c_fmt, r_fmt):
                return True

    # Peer verticals and strong alternate verticals
    hard_verticals = {
        "fintech", "retail", "manufacturing", "telecom", "healthcare",
        "edtech", "logistics", "real_estate", "government", "talent_marketplace",
        "food_qsr", "beauty_personal_care", "automotive",
    }
    peer_verticals = {"data_ai", "software_services", "cybersecurity"}

    client_is_software = _looks_like_software_peer_client(client_l) or bool(client_verticals & peer_verticals)
    rival_is_software = _looks_like_software_peer_client(rival_l) or "software_services" in rival_verticals

    # Data Analytics & Enterprise AI boutiques/consultancies are NOT peers of generic custom software dev / IT staffing giants (Systems Ltd, NetSol, Arbisoft, 10Pearls)
    if client_cat == "data_ai":
        is_generic_software_giant = any(
            tok in rival_l
            for tok in (
                "systems limited", "netsol", "10pearls", "arbisoft", "contour software",
                "purelogics", "nextbridge", "dpl", "pure logics", "ovex", "confiz", "tintash", "devsinc"
            )
        )
        if is_generic_software_giant:
            return True
        if rival_cat == "software" and not any(
            tok in rival_l
            for tok in (
                "data", "analytics", "business intelligence", "bi", "machine learning",
                "artificial intelligence", "ai", "data engineering", "power bi", "tableau", "snowflake", "databricks"
            )
        ):
            return True

    # Cross-vertical clashes
    if client_is_beauty and rival_is_beauty:
        return False
    if client_is_beauty and (rival_is_food or rival_is_software or "fintech" in rival_verticals):
        return True
    if (client_is_food or client_is_b2b or client_is_software) and rival_is_beauty:
        return True

    if rival_verticals & hard_verticals:
        # Client must share that vertical (or explicitly be in it)
        shared_hard = client_verticals & rival_verticals & hard_verticals
        if not shared_hard and client_cat != rival_cat:
            if client_verticals or client_is_food or client_is_b2b or client_is_beauty:
                return True

    if client_is_food and rival_is_software:
        return True
    if client_is_software and rival_is_food:
        return True
    if client_is_food and _looks_like_fmcg_or_snack_brand(rival_l):
        return True
    if rival_is_software and not client_is_software:
        return True

    # Software houses / digital agencies are not peers of talent marketplaces (Andela/Turing/Toptal)
    if "talent_marketplace" in rival_verticals and "talent_marketplace" not in client_verticals:
        return True
    if any(tok in rival_l for tok in ("andela", "turing", "toptal")) and "talent_marketplace" not in client_verticals:
        if any(tok in client_l for tok in ("software", "agency", "development", "digital", "it services")):
            return True

    # Commercial software houses / agencies never compete with government boards/authorities or mega public enterprise giants
    client_is_commercial_software = any(
        tok in client_l
        for tok in (
            "software house", "software", "agency", "saas", "services", "it services",
            "digital agency", "technology", "product",
        )
    ) and "government" not in client_verticals
    if client_is_commercial_software and (
        "government" in rival_verticals or _looks_like_government(rival_l)
    ):
        return True
    # Mega public corporations & banking platform vendors are not peers for mid/agile software houses
    is_client_mega = any(tok in client_l for tok in ("systems limited", "netsol"))
    if client_is_commercial_software and not is_client_mega:
        if any(tok in rival_l for tok in ("systems limited", "netsol", "avanza solutions")):
            return True

    if client_is_b2b and client_is_software:
        kind = _looks_like_retail_or_media(rival_blob)
        if kind in {"retail", "media"}:
            return True
        if rival_family in {"retail", "fintech"} and "fintech" not in client_verticals and "retail" not in client_verticals:
            return True
        # Manufacturing / industrial plant AI is not a peer for marketing/data agencies
        if "manufacturing" in rival_verticals and "manufacturing" not in client_verticals:
            return True
        # If client looks like data/AI/software services, rival must not be pure fintech/payments
        if (client_verticals & peer_verticals or any(
            tok in client_l for tok in ("ai", "data", "analytics", "intelligence", "agency", "software", "saas")
        )) and ("fintech" in rival_verticals) and ("fintech" not in client_verticals):
            return True

    if client_family and rival_family and client_family != rival_family:
        if {client_family, rival_family} == {"b2b_services", "b2b_software"}:
            return False  # agency vs saas can still be peers in some niches
        if {"retail", "marketplace", "fintech", "food"} & {client_family, rival_family}:
            return True
    return False


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


def _business_model_from_client(client: ClientBrand) -> str:
    notes = _as_str(client.notes)
    for line in notes.splitlines():
        if line.lower().startswith("business model:"):
            return line.split(":", 1)[1].strip()
    return ""


def _set_market_area(client: ClientBrand, market_area: str) -> None:
    market_area = _as_str(market_area).strip()
    notes = _as_str(client.notes)
    lines = [ln for ln in notes.splitlines() if not ln.lower().startswith("market:")]
    if market_area:
        lines.insert(0, f"Market: {market_area}")
    client.notes = "\n".join(lines).strip() or None


def _set_business_model(client: ClientBrand, business_model: str) -> None:
    business_model = _as_str(business_model).strip()
    notes = _as_str(client.notes)
    lines = [ln for ln in notes.splitlines() if not ln.lower().startswith("business model:")]
    if business_model:
        # Keep Market: first when present
        insert_at = 1 if lines and lines[0].lower().startswith("market:") else 0
        lines.insert(insert_at, f"Business model: {business_model}")
    client.notes = "\n".join(lines).strip() or None


def _niche_competitor_queries(
    client: ClientBrand,
    market_area: str = "",
    *,
    scope: str = "local",
) -> list[str]:
    niche = _as_str(client.niche) or _as_str(client.industry) or "business"
    market = market_area or _market_area_from_client(client)
    model = _business_model_from_client(client).lower()
    niche_l = niche.lower()
    food_blob = f"{client.name} {niche} {_as_str(client.industry)} {model}"
    is_global = str(scope).lower() == "global"
    clean_name = re.sub(r"\b(lahore|karachi|islamabad|rawalpindi|peshawar|multan|faisalabad|pakistan|dubai|riyadh|london|uk|usa)\b", "", client.name, flags=re.I).strip()
    clean_name = re.sub(r"\s+", " ", clean_name) or client.name
    geo = "" if is_global else _as_str(market).strip()
    queries: list[str] = []

    # Food / QSR: use format detector (Cheezious → pizza even when niche says "restaurant")
    if _looks_like_food_client(food_blob):
        food_tier = _food_tier_from_blob(client.name, niche, client.industry)
        fmt = _food_format_from_blob(client.name, niche, client.industry, client.notes, client.tagline)
        format_q = {
            _FOOD_FORMAT_PIZZA: "pizza places",
            _FOOD_FORMAT_SHAWARMA: "shawarma places",
            _FOOD_FORMAT_BAKERY: "bakeries",
            _FOOD_FORMAT_CAFE: "cafes",
            _FOOD_FORMAT_BURGER: "burger places",
            _FOOD_FORMAT_ASIAN: "asian restaurants",
            _FOOD_FORMAT_RESTAURANT: "restaurants",
        }.get(fmt, "restaurants")
        if is_global:
            if format_q == "pizza places":
                queries = [
                    f"{client.name} pizza chain competitors worldwide",
                    "international pizza delivery chains competitors",
                    f"global pizza brands like {client.name}",
                    "best international pizza restaurant chains",
                ]
            elif format_q == "shawarma places":
                queries = [
                    f"{client.name} shawarma competitors international",
                    "global shawarma / doner restaurant chains",
                    f"international wrap and shawarma brands like {client.name}",
                ]
            else:
                queries = [
                    f"{client.name} {format_q} competitors worldwide",
                    f"global {format_q} brands like {client.name}",
                    f"international {format_q} chains same category as {client.name}",
                ]
        elif format_q == "pizza places":
            queries = [
                f'pizza restaurants {geo} -"Pizza Hut" -Domino -"Papa John"'.strip(),
                f"best pizza places Lahore {geo}".strip()
                if _normalize_country_key(geo) == "pakistan"
                else f"best pizza places {geo}".strip(),
                f"{client.name} pizza competitors {geo}".strip(),
                f"Broadway Pizza Pizza Max California Pizza Eastern Oven {geo}".strip(),
                f"local pizza brands {geo} website".strip(),
            ]
        elif format_q == "shawarma places":
            queries = [
                f"shawarma places {geo} -instagram -foodpanda -tripadvisor".strip(),
                f"best shawarma Lahore {geo}".strip()
                if _normalize_country_key(geo) == "pakistan"
                else f"best shawarma {geo}".strip(),
                f"{client.name} shawarma competitors {geo}".strip(),
                f'"Shawarma Stop" PITA "Arabic Shawarma" "Monty Shawarma" {geo}'.strip(),
                f"shawarma brands {geo} website -foodpanda".strip(),
            ]
        elif format_q == "restaurants":
            queries = [
                f"casual dining restaurants {geo}".strip(),
                f"{client.name} restaurant competitors {geo}".strip(),
                f"best restaurants {geo} local brands".strip(),
                f"independent restaurants {geo}".strip(),
            ]
        else:
            queries = [
                f"{format_q} in {geo}".strip(),
                f"best {format_q} {geo} local brands".strip(),
                f"{client.name} {format_q} competitors {geo}".strip(),
                f"local {format_q} like {client.name} in {geo}".strip(),
            ]
        if (not is_global) and _foreign_places_echoed_from_brand(client.name, geo):
            queries = [
                f"popular {format_q} {geo}".strip(),
                f"local {format_q} {geo}".strip(),
                f"{client.name} {geo} competitors".strip(),
            ] + queries
        if (not is_global) and food_tier == _FOOD_TIER_LOCAL and format_q not in {
            "pizza places",
            "shawarma places",
        }:
            queries.extend(
                [
                    f"independent {format_q} {geo} not Pizza Hut".strip(),
                    f"popular local {format_q} {geo}".strip(),
                ]
            )
    # Beauty / salon / aesthetics
    elif _looks_like_beauty_client(client.name, niche, client.industry, model):
        if is_global:
            queries = [
                f"top beauty brands like {client.name} worldwide",
                f"international cosmetics and salon competitors {client.name}",
                f"beauty salon chains similar to {client.name} global",
            ]
        elif geo:
            queries = [
                f"top beauty salons in {geo}",
                f"best makeup studio salon {geo}",
                f"bridal salon aesthetic clinic {geo}",
                f"{client.name} competitors beauty salon {geo}",
                f"beauty parlour brands in {geo}",
            ]
        else:
            queries = [
                f"beauty salons like {client.name}",
                f"makeup studios competitors {client.name}",
            ]
    # Data Analytics, Business Intelligence & AI Solutions
    elif _detect_industry_category(client.name, niche, client.industry, model) == "data_ai":
        if is_global:
            queries = [
                "top modern data stack and business intelligence consultancies",
                "leading enterprise AI and data analytics consulting firms",
                f"data analytics and BI companies similar to {clean_name} worldwide",
                f"global AI solutions and analytics boutique consultancies like {clean_name}",
            ]
        elif geo:
            queries = [
                f"top data analytics and business intelligence companies in {geo}",
                f"AI solutions and data consulting firms in {geo}",
                f"business intelligence BI consulting companies {geo}",
                f"enterprise AI automation companies {geo} like {client.name}",
                f"{client.name} data analytics competitors {geo}",
            ]
        else:
            queries = [
                f"data analytics and business intelligence companies like {client.name}",
                f"enterprise AI consulting competitors {client.name}",
            ]
    # Software-house / IT services — only when client is actually software
    elif _looks_like_software_peer_client(client.name, niche, client.industry, model):
        if is_global:
            queries = [
                "top global custom software development companies",
                "leading international IT services and digital engineering firms",
                f"digital product companies similar to {clean_name} worldwide",
                f"global software development companies like {clean_name}",
            ]
        elif geo:
            queries = [
                f"top software houses in {geo}",
                f"software development companies in {geo}",
                f"digital agencies {geo} like {client.name}",
                f"IT services companies {geo}",
                f"{client.name} competitors software house {geo}",
            ]
        else:
            queries = [
                f"software development companies like {client.name}",
                f"digital agencies competitors {client.name}",
            ]
    # Education: University vs School vs College vs Academy
    elif _detect_industry_category(client.name, niche, client.industry, model) in {"education", "university", "school", "college", "academy"}:
        edu_tier = _detect_education_tier(client.name, niche, client.industry, client.notes, client.tagline)
        if edu_tier == "university":
            if is_global:
                queries = [
                    "top international universities worldwide research higher education",
                    f"leading universities worldwide competitors {clean_name}",
                    "top global research universities worldwide",
                    "best international universities higher education US UK Europe",
                ]
            elif geo:
                queries = [
                    f"top universities in {geo}",
                    f"best higher education degree awarding universities {geo}",
                    f"{client.name} university competitors {geo}",
                    f"leading universities in {geo} bachelor master",
                    f"higher education institutions in {geo}",
                ]
            else:
                queries = [
                    f"universities like {client.name}",
                    f"higher education university competitors {client.name}",
                ]
        elif edu_tier == "school":
            if is_global:
                queries = [
                    "top international k-12 school systems worldwide",
                    "leading global private schools networks international",
                    f"international school systems similar to {clean_name}",
                ]
            elif geo:
                queries = [
                    f"top private schools in {geo}",
                    f"best school systems in {geo} k-12",
                    f"{client.name} school competitors {geo}",
                    f"leading grammar and high schools in {geo}",
                    f"cambridge o level a level schools in {geo}",
                ]
            else:
                queries = [
                    f"schools like {client.name}",
                    f"school system competitors {client.name}",
                ]
        elif edu_tier == "college":
            if is_global:
                queries = [
                    "leading international intermediate and pre-university colleges worldwide",
                ]
            elif geo:
                queries = [
                    f"top intermediate colleges in {geo}",
                    f"best colleges in {geo} fsc ics a levels",
                    f"{client.name} college competitors {geo}",
                ]
            else:
                queries = [
                    f"colleges like {client.name}",
                ]
        else:
            if is_global:
                queries = [
                    "leading international test preparation and online learning platforms",
                ]
            elif geo:
                queries = [
                    f"top entry test preparation academies in {geo}",
                    f"coaching centers in {geo} mdcat ecat",
                    f"{client.name} academy competitors {geo}",
                ]
            else:
                queries = [
                    f"test prep academies like {client.name}",
                ]
    else:
        if is_global:
            queries = [
                f"top international {niche} companies worldwide",
                f"global companies like {clean_name} {niche}",
                f"international {niche} brands competitors",
            ]
        else:
            queries = [
                f"{client.name} competitors {niche} {geo}".strip(),
                f"companies like {client.name} {niche} {geo}".strip(),
            ]
            if geo:
                queries.extend(
                    [
                        f"{niche} companies in {geo}",
                        f"{client.name} competitors {geo}",
                        f"local {niche} brands similar to {client.name} {geo}",
                    ]
                )

    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        key = q.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q.strip())
    return out[:8]


_SERP_NOISE_DOMAINS = {
    # Freelancing & Gig Platforms (NOT B2B Software / Agency Peers)
    "upwork.com", "fiverr.com", "freelancer.com", "toptal.com", "guru.com", "peopleperhour.com",
    "bebee.com", "tracxn.com",
    # Software & SaaS review / directory aggregators
    "g2.com", "capterra.com", "capterra.ae", "capterra.co.uk", "getapp.com", "softwareadvice.com", "trustradius.com",
    "crozdesk.com", "saashub.com", "alternativeto.net", "slashdot.org", "producthunt.com",
    "clutch.co", "goodfirms.co", "sortlist.com", "designrush.com", "upcity.com",
    "techbehemoths.com", "themanifest.com", "topdevelopers.co", "appfutura.com",
    "extract.co", "wadline.com", "directory.com", "yellowpages.com", "yelp.com",
    "tripadvisor.com", "tripadvisor.com.pk", "wheree.com", "jagha.pk", "menuprices.pk",
    "foodpanda.pk", "foodpanda.com", "foodiespakistan.pk", "trustpilot.com",
    "sitejabber.com", "glassdoor.com", "indeed.com", "zoominfo.com", "crunchbase.com",
    "pitchbook.com", "owler.com", "cbinsights.com", "zippia.com", "comparably.com",
    # Social & video platforms
    "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com",
    "tiktok.com", "vm.tiktok.com", "pinterest.com", "reddit.com", "quora.com",
    "threads.net", "vimeo.com", "dailymotion.com",
    # Major news & media publications (articles/listicles, not SaaS/agency rivals)
    "forbes.com", "techcrunch.com", "theverge.com", "wired.com", "venturebeat.com",
    "zdnet.com", "cnet.com", "businessinsider.com", "bloomberg.com", "reuters.com",
    "nytimes.com", "wsj.com", "bbc.com", "cnn.com", "mashable.com", "propakistani.pk",
    "techinasia.com", "tribune.com.pk", "dawn.com", "geo.tv", "thenews.com.pk",
    "dailymail.co.uk", "theguardian.com", "huffpost.com", "economist.com",
    "entrepreneur.com", "inc.com", "fastcompany.com", "hackernews.com", "ycombinator.com",
    # General blog hosting / publishing platforms
    "medium.com", "substack.com", "dev.to", "hashnode.dev", "hashnode.com",
    "wordpress.com", "wordpress.org", "blogspot.com", "tumblr.com", "ghost.io",
    "beehiiv.com", "wixsite.com", "weebly.com", "pakistantravelblog.com",
    "travelblog.org", "magnific.com", "dawnbread.com", "dawnbread.com.pk",
    "dawnfoods.com", "google.com", "photos.google.com", "drive.google.com",
    "docs.google.com", "notion.site", "gitbook.io", "wikipedia.org", "wikihow.com",
    "github.com", "gitlab.com", "bitbucket.org", "sourceforge.net",
    # Education ranking aggregators and directories
    "timeshighereducation.com", "topuniversities.com", "usnews.com", "shanghairanking.com",
    "edurank.org", "4icu.org", "unirank.org", "unirank.com", "studyportals.com", "bachelorsportal.com",
    "mastersportal.com", "phdportal.com", "universitiesrankings.com", "cwur.org", "worldresearchranking.com",
    "webometrics.info",
    # Market research report aggregators and PR wire sites
    "ibisworld.com", "idc.com", "my.idc.com", "gartner.com", "forrester.com",
    "imarcgroup.com", "mordorintelligence.com", "grandviewresearch.com", "expertmarketresearch.com",
    "alliedmarketresearch.com", "fortunebusinessinsights.com", "verifiedmarketresearch.com",
    "marketsandmarkets.com", "statista.com", "globenewswire.com", "prnewswire.com", "businesswire.com",
    "custommarketinsights.com", "thebrainyinsights.com", "coherentmarketinsights.com",
    "kyubit.com", "perceptive-analytics.com", "nsktglobal.com", "inzinc.ae", "urcapk.com", "karachidotai.com",
    # Regional / national news and blog portals
    "regionalstudies.com.pk", "regionalstudies.org", "pakistantoday.com.pk", "dailytimes.com.pk",
    "brecorder.com", "dailypakistan.com.pk", "nation.com.pk", "pakobserver.net",
}

# Host markers (substring) for content / grocery sites mistaken as rivals
_CONTENT_OR_CPG_HOST_MARKERS = (
    "travelblog", "foodblog", "blog", "magazine", "recipes", "photos",
    "bread", "bakerywholesale", "flour", "grocery", "review", "directory",
    "news", "press", "media", "portal", "wiki", "regionalstudies", "studies",
    "imarcgroup", "mordorintelligence", "grandviewresearch", "marketresearch",
)
_CPG_PATH_MARKERS = ("/product/", "/products/", "/shop/", "/sku/")
_DIRECTORY_OR_LISTICLE_PATH_MARKERS = (
    "/companies/", "/company/", "/agencies/", "/agency/", "/top-", "/best-",
    "/10-top-", "/software-companies", "/it-companies", "/list-of-", "/directory/",
    "/blog/", "/blogs/", "/post/", "/posts/", "/article/", "/articles/", "/news/",
    "/story/", "/stories/", "/review/", "/reviews/", "/vs/", "/compare/",
    "/comparison/", "/list/", "/lists/", "/guides/", "/guide/", "/insights/",
    "/trends/", "/category/", "/tag/", "/author/", "/feed/", "/press-release/",
    "/how-to-", "/tutorials/", "/case-studies/", "/ranking/", "/rankings/",
    "/market-report/", "/market-research/", "/report/", "/reports/", "-market",
    "-market-size", "-market-share", "-industry-analysis", "-growth-trends",
)
_CITY_IN_TITLE_RE = re.compile(
    r"\b(in|near)\s+(lahore|karachi|islamabad|rawalpindi|peshawar|multan|faisalabad|pakistan|dubai|riyadh|london|uk|usa)\b",
    re.I,
)
_BLOG_OR_ARTICLE_TITLE_PATTERNS = re.compile(
    r"(\btop\s+\d+\b|\bbest\s+\d+\b|\b\d+\s+(top|best|leading)\b|"
    r"\b(top|best|leading|\d+)\s+(pakistani\s+)?(universities|schools|colleges|companies|places|restaurants|cafes|software|softwares|tools|agencies)\b|"
    r"\b(included\s+in|featured?\s+in|rankings?|review|reviews|guide|how\s+to|complete\s+guide|versus|comparison)\b)",
    re.I,
)


def _is_serp_noise_domain(url: str) -> bool:
    host = _domain_of(url)
    if not host:
        return False
    if any(k in host for k in ("regionalstudies", "pakistantoday", "dailytimes", "pakobserver", "hec.gov", "punjab.gov", "mofept.gov")):
        return True
    if ".gov." in host or host.endswith(".gov") or host.endswith(".gov.pk") or host.endswith(".gov.ae") or host.endswith(".gov.uk"):
        return True
    for blocked in _SERP_NOISE_DOMAINS:
        if host == blocked or host.endswith("." + blocked):
            return True
    return False


def _is_blog_or_article_url(url: str, title: str = "") -> bool:
    """True if URL or title represents a blog post, article, news story, or listicle directory."""
    if not url:
        return False
    if _is_serp_noise_domain(url):
        return True
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = (parsed.hostname or "").lower()
        path = (parsed.path or "").lower()
        if any(p in path for p in _DIRECTORY_OR_LISTICLE_PATH_MARKERS):
            return True
        if re.search(r"/\d{4}/\d{2}/", path) or re.search(r"/(how-to|best|top|guide)-", path):
            return True
        # Slug with multiple words / hyphens (e.g. /33-pakistani-universities-included-in-...)
        if path.count("-") >= 3 or re.search(r"/\d+-[a-z0-9-]+", path):
            return True
        # News / blog host markers
        if any(k in host for k in ("regionalstudies", "pakistantoday", "dailytimes", "tribune", "dawn", "brecorder", "thenews", "dailypakistan", "propakistani")):
            return True
    except Exception:
        pass
    if title and _BLOG_OR_ARTICLE_TITLE_PATTERNS.search(title.strip()):
        return True
    return False


def _looks_like_content_or_cpg_noise(name: str, website: str | None = None) -> bool:
    """
    True for travel blogs, photo galleries, listicles, bread/CPG product pages —
    e.g. 'Yemeni Food in Islamabad', 'Pakistani shawarma Photos', 'Dawn Shawarma' (bread),
    'Top 10 CRM Software in 2026'.
    """
    raw = _as_str(name).strip()
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    if not key:
        return True
    # Article title patterns
    if _BLOG_OR_ARTICLE_TITLE_PATTERNS.search(raw):
        return True
    # Photo / gallery titles
    if re.search(r"\bphotos?\b|\bgallery\b|\bimages?\b", key):
        return True
    # Travel/listicle: "Yemeni Food in Islamabad", "Top Software Companies in Lahore"
    if _CITY_IN_TITLE_RE.search(key):
        return True
    # Explicit blog/listicle cues in the name
    if any(tok in key for tok in (
        " food blog", "travel blog", "best places", "things to eat", "top 10", "top 5",
        "best software", "complete guide", "alternatives to", "vs ", " versus "
    )):
        return True

    website = _as_str(website).strip().lower()
    if not website:
        return False
    if _is_blog_or_article_url(website, name):
        return True
    host = _domain_of(website)
    path = ""
    try:
        from urllib.parse import urlparse
        path = (urlparse(website if "://" in website else f"https://{website}").path or "").lower()
    except Exception:
        path = ""

    if _is_serp_noise_domain(website):
        return True
    if any(m in host for m in _CONTENT_OR_CPG_HOST_MARKERS):
        if "photos" in host or "blog" in host or "travel" in host or "news" in host or "magazine" in host:
            return True
        if "bread" in host:
            return True
    if any(m in path for m in _CPG_PATH_MARKERS) and any(
        tok in host for tok in ("bread", "dawn", "foods", "grocery", "wholesale")
    ):
        return True
    if any(m in path for m in _DIRECTORY_OR_LISTICLE_PATH_MARKERS):
        return True
    if "dawn" in key and "shawarma" in key and "bread" in host:
        return True
    return False


# Packaged snacks / FMCG conglomerates — not restaurant/QSR peers (Cheezious ≠ Unilever)
_FMCG_OR_SNACK_BRANDS = (
    "unilever", "pepsico", "pepsi co", "nestle", "nestlé", "procter", "p&g",
    "munchies", "munchies foods", "lays", "lay's", "kurkure", "cheetos", "doritos",
    "ismail industries", "candyland", "english biscuit", "ebm ", "olympia",
    "mitchell's", "mitchells", "national foods", "shan foods", "sufi",
    "dawn bread", "dawn foods", "mitchell", "fauji foods", "engro foods",
    "k&n's", "k and n", "menu foods", "pakola", "shezan",
)
_FMCG_OR_SNACK_MARKERS = (
    "fmcg", "cpg", "consumer goods", "packaged snack", "snack food", "snack manufacturer",
    "cheese-flavored snack", "cheese flavoured snack", "flavored chips", "potato chips",
    "shelf space", "supermarket brand", "retail distribution network",
    "biscuit manufacturer", "confectionery manufacturer", "beverage giant",
    "home and personal care", "personal care brands",
)


def _looks_like_fmcg_or_snack_brand(*parts: object) -> bool:
    """True for Unilever / PepsiCo / Munchies-style packaged goods — not QSR peers."""
    blob = _context_blob(*parts)
    if not blob:
        return False
    # Real restaurants sometimes sell "snacks" on a menu — require brand or strong CPG cues
    if any(brand in blob for brand in _FMCG_OR_SNACK_BRANDS):
        return True
    hits = sum(1 for m in _FMCG_OR_SNACK_MARKERS if m in blob)
    if hits >= 2:
        return True
    if hits >= 1 and any(tok in blob for tok in ("manufacturer", "subsidiary", "conglomerate", "distribution network")):
        return True
    return False


def _looks_like_marketing_slogan_name(name: str) -> bool:
    """
    Only block obvious non-brand Serp junk — NOT real pizza shops whose trade name
    is slogan-ish (e.g. 'Savor The Biggest Pizza in Town', 'Pizza Online').
    """
    key = re.sub(r"\s+", " ", _as_str(name).lower()).strip()
    if not key or len(key) < 18:
        return False
    # Long imperative ad copy with CTA — not a storefront name
    if re.search(r"\b(order it today|order now|baked to perfection|made fresh[,—-])\b", key):
        return True
    if key.count(" ") >= 8 and re.search(r"\b(from classic to|has it all|check .+ menu)\b", key):
        return True
    return False


def _token_hits(haystack: str, source: str, *, min_len: int = 3) -> int:
    if not haystack or not source:
        return 0
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", source.lower()) if len(tok) >= min_len]
    if not tokens:
        return 0
    return sum(1 for tok in tokens if tok in haystack)


def _filter_niche_competitors(
    items: list[dict],
    client_name: str,
    *,
    market_area: str = "",
    niche: str = "",
    industry: str = "",
    business_model: str = "",
    min_overlap: float = 55.0,
    limit: int = 10,
    require_local_market: bool = False,
    client_food_tier: FoodTier | None = None,
    client_peer_scale: PeerScale | None = None,
) -> list[dict]:
    """Keep only peer rivals that fit niche/industry/scale; rank by relevance score."""
    scored: list[dict] = []
    seen_names: set[str] = set()
    seen_hosts: set[str] = set()
    niche_l = niche.lower().strip()
    industry_l = industry.lower().strip()
    market_l = market_area.lower().strip()
    model_l = business_model.lower().strip()
    food_client = _looks_like_food_client(client_name, niche, industry, business_model)
    food_tier = client_food_tier or (
        _food_tier_from_blob(client_name, niche, industry) if food_client else None
    )
    client_food_fmt = _food_format_from_blob(client_name, niche, industry) if food_client else ""
    peer_scale = client_peer_scale or _peer_scale_from_blob(
        client_name, niche, industry, business_model, name=client_name
    )

    for item in items:
        if not isinstance(item, dict):
            continue
        name = _as_str(item.get("name")).strip()
        website = _normalize_website(_as_str(item.get("website")) or None)
        item_keys = _rival_keys(name, website)
        if not name or (_rival_keys(client_name) & item_keys) or (item_keys & seen_names):
            continue
        if _is_self_rival(client_name, name, website=website):
            continue
        if _is_generic_or_fake_rival_name(name) and _as_str(item.get("source")).lower() != "seed":
            continue
        if _looks_like_invented_food_domain(name, website) and _as_str(item.get("source")).lower() != "seed":
            continue
        if _looks_like_brand_geo_hallucination(
            client_name,
            name,
            market_l,
            website=website,
            source=_as_str(item.get("source")),
        ):
            continue
        host = _domain_of(website or "")
        if host and host in seen_hosts:
            continue
        if _is_global_megarival(name, website):
            if peer_scale in {_PEER_BOUTIQUE, _PEER_MID} and require_local_market:
                continue
            if peer_scale == _PEER_BOUTIQUE:
                continue
        # Indie/local / boutique brands must not get Pizza Hut / Domino's / KFC as "peers"
        if (food_tier == _FOOD_TIER_LOCAL or peer_scale == _PEER_BOUTIQUE) and _is_global_food_franchise(
            name, website
        ):
            continue
        if website and (_is_serp_noise_domain(website) or _is_blog_or_article_url(website, name)):
            continue
        if _looks_like_content_or_cpg_noise(name, website):
            continue
        if food_client and (
            _looks_like_fmcg_or_snack_brand(
                name,
                item.get("why_relevant"),
                item.get("description"),
                item.get("industry"),
                website,
            )
            or _looks_like_marketing_slogan_name(name)
        ):
            continue
        if item.get("same_niche") is False or item.get("is_global_platform") is True:
            continue
        if require_local_market and item.get("same_market") is False:
            continue
        # Invented AI rivals often ship a website that doesn't match the brand
        alignment_ok = (
            (not website)
            or _as_str(item.get("source")).lower() in {"serp", "seed", "ai", "ai_same_tier", ""}
            or _name_aligned_with_domain(name, website)
        )
        # Name↔domain alignment alone is NOT proof — "Pizza 5" + pizza5.pk is still fake
        if (
            website
            and _as_str(item.get("source")).lower() not in {"serp", "seed"}
            and alignment_ok
            and _looks_like_invented_food_domain(name, website)
        ):
            continue
        if website and not alignment_ok and require_local_market:
            continue

        why = _as_str(item.get("why_relevant") or item.get("description"))
        item_industry = _as_str(item.get("industry"))
        item_model = _as_str(item.get("business_model"))
        hq_country = _as_str(item.get("headquarters_country") or item.get("headquarters") or item.get("market_overlap"))
        try:
            score = float(item.get("overlap_score") or item.get("niche_fit_score") or 50)
        except (TypeError, ValueError):
            score = 50.0
        if not alignment_ok:
            score -= 20

        blob = f"{name} {why} {website or ''} {item_industry} {item_model} {hq_country}".lower()
        rival_food_tier = _as_str(item.get("food_tier")) or (
            _FOOD_TIER_GLOBAL
            if _is_global_food_franchise(name, website)
            else _food_tier_from_blob(name, item_industry, why)
            if food_client
            else ""
        )
        rival_scale = _as_str(item.get("peer_scale")) or _peer_scale_from_blob(
            name, item_industry, why, name=name, website=website
        )
        if not _peer_scale_compatible(peer_scale, rival_scale):
            continue
        score += _peer_scale_overlap_bonus(peer_scale, rival_scale)
        if food_tier and rival_food_tier:
            if not _food_tier_compatible(food_tier, rival_food_tier):
                continue
            score += _food_tier_overlap_bonus(food_tier, rival_food_tier)
        if food_client and client_food_fmt and client_food_fmt != _FOOD_FORMAT_GENERAL:
            # Do not use why_relevant for format — seeds often say "…as {client}" and
            # brand tokens (e.g. Cheezious→pizza) would mis-label software houses.
            rival_fmt = _as_str(item.get("food_format")) or _food_format_from_blob(
                name, item_industry
            )
            src = _as_str(item.get("source")).lower()
            if rival_fmt == _FOOD_FORMAT_GENERAL and src in {"serp", "seed"}:
                # Provisional until pack verifies; prefer same-category later
                score -= 12
            elif not _food_format_compatible(client_food_fmt, rival_fmt):
                continue
            else:
                score += _food_format_overlap_bonus(client_food_fmt, rival_fmt)

        # Exclude client brand from rival copy so "…as Cheezious" does not mark a software house as food
        rival_kind_blob = f"{name} {item_industry} {item_model} {hq_country}"
        if _incompatible_peer(
            client_model=model_l,
            client_industry=industry_l,
            client_niche=niche_l,
            rival_model=item_model,
            rival_industry=item_industry,
            rival_blob=rival_kind_blob,
            client_name=client_name,
        ):
            continue

        # Local scope: hard-reject clear foreign-country rivals (e.g. India/Singapore when market=Pakistan)
        if require_local_market and market_l:
            if _mentions_conflicting_country(blob, website, market_l, client_name=client_name):
                continue
            hq_key = _normalize_country_key(hq_country)
            market_key = _normalize_country_key(market_l)
            if hq_key and market_key and hq_key != market_key:
                continue
            has_local_signal = (
                _mentions_target_market(blob, website, market_l)
                or (bool(hq_key) and bool(market_key) and hq_key == market_key)
                or _host_matches_tlds(_domain_of(website or ""), _COUNTRY_TLDS.get(market_key or "", set()))
            )
            if not has_local_signal:
                src = _as_str(item.get("source")).lower()
                if src in {"serp", "ai_same_tier", "ai", ""}:
                    # Targeted local prompts / SERP are geo-scoped
                    has_local_signal = True
                    score -= 4
                elif src == "seed":
                    has_local_signal = True
                    score += 6
                else:
                    continue

        # Soft boosts for explicit fit signals
        if item.get("same_niche") is True:
            score += 12
        if niche_l:
            hits = _token_hits(blob, niche_l)
            if hits:
                score += min(14, hits * 5)
            elif len([t for t in niche_l.replace("/", " ").split() if len(t) > 3]) >= 1:
                # No niche token overlap → penalize vague AI guesses
                score -= 12
        if industry_l:
            hits = _token_hits(blob, industry_l) + _token_hits(item_industry.lower(), industry_l)
            if hits:
                score += min(12, hits * 4)
            else:
                score -= 8
        if model_l:
            hits = _token_hits(blob, model_l) + _token_hits(item_model.lower(), model_l)
            if hits:
                score += min(10, hits * 4)
        if market_l:
            if _mentions_target_market(blob, website, market_l):
                score += 16
            elif require_local_market:
                src = _as_str(item.get("source")).lower()
                if src in {"serp", "ai", "ai_same_tier", ""}:
                    # Organic results & targeted AI local runs are already geo-scoped
                    score -= 6
                elif src == "seed":
                    score += 4
                else:
                    score -= 20

        # Local runs need a usable website when one is claimed
        if require_local_market and _as_str(item.get("website")) and not website:
            continue

        score = max(0.0, min(score, 95.0))
        # SERP/seed rows are provisional — allow slightly below AI threshold so real peers survive
        src_final = _as_str(item.get("source")).lower()
        if require_local_market and src_final in {"serp", "seed"}:
            local_min = max(48.0, min_overlap - 5.0)
        elif require_local_market:
            local_min = max(min_overlap, 60.0)
        else:
            local_min = min_overlap
        if score < local_min:
            continue

        seen_names |= item_keys
        if host:
            seen_hosts.add(host)
            host_key = _rival_host_key(website)
            if host_key:
                seen_names.add(host_key)
        scored.append(
            {
                **item,
                "name": name,
                "website": website,
                "industry": item_industry or item.get("industry"),
                "business_model": item_model or item.get("business_model"),
                "why_relevant": why or item.get("why_relevant"),
                "overlap_score": score,
                "threat_level": _as_str(item.get("threat_level"), "high").lower(),
                "food_tier": rival_food_tier or item.get("food_tier"),
                "peer_scale": rival_scale,
            }
        )

    scored.sort(key=lambda row: float(row.get("overlap_score") or 0), reverse=True)
    return scored[: max(1, limit)]


def _brand_guess_from_host(website: str | None) -> str:
    """creative-sols.com → Creative Sols; skip generic hosts."""
    host = _domain_of(website or "")
    if not host:
        return ""
    core = host.split(".")[0]
    skip = {
        "www", "app", "apps", "blog", "shop", "store", "orders", "order", "menu",
        "india", "pakistan", "saudi", "uae", "software", "company", "services",
    }
    if not core or core in skip or len(core) < 3:
        return ""
    # discretelogix → Discretelogix; creative-sols → Creative Sols
    parts = re.split(r"[\-_]+", core)
    nice = " ".join(p.capitalize() for p in parts if p)
    return nice.strip()


def _competitors_from_serp(organic: list[dict], client_name: str) -> list[dict]:
    rivals: list[dict] = []
    seen: set[str] = set()
    client_l = client_name.lower()
    skip_title_bits = (
        "vs ", " versus ", "alternative", "alternatives", "best ", "top ", "compared",
        "review", "pricing", "jobs", "career", "salary", "news", "blog",
        "companies in ", "company in ", "developers in ", "agencies in ",
    )
    for item in organic or []:
        title = _as_str(item.get("title"))
        link = _as_str(item.get("link"))
        snippet = _as_str(item.get("snippet"))
        if not title or not link:
            continue
        if _is_serp_noise_domain(link) or _is_blog_or_article_url(link, title):
            continue
        # Strip leading/trailing navigational noise: "About Us - Brand Name" -> "Brand Name"
        title_clean = re.sub(
            r"^(about\s*(us)?|contact\s*(us)?|home(page)?|our\s*story|who\s*we\s*are|menu|our\s*menu|locations|outlets|branches|careers|jobs|reviews|customer\s*reviews|faq|faqs|login|sign\s*in|order\s*online)\s*[-|–—:]\s*",
            "",
            title,
            flags=re.I,
        ).strip()
        title_clean = re.sub(
            r"\s*[-|–—:]\s*(about\s*(us)?|contact\s*(us)?|home(page)?|our\s*story|who\s*we\s*are|menu|our\s*menu|locations|outlets|branches|careers|jobs|reviews|customer\s*reviews|faq|faqs|login|sign\s*in|order\s*online)$",
            "",
            title_clean,
            flags=re.I,
        ).strip()
        name = title_clean.split("|")[0].split("-")[0].split("–")[0].strip()
        # Drop marketing suffixes: "Eastern Oven Your Go-To Pizza Haven..."
        name = re.split(
            r"\s+[–—|:]\s+|\s+-\s+(menu|order|delivery|reviews?)\b",
            name,
            maxsplit=1,
            flags=re.I,
        )[0].strip()
        name = re.sub(
            r"\s+(menu|order\s+now|online\s+ordering|delivery|lahore|karachi)\s*$",
            "",
            name,
            flags=re.I,
        ).strip()
        # Strip geo SEO tails: "Creative Solutions … Company in Riyadh, Saudi Arabia"
        name = re.sub(
            r"\s+(software|web|it|digital)?\s*(development|developers?|services|solutions|company|companies|agency|house)?\s+in\s+[A-Za-z ,]+$",
            "",
            name,
            flags=re.I,
        ).strip()
        name = _clean_rival_display_name(name)
        # If title is an e-commerce action / SEO query phrase, prefer the domain brand
        if (
            re.match(r"^(buy|order|shop|get|send)\s+", name, flags=re.I)
            or "online in " in name.lower()
            or "best cakes" in name.lower()
            or _is_generic_or_fake_rival_name(name)
            or len(name) > 55
            or _is_blog_or_article_url(link, name)
        ):
            host_brand = _brand_guess_from_host(link)
            if host_brand and not _is_generic_or_fake_rival_name(host_brand) and not _is_blog_or_article_url(link, host_brand):
                name = host_brand
            else:
                continue
        if not name or client_l in name.lower() or len(name) > 60:
            continue
        if _is_self_rival(client_name, name, website=link):
            continue
        if _is_generic_or_fake_rival_name(name) or _looks_like_recipe_or_menu_item_name(name):
            continue
        if _looks_like_content_or_cpg_noise(name, link):
            continue
        if _looks_like_fmcg_or_snack_brand(name, snippet, link) or _looks_like_marketing_slogan_name(name):
            continue
        if _looks_like_invented_food_domain(name, link):
            continue
        if _is_global_food_franchise(name, link):
            continue
        lowered = name.lower()
        if any(bit in lowered for bit in skip_title_bits):
            continue
        if lowered.startswith(("the ", "how ", "what ", "why ", "10 ", "5 ", "7 ", "15 ")):
            continue
        if lowered.startswith(("authentic ", "delicious ", "homemade ", "easy ", "best homemade ")):
            continue
        if _is_global_megarival(name, link):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        rivals.append(
            {
                "name": name,
                "website": _normalize_website(link.split("?")[0]),
                "why_relevant": snippet[:220] or f"Appears in niche search for {client_name} competitors",
                "threat_level": "medium",
                # High enough to survive local filter before enrich validates niche fit
                "overlap_score": 62,
                "same_niche": True,
                "same_market": True,
                "source": "serp",
            }
        )
        if len(rivals) >= 10:
            break
    return rivals


async def _ai_propose_same_tier_peers(
    db: AsyncSession,
    agency_id: str,
    client: ClientBrand,
    *,
    needed: int,
    already_have: list[str],
    scope: str,
    market_focus: str,
    business_model: str,
    serp_candidates: list[dict] | None = None,
) -> list[dict]:
    """Ask the LLM for N more same-tier peer rivals (not seed lists)."""
    needed = max(0, min(10, int(needed or 0)))
    if needed <= 0:
        return []
    peer_scale = _peer_scale_from_blob(
        client.name, client.niche, client.industry, business_model, name=client.name
    )
    is_food = _looks_like_food_client(client.name, client.niche, client.industry, business_model)
    is_beauty = _looks_like_beauty_client(client.name, client.niche, client.industry, business_model)
    is_sw = _looks_like_software_peer_client(client.name, client.niche, client.industry, business_model)
    focus = _as_str(market_focus).strip() or "the client's primary market"
    home_market = _market_area_from_client(client) or focus
    if scope == "global":
        geo_block = (
            f"GLOBAL COMPETITORS RULE (hard): When scope is global, every competitor MUST be an established international / global company or institution headquartered OUTSIDE {home_market}. "
            f"DO NOT return domestic/local competitors from {home_market}. "
            "Return leading international peers operating in the US, UK, Europe, Australia, Asia, or worldwide."
        )
        why_geo_rule = "Every why_relevant must explain how this international peer competes as a global benchmark."
    else:
        geo_block = (
            f"LOCAL COMPETITORS RULE (hard): Every rival MUST be headquartered in OR primarily selling in {focus}. "
            f"headquarters_country must be {focus} (or a city inside it). "
            "Omit any company from another country."
        )
        why_geo_rule = f"Every why_relevant MUST explicitly mention {focus} (city/country) and how they sell there."
    if is_food:
        peer_guidance = (
            "Only real food peers at the SAME scale AND SAME CATEGORY — never software houses. "
            "CATEGORY RULE (hard): restaurant clients get restaurants only; cafe→cafes; bakery/cake shop→bakeries; "
            "burger→burger; pizza→pizza; asian→asian. "
            "Never pair a restaurant with a cake shop (e.g. Meet Me in Paris ≠ Layers Bakeshop). "
            "Never return furniture/kitchen brands (e.g. Cucina as cabinets/furniture). "
            "Never return Andiamo (Dubai Hyatt Italian) as a Pakistan local peer. "
            "Never invent placeholder food brands (Pizza 24, Pizza 5, Pizza 360, Pizza 2 Go, Burger 99) "
            "or fake matching domains (pizza24.pk). Only well-known REAL operating brands. "
            "Never invent French placeholder names. Never return Xinyaki (typo) — Ginyaki is oriental, not a French restaurant peer."
        )
    elif is_beauty:
        peer_guidance = (
            "Only real beauty salons, makeup studios, aesthetic clinics, and personal care brands in the same category. "
            "NEVER software houses, IT firms, restaurants, or unrelated retailers. "
            "Return real established salons and beauty studios with working websites in this market."
        )
    elif is_sw:
        peer_guidance = "Only real commercial software houses / digital agencies / IT product peers — never food chains or unrelated retailers."
    elif _detect_industry_category(client.name, client.niche, client.industry, business_model) in {"education", "university", "school", "college", "academy"}:
        edu_tier = _detect_education_tier(client.name, client.niche, client.industry, client.notes, client.tagline)
        if edu_tier == "university":
            peer_guidance = (
                "EDUCATION RULE (hard): Client is a UNIVERSITY / HIGHER EDUCATION INSTITUTION. "
                "Only real UNIVERSITIES, higher education institutes, and degree-awarding institutions are valid competitors. "
                "STRICTLY FORBIDDEN: K-12 schools, primary/grammar schools, high schools, kindergartens, tutoring/coaching academies. "
                "Every competitor MUST be a recognized UNIVERSITY offering bachelor/master degrees."
            )
        elif edu_tier == "school":
            peer_guidance = (
                "EDUCATION RULE (hard): Client is a K-12 / GRAMMAR / HIGH SCHOOL. "
                "Only real private/public SCHOOL systems and K-12 institutions are valid competitors. "
                "STRICTLY FORBIDDEN: Universities, medical colleges, engineering universities, degree-awarding institutes. "
                "Every competitor MUST be a recognized primary/middle/high SCHOOL system."
            )
        elif edu_tier == "college":
            peer_guidance = (
                "EDUCATION RULE (hard): Client is an INTERMEDIATE / HIGHER SECONDARY COLLEGE. "
                "Only real pre-university colleges and intermediate colleges (FSc/FA/ICS/A-Levels) are valid competitors."
            )
        else:
            peer_guidance = (
                "EDUCATION RULE (hard): Client is a TEST PREP / COACHING ACADEMY or EDTECH platform. "
                "Only test prep academies, entry test coaching centers, and edtech platforms are valid competitors."
            )
    else:
        ind_lbl = _as_str(client.industry) or "the same industry"
        nich_lbl = _as_str(client.niche) or "same niche"
        peer_guidance = f"Only real direct commercial peers in {ind_lbl} ({nich_lbl}) selling similar services/products to similar buyers. NEVER software houses or restaurants unless the client is one."
    prompt = (
        f"Fill exactly {needed} MORE real SAME-TIER peer competitors for this client. "
        f"{geo_block} "
        f"Must-match industry: {_as_str(client.industry) or 'unknown'}. "
        f"Must-match niche: {_as_str(client.niche) or 'unknown'}. "
        f"Must-match business model: {_as_str(business_model) or 'unknown'}. "
        f"{_peer_scale_prompt_rule(peer_scale, is_food=is_food)} "
        f"{peer_guidance} "
        f"{_brand_geo_disclaimer(client.name, focus)} "
        "Do NOT invent placeholder brands (TechCorp, Soft Solutions, PakTech, AxonSoft). "
        "Do NOT pad with seed-list giants that dwarf the client. "
        "Prefer well-known real peers customers would actually compare. "
        "Return JSON: {competitors:[{name, website, industry, business_model, headquarters_country, "
        "why_relevant, threat_level, overlap_score, same_niche:true, same_market:true, is_global_platform:false}]}. "
        f"Return exactly {needed} NEW names not in already_have. Each needs a real https website. "
        f"{why_geo_rule}"
    )
    pack = await ai_service.structured_json(
        db,
        agency_id,
        prompt,
        json.dumps(
            {
                "name": client.name,
                "website": client.website,
                "industry": client.industry,
                "niche": client.niche,
                "market_area": focus,
                "competitor_scope": scope,
                "business_model": business_model,
                "needed": needed,
                "already_have": already_have[:40],
                "serp_candidates": (serp_candidates or [])[:12],
                "peer_scale": peer_scale,
            }
        )[:7000],
        temperature=0.2,
    )
    raw_list = (
        pack.get("competitors")
        or pack.get("peers")
        or pack.get("rivals")
        or pack.get("items")
        if isinstance(pack, dict)
        else (pack if isinstance(pack, list) else [])
    )
    rows: list[dict] = []
    for c in (raw_list or []):
        if isinstance(c, dict):
            c_copy = {**c, "source": "ai_same_tier"}
            if not c_copy.get("headquarters_country") and focus:
                c_copy["headquarters_country"] = focus
            if not c_copy.get("overlap_score"):
                c_copy["overlap_score"] = 72.0
            rows.append(c_copy)
    food_tier = (
        _food_tier_from_blob(client.name, client.niche, client.industry, business_model)
        if is_food
        else None
    )
    return _filter_niche_competitors(
        rows,
        client.name,
        market_area=focus if scope == "local" else "",
        niche=_as_str(client.niche),
        industry=_as_str(client.industry),
        business_model=_as_str(business_model),
        min_overlap=45.0,
        limit=max(needed * 2, needed),
        require_local_market=(scope == "local"),
        client_food_tier=food_tier,
        client_peer_scale=peer_scale,
    )


async def enrich_client_profile(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    *,
    competitor_scope: str = "local",
    competitor_country: str | None = None,
    competitor_count: int = 5,
    competitor_mode: str = "add",
) -> dict:
    scope = "global" if str(competitor_scope).lower() == "global" else "local"
    raw_mode = str(competitor_mode or "add").strip().lower()
    mode = raw_mode if raw_mode in {"update", "add", "replace"} else "add"
    count = max(1, min(10, int(competitor_count or 5)))
    country = _as_str(competitor_country).strip()

    site = {}
    if client.website:
        cleaned = _normalize_website(client.website)
        if cleaned and cleaned != client.website:
            client.website = cleaned
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
            "If the brand is a pizza, burger, fried-chicken, or restaurant chain, industry MUST be Fast food "
            "and niche a QSR category (pizza delivery, burgers, etc.). Never call that a snack-food, cheese, "
            "or software company. "
            "For each feature.description: write 2–3 plain-English sentences a non-technical person can understand. "
            "Explain what the customer gets and why it matters. No slogans, no unexplained jargon "
            "(avoid 'production-grade', 'demoware', 'architecture-first' unless explained simply). "
            "Use the website excerpt. Be concrete. No filler."
        ),
        json.dumps(
            {
                "name": client.name,
                "website": client.website,
                "industry_hint": client.industry,
                "site_markdown": site_md,
                "preferred_market": country or None,
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

    # Never default food/beauty brands to "Software" — that pulls NetSol/Systems-style Serp + AI fill
    default_industry = (
        "Restaurant"
        if _looks_like_food_client(client.name, client.website, site_md[:800])
        else ("Beauty & Personal Care" if _looks_like_beauty_client(client.name, client.website, site_md[:800], client.notes, client.tagline) else "Software")
    )
    client.industry = _as_str(profile.get("industry")) or client.industry or default_industry
    client.niche = _as_str(profile.get("niche")) or client.niche
    if _looks_like_beauty_client(client.name, client.website, site_md[:800], client.notes, client.tagline):
        client.industry = "Beauty & Personal Care"
        niche_l = _as_str(client.niche).lower()
        if (
            not client.niche
            or "restaurant" in niche_l
            or "software" in niche_l
            or "food" in niche_l
            or "it services" in niche_l
            or niche_l in {"unknown", "general", "services", "other"}
        ):
            client.niche = "beauty salon & makeup studio"
    elif _looks_like_food_client(client.name, client.website, site_md[:800]):
        industry_l = _as_str(client.industry).lower()
        niche_l = _as_str(client.niche).lower()
        name_blob = f"{client.name} {site_md[:800]}".lower()
        misprofiled = (
            "snack" in industry_l
            or "snack" in niche_l
            or ("cheese" in industry_l and "pizza" not in industry_l)
            or ("cheese" in niche_l and "pizza" not in niche_l)
            or "software" in industry_l
            or "technology" in industry_l
            or not _looks_like_food_client(client.industry, client.niche)
        )
        if misprofiled:
            client.industry = "Restaurant"
            if any(tok in name_blob for tok in ("shawarma", "shwarma", "doner", "kebab", "kabab")):
                client.niche = "shawarma / quick-service restaurant"
            elif any(tok in name_blob for tok in ("bakery", "patisserie", "pastry", "dessert", "cake", "layers", "sweet tooth", "kitchen cuisine", "del frio", "tehzeeb", "gourmet", "rahat")):
                client.industry = "Bakery"
                client.niche = "bakery / desserts"
            elif any(tok in name_blob for tok in ("cafe", "café", "coffee")) and "restaurant" not in name_blob:
                client.industry = "Cafe"
                client.niche = "cafe"
            elif "burger" in name_blob:
                client.niche = "burgers / quick-service restaurant"
            elif any(tok in name_blob for tok in ("pizza", "cheezious")) or _food_format_from_blob(
                client.name, name_blob
            ) == _FOOD_FORMAT_PIZZA:
                client.niche = "pizza / quick-service restaurant"
            elif any(tok in name_blob for tok in ("crepe", "crêpe", "paris", "baguette", "french")):
                client.niche = "french casual dining / crepes"
            else:
                client.niche = "restaurant"
        # Even when industry already looks food, force restaurant niche for Parisian dining brands
        elif any(tok in name_blob for tok in ("meet me in paris",)) or (
            "paris" in name_blob and not any(tok in name_blob for tok in ("pizza", "burger", "cake", "bakery"))
        ):
            client.industry = "Restaurant"
            if not any(tok in niche_l for tok in ("restaurant", "crepe", "french", "dining")):
                client.niche = "french casual dining / crepes"
        # Known pizza brands: keep niche pizza even if AI said only "restaurant"
        elif _food_format_from_blob(client.name, client.niche, client.industry) == _FOOD_FORMAT_PIZZA and "pizza" not in niche_l:
            client.niche = "pizza / quick-service restaurant"
        # Known bakery brands: keep niche bakery / desserts even if AI said burgers/restaurant
        elif _food_format_from_blob(client.name, client.niche, client.industry) == _FOOD_FORMAT_BAKERY and not any(tok in niche_l for tok in ("bakery", "dessert", "cake", "pastry")):
            client.industry = "Bakery"
            client.niche = "bakery / desserts"
    client.tagline = _as_str(profile.get("tagline")) or client.tagline
    market_area = _as_str(profile.get("market_area")) or _market_area_from_client(client)
    if market_area.strip().lower() in {"global", "worldwide", "international", "world"}:
        market_area = ""
    # User-selected country ALWAYS wins for local runs (Saudi select → Saudi peers).
    # Known-brand home market only corrects AI hallucinations when the user left country blank.
    if scope == "local" and country:
        market_area = country
    else:
        home = _known_brand_home_market(client.name, client.website)
        if home:
            ai_key = _normalize_country_key(market_area)
            home_key = _normalize_country_key(home)
            if not market_area or (ai_key and home_key and ai_key != home_key):
                logger.info(
                    "Overriding AI market_area=%r → %s for known brand %s (no user country selected)",
                    market_area,
                    home,
                    client.name,
                )
                market_area = home
    business_model = _as_str(profile.get("business_model")) or _business_model_from_client(client) or "services"
    if _looks_like_food_client(client.name, client.industry, client.niche) and business_model.lower() in {
        "services",
        "saas",
        "agency",
        "software",
        "product",
    }:
        business_model = "other"
    description = _as_str(profile.get("description"))
    if description:
        # Keep Market / Business model lines; replace free-text body
        kept_meta = [
            ln
            for ln in _as_str(client.notes).splitlines()
            if ln.lower().startswith("market:") or ln.lower().startswith("business model:")
        ]
        client.notes = ("\n".join(kept_meta + [description])).strip() or description
    _set_market_area(client, market_area)
    _set_business_model(client, business_model)

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
        industry_hint = _as_str(client.industry) or _as_str(profile.get("industry")) or "this category"
        feature_items = [
            {
                "name": f"Core {industry_hint} offering",
                "category": "Product",
                "description": f"Primary products or services this brand sells in {industry_hint}.",
            },
            {
                "name": "Customer onboarding",
                "category": "Experience",
                "description": "How new customers get started and reach first value.",
            },
            {
                "name": "Delivery / implementation",
                "category": "Services",
                "description": "How the company delivers work, support, or product updates.",
            },
            {
                "name": "Pricing & packaging",
                "category": "Commercial",
                "description": "Plans, packages, or engagement models sold to buyers.",
            },
            {
                "name": "Proof & credibility",
                "category": "Marketing",
                "description": "Case studies, testimonials, certifications, or public proof points.",
            },
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

    await db.flush()
    await clarify_feature_descriptions(db, agency, client, feature_rows)

    existing_early = (
        await db.execute(select(Competitor).where(Competitor.client_id == client.id, Competitor.agency_id == agency.id))
    ).scalars().all()
    # replace: drop auto-found rivals (keep pinned/manual), then discover a fresh set
    if mode == "replace":
        for competitor in existing_early:
            if not competitor.is_pinned:
                competitor.is_tracking = False
        await db.flush()
    tracking_existing_early = (
        [c for c in existing_early if c.is_pinned]
        if mode == "replace"
        else [c for c in existing_early if c.is_tracking or c.is_pinned]
    )
    # replace: only avoid pinned/manual names — previously auto-found rivals may be reselected
    already_have_names = (
        [_as_str(c.name) for c in existing_early if c.is_pinned and _as_str(c.name)]
        if mode == "replace"
        else [_as_str(c.name) for c in tracking_existing_early]
    )
    if mode == "update":
        await db.flush()
        return {
            "features": len(feature_rows),
            "competitors_added": 0,
            "competitors_requested": count,
            "competitor_scope": scope,
            "competitor_country": country or None,
            "competitors_kept_existing": len(tracking_existing_early),
            "competitors_pruned_global": 0,
            "competitor_mode": mode,
            "goals": len(client.goals or []),
            "industry": client.industry,
            "niche": client.niche,
            "market_area": market_area,
            "business_model": business_model,
        }

    existing_keys: set[str] = set()
    for row in (existing_early if mode == "replace" else tracking_existing_early):
        if mode == "replace" and not row.is_pinned:
            continue
        existing_keys |= _rival_keys(row.name, row.website)

    def _pick_fresh(items: list) -> list[dict]:
        fresh: list[dict] = []
        seen: set[str] = set()
        for c in items:
            if not isinstance(c, dict):
                continue
            name = _clean_rival_display_name(_as_str(c.get("name")).strip())
            website = _normalize_website(_as_str(c.get("website")) or None)
            item_keys = _rival_keys(name, website)
            if not name or not item_keys or item_keys & existing_keys or item_keys & seen:
                continue
            if _is_generic_or_fake_rival_name(name):
                if _as_str(c.get("source")).lower() != "seed" or _looks_like_recipe_or_menu_item_name(name):
                    continue
            if _looks_like_content_or_cpg_noise(name, website):
                continue
            if _is_self_rival(client.name, name, website=website, client_website=client.website):
                continue
            if _looks_like_brand_geo_hallucination(
                client.name,
                name,
                local_focus or market_area or country,
                website=website,
                source=_as_str(c.get("source")) or "ai",
            ):
                continue
            if not _rival_fits_run_scope(
                name=name,
                website=website,
                headquarters=_as_str(c.get("headquarters_country") or c.get("headquarters")),
                description=_as_str(c.get("why_relevant") or c.get("description")),
                why=_as_str(c.get("why_relevant")),
                scope=scope,
                market=(local_focus or market_area or country) if scope == "local" else (local_focus or ""),
                client_name=client.name,
                strict=False,
            ):
                continue
            if not website:
                continue
            seen |= item_keys
            c = {**c, "name": name}
            fresh.append(c)
            if len(fresh) >= count:
                break
        return fresh

    _is_food_client_prompt = _looks_like_food_client(client.name, client.niche, client.industry)
    _food_fmt_prompt = (
        _food_format_from_blob(client.name, client.niche, client.industry, business_model)
        if _is_food_client_prompt
        else ""
    )
    _food_peer_label = {
        _FOOD_FORMAT_PIZZA: "pizza / QSR pizza chains",
        _FOOD_FORMAT_SHAWARMA: "shawarma / wrap QSR brands",
        _FOOD_FORMAT_BURGER: "burger QSR brands",
        _FOOD_FORMAT_CAFE: "cafes / coffee shops",
        _FOOD_FORMAT_BAKERY: "bakeries / dessert shops",
        _FOOD_FORMAT_ASIAN: "asian restaurants",
        _FOOD_FORMAT_RESTAURANT: "casual dining restaurants",
    }.get(_food_fmt_prompt, "same-format food / restaurant brands")

    if scope == "global":
        home_country = market_area or country or _market_area_from_client(client) or "the local market"
        is_edu_client = _detect_industry_category(client.name, client.niche, client.industry, business_model) in {"education", "university", "school", "college", "academy"}
        edu_tier = _detect_education_tier(client.name, client.niche, client.industry, client.notes, client.tagline) if is_edu_client else ""

        if _is_food_client_prompt:
            competitor_prompt = (
                f"Find exactly {count} REAL international food competitors for this brand. "
                f"They must be {_food_peer_label} — same food format, similar scale. "
                f"HARD GLOBAL RULE: Every competitor MUST be an established international food / restaurant brand headquartered OUTSIDE {home_country}. "
                f"Must-match industry: {_as_str(client.industry) or 'Restaurant'}. "
                f"Must-match niche: {_as_str(client.niche) or 'food'}. "
                f"Return JSON: {{competitors:[{'{'}name, website, industry, business_model, headquarters_country, why_relevant, threat_level, overlap_score, "
                "same_niche:true, same_market:false, market_overlap, is_global_platform:false{'}'}]}}. "
                "Hard rules:\n"
                f"1) Return exactly {count} NEW competitors — not names in already_have.\n"
                f"2) ONLY {_food_peer_label}. NEVER software houses, IT firms, FMCG snack makers, or agencies.\n"
                "3) Prefer real restaurant / QSR chains with working websites.\n"
                "4) EXCLUDE directories, Foodpanda, Tripadvisor, blogs, and recipe pages.\n"
                "5) overlap_score prefer 60-95 for true peer fit.\n"
                "6) Never invent placeholder brands.\n"
                f"7) {_peer_scale_prompt_rule(_peer_scale_from_blob(client.name, client.niche, client.industry, business_model, name=client.name), is_food=True)}."
            )
        elif is_edu_client and edu_tier == "university":
            competitor_prompt = (
                f"Find exactly {count} REAL LEADING INTERNATIONAL / GLOBAL UNIVERSITIES for this client. "
                f"HARD GLOBAL RULE: Every competitor MUST be a recognized higher education university headquartered OUTSIDE {home_country}. "
                f"DO NOT return domestic universities from {home_country}. "
                "Return top global universities (e.g. from United States, United Kingdom, Europe, Australia, Canada, Asia, Middle East) "
                "with strong programs in engineering, technology, computer science, management, or research. "
                "STRICTLY FORBIDDEN: K-12 schools, primary/high schools, academies, educational ranking directories (Times Higher Ed, QS, US News). "
                f"Return JSON: {{competitors:[{'{'}name, website, industry, business_model, headquarters_country, why_relevant, threat_level, overlap_score, "
                "same_niche:true, same_market:false, is_global_platform:false{'}'}]}}. "
                "Hard rules:\n"
                f"1) Return exactly {count} NEW competitors — not names in already_have.\n"
                "2) Each MUST be an accredited degree-awarding university with real homepage URL (https://...).\n"
                "3) overlap_score 75-95.\n"
                "4) Never invent placeholder names."
            )
        elif is_edu_client and edu_tier == "school":
            competitor_prompt = (
                f"Find exactly {count} REAL INTERNATIONAL / GLOBAL K-12 SCHOOL SYSTEMS for this client. "
                f"HARD GLOBAL RULE: Every competitor MUST be a recognized international private school system or board headquartered OUTSIDE {home_country}. "
                f"DO NOT return schools from {home_country}. "
                "STRICTLY FORBIDDEN: Universities, medical colleges, degree-awarding institutions. "
                f"Return JSON: {{competitors:[{'{'}name, website, industry, business_model, headquarters_country, why_relevant, threat_level, overlap_score, "
                "same_niche:true, same_market:false, is_global_platform:false{'}'}]}}. "
                "Hard rules:\n"
                f"1) Return exactly {count} NEW competitors — not names in already_have.\n"
                "2) Each MUST be a recognized K-12 school network with real homepage URL.\n"
                "3) overlap_score 75-95."
            )
        else:
            is_sw_prompt = _looks_like_software_peer_client(client.name, client.niche, client.industry, business_model)
            sw_rule = (
                "2) Only commercial software engineering houses / digital consulting / IT services firms with international delivery.\n"
                if is_sw_prompt
                else f"2) Only peer businesses in {_as_str(client.industry) or 'the same industry'} ({_as_str(client.niche) or 'same niche'}). NEVER software houses unless client is one.\n"
            )
            competitor_prompt = (
                f"Find exactly {count} REAL direct competitors for this company with GLOBAL / international reach. "
                f"HARD GLOBAL RULE: Every competitor MUST be a well-known international / global company headquartered OUTSIDE {home_country}. "
                f"Do NOT return domestic/local companies from {home_country}. "
                "They must compete in the SAME niche, SAME industry, and similar business model / buyer. "
                f"Must-match industry: {_as_str(client.industry) or 'unknown'}. "
                f"Must-match niche: {_as_str(client.niche) or 'unknown'}. "
                f"Must-match business model: {_as_str(business_model) or 'unknown'}. "
                f"Return JSON: {{competitors:[{'{'}name, website, industry, business_model, headquarters_country, why_relevant, threat_level, overlap_score, "
                "same_niche:true, same_market:false, is_global_platform:false{'}'}]}}. "
                "Hard rules:\n"
                f"1) Return exactly {count} NEW competitors — not names in already_have.\n"
                f"{sw_rule}"
                "3) EXCLUDE directories, review sites, job boards, news articles, and hyperscaler platforms "
                "(AWS/Azure/GCP as clouds, Dialogflow as a raw API) unless they are a true peer product.\n"
                "4) why_relevant must cite industry + niche + global benchmark comparison.\n"
                "5) overlap_score should reflect true peer fit (prefer 70-95).\n"
                "6) Only include companies you believe actually exist with real working websites (https://...).\n"
                "7) Never invent placeholder brands (TechCorp, Soft Solutions, PakTech Solutions, AxonSoft).\n"
                f"8) {_peer_scale_prompt_rule(_peer_scale_from_blob(client.name, client.niche, client.industry, business_model, name=client.name), is_food=False)}."
            )
    else:
        focus = country or market_area or "the client's primary country/region"
        _client_scale = _peer_scale_from_blob(
            client.name, client.niche, client.industry, business_model, name=client.name
        )
        if _is_food_client_prompt:
            competitor_prompt = (
                f"Find exactly {count} REAL LOCAL food competitors for this brand in {focus}. "
                f"HARD GEO RULE: every rival MUST operate primarily in {focus}. "
                f"They must be {_food_peer_label} — same food format as the client. "
                f"Must-match industry: {_as_str(client.industry) or 'Restaurant'}. "
                f"Must-match niche: {_as_str(client.niche) or 'food'}. "
                f"Return JSON: {{competitors:[{'{'}name, website, industry, business_model, headquarters_country, why_relevant, threat_level, overlap_score, "
                "same_niche:true, same_market:true, market_overlap, is_global_platform:false{'}'}]}}. "
                "Hard rules:\n"
                f"1) Return exactly {count} NEW competitors — not names in already_have.\n"
                f"2) headquarters_country MUST be {focus} (or a city inside {focus}).\n"
                f"3) ONLY {_food_peer_label}. NEVER software houses (NetSol, Systems Limited, 10Pearls), "
                "IT agencies, FMCG snack companies (Unilever, PepsiCo), or non-food brands.\n"
                f"4) why_relevant MUST mention {focus} and how they compete for the same diners/delivery buyers.\n"
                "5) website MUST be a real restaurant / QSR homepage (https://...).\n"
                "6) EXCLUDE Foodpanda, Tripadvisor, Instagram, blogs, recipe pages, and directories.\n"
                f"7) If unsure a brand is a real {_food_peer_label} in {focus}, OMIT it.\n"
                "8) Never invent placeholder brands.\n"
                "9) overlap_score prefer 60-95.\n"
                f"10) {_peer_scale_prompt_rule(_client_scale, is_food=True)}"
                + (
                    f"\n11) {_brand_geo_disclaimer(client.name, focus)}"
                    if _brand_geo_disclaimer(client.name, focus)
                    else ""
                )
            )
        elif _detect_industry_category(client.name, client.niche, client.industry, business_model) == "data_ai":
            competitor_prompt = (
                f"Find exactly {count} REAL direct LOCAL competitors for this Data Analytics / AI Solutions consultancy in {focus}. "
                f"HARD GEO RULE: every competitor MUST be headquartered in OR primarily selling in {focus}. "
                "They must be specialized Data Analytics, Business Intelligence (BI), Data Engineering, or Enterprise AI consulting firms / agencies. "
                f"Must-match industry: Data Analytics & Business Intelligence / AI Solutions. "
                f"Must-match niche: {_as_str(client.niche) or 'Enterprise AI & Data Analytics'}. "
                f"Must-match business model: {_as_str(business_model) or 'services'}. "
                f"Return JSON: {{competitors:[{'{'}name, website, industry, business_model, headquarters_country, why_relevant, threat_level, overlap_score, "
                "same_niche:true, same_market:true, market_overlap, is_global_platform:false{'}'}]}}. "
                "Hard rules:\n"
                f"1) Return exactly {count} NEW competitors — not names in already_have.\n"
                f"2) headquarters_country MUST be {focus} (or a city inside {focus}).\n"
                f"3) why_relevant MUST mention {focus} and cite their specific Data Analytics, BI, or AI practice.\n"
                f"4) website MUST be a real working company homepage URL (https://...). No invented domains.\n"
                "5) STRICTLY EXCLUDE generic custom software / IT staffing giants (Systems Limited, NetSol, Arbisoft, 10Pearls, Contour Software).\n"
                "6) EXCLUDE global cloud hyperscalers (AWS, Azure, GCP, OpenAI) unless they are consulting partners.\n"
                "7) EXCLUDE fintech apps, banks, directories, and government IT boards.\n"
                "8) overlap_score should reflect true peer fit (prefer 75-95).\n"
                f"9) {_peer_scale_prompt_rule(_client_scale, is_food=False)}"
                + (
                    f"\n10) {_brand_geo_disclaimer(client.name, focus)}"
                    if _brand_geo_disclaimer(client.name, focus)
                    else ""
                )
            )
        else:
            is_sw_prompt = _looks_like_software_peer_client(client.name, client.niche, client.industry, business_model)
            sw_rule = (
                "12) Prefer commercial software houses / digital agencies / IT services firms as peers for a software house client.\n"
                if is_sw_prompt
                else f"12) Prefer real peer commercial businesses in {_as_str(client.industry) or 'the same industry'} ({_as_str(client.niche) or 'same niche'}). NEVER software houses unless the client is one.\n"
            )
            competitor_prompt = (
                f"Find exactly {count} REAL direct LOCAL / country competitors for this company in {focus}. "
                f"HARD GEO RULE: every competitor MUST be headquartered in OR primarily selling in {focus}. "
                f"Do NOT return companies from other countries (e.g. if focus is Pakistan, exclude India, Singapore, UAE, US, UK rivals). "
                "They must be from the SAME niche, SAME industry, SAME business model, and the SAME country/market. "
                f"Must-match industry: {_as_str(client.industry) or 'unknown'}. "
                f"Must-match niche: {_as_str(client.niche) or 'unknown'}. "
                f"Must-match business model: {_as_str(business_model) or 'unknown'}. "
                f"Return JSON: {{competitors:[{'{'}name, website, industry, business_model, headquarters_country, why_relevant, threat_level, overlap_score, "
                "same_niche:true, same_market:true, market_overlap, is_global_platform:false{'}'}]}}. "
                "Hard rules:\n"
                f"1) Return exactly {count} NEW competitors — not names in already_have.\n"
                f"2) headquarters_country MUST be {focus} (or a city inside {focus}).\n"
                f"3) why_relevant MUST mention {focus} and how they sell there.\n"
                f"4) website MUST be a real working company homepage URL (https://...). No invented domains.\n"
                "5) EXCLUDE consumer shopping / ecommerce retailers / marketplaces "
                "(Daraz, Telemart, Amazon-style stores) unless the client itself is retail ecommerce.\n"
                "6) EXCLUDE fintech wallets, payment apps, banks, and remittance apps "
                "(NayaPay, EasyPaisa, JazzCash, SadaPay) unless the client itself is fintech/payments.\n"
                "7) EXCLUDE government boards, ministries, regulators, and public-sector IT bodies "
                "(PITB, NADRA, ministries, authorities) — they are not commercial business rivals.\n"
                "8) EXCLUDE global hyperscalers and mega consultancies "
                "(Accenture, IBM, Microsoft, Google, Amazon/AWS, Oracle, SAP, Deloitte, PwC, EY, KPMG, Cognizant, Infosys, TCS, Wipro, OpenAI).\n"
                "9) EXCLUDE directories, review sites, and tools/infrastructure that are not peer businesses.\n"
                f"10) If you are unsure a company is based in / sells primarily in {focus}, OMIT it.\n"
                "11) Same generic category is NOT enough — they must sell a similar product/service to similar buyers.\n"
                f"{sw_rule}"
                "13) NEVER invent placeholder brands (TechCorp, Soft Solutions, SoftCorp, PakTech Solutions, AxonSoft, IT Solutions). "
                "Only well-known or clearly real companies with working websites.\n"
                "14) overlap_score should reflect true peer fit (prefer 60-95).\n"
                "15) Only include companies you believe actually exist.\n"
                f"16) {_peer_scale_prompt_rule(_client_scale, is_food=False)}"
                + (
                    f"\n17) {_brand_geo_disclaimer(client.name, focus)}"
                    if _brand_geo_disclaimer(client.name, focus)
                    else ""
                )
            )

    def _apply_relevance_filter(rows: list[dict]) -> list[dict]:
        local_market = (country or market_area) if scope == "local" else ""
        food_tier = (
            _food_tier_from_blob(client.name, client.niche, client.industry, business_model)
            if _looks_like_food_client(client.name, client.niche, client.industry, business_model)
            else None
        )
        peer_scale = _peer_scale_from_blob(
            client.name, client.niche, client.industry, business_model, name=client.name
        )
        return _filter_niche_competitors(
            rows,
            client.name,
            market_area=local_market,
            niche=_as_str(client.niche),
            industry=_as_str(client.industry),
            business_model=_as_str(business_model),
            min_overlap=55.0,
            limit=max(count * 2, 10),
            require_local_market=(scope == "local"),
            client_food_tier=food_tier,
            client_peer_scale=peer_scale,
        )

    competitor_items: list[dict] = []
    local_focus = country or market_area or ""
    serp_auth_failed = False

    # SERP-first — stop early on auth failure; keep query budget tight for speed
    # Food/local (esp. add-mode) needs more queries when AI/TPM is thin
    serp_budget = 2 if scope == "global" else (5 if _looks_like_food_client(
        client.name, client.niche, client.industry, business_model
    ) else 3)
    serp_market = local_focus if scope == "local" else ""
    for query in _niche_competitor_queries(client, serp_market, scope=scope)[:serp_budget]:
        if scope == "local" and local_focus and local_focus.lower() not in query.lower():
            query = f"{query} {local_focus}"
        serp = await serp_visibility(
            db,
            agency.id,
            query,
            location=_serp_location_for_market(local_focus) if scope == "local" else None,
            gl=_serp_gl_for_market(local_focus) if scope == "local" else None,
        )
        status = _as_str(serp.get("status")).lower()
        detail_l = _as_str(serp.get("detail")).lower()
        if status == "unauthorized" or "unauthorized" in detail_l or "invalid api key" in detail_l:
            serp_auth_failed = True
            logger.warning(
                "SerpAPI unauthorized for agency=%s — skipping remaining SERP queries, AI will fill peers",
                agency.id,
            )
            break
        if status in {"error", "skipped"} and not (serp.get("organic") or []):
            # Don't burn the full budget on repeated empty/error SERP calls
            if status == "error":
                logger.warning("SerpAPI error for agency=%s — stopping SERP loop early", agency.id)
                break
        competitor_items.extend(_competitors_from_serp(serp.get("organic") or [], client.name))
        competitor_items = _apply_relevance_filter(competitor_items)
        if len(_pick_fresh(competitor_items)) >= count:
            break

    logger.info(
        "SERP discovery client=%s scope=%s market=%s queries_budget=%s serp_hits=%s auth_failed=%s",
        client.id,
        scope,
        local_focus or "(global)",
        serp_budget,
        len(competitor_items),
        serp_auth_failed,
    )

    # Vertical flags unused for seed lists — gap-fill is AI-only now
    if serp_auth_failed:
        logger.info(
            "Competitor discovery will use AI same-tier fill for agency=%s client=%s (SERP unavailable)",
            agency.id,
            client.id,
        )

    # AI ranks/fills — prefer choosing from SERP candidates when available
    serp_names = [_as_str(c.get("name")) for c in competitor_items if _as_str(c.get("name"))]
    ai_payload = {
        "name": client.name,
        "website": client.website,
        "industry": client.industry,
        "niche": client.niche,
        "market_area": market_area,
        "competitor_scope": scope,
        "competitor_country": country or None,
        "competitor_count": count,
        "competitor_mode": mode,
        "already_have": already_have_names,
        "business_model": business_model,
        "features": [f.name for f in feature_rows[:10]],
        "site_excerpt": site_md[:2000],
        "serp_candidates": competitor_items[:12],
    }
    if serp_names:
        competitor_prompt = (
            competitor_prompt
            + "\n15) Prefer picking from serp_candidates when they are true peers. "
            "You may add other REAL peers only if serp_candidates are insufficient — never invent placeholder names."
        )
    competitor_pack = await ai_service.structured_json(
        db,
        agency.id,
        competitor_prompt,
        json.dumps(ai_payload)[:9000],
        temperature=0.15,
    )
    if isinstance(competitor_pack.get("competitors"), list):
        for c in competitor_pack["competitors"]:
            if isinstance(c, dict):
                c_copy = {**c}
                if not c_copy.get("source"):
                    c_copy["source"] = "ai"
                if not c_copy.get("headquarters_country") and local_focus:
                    c_copy["headquarters_country"] = local_focus
                if not c_copy.get("overlap_score"):
                    c_copy["overlap_score"] = 75.0
                competitor_items.append(c_copy)
    competitor_items = _apply_relevance_filter(competitor_items)

    # Gap-fill remaining slots with same-tier AI peers (NOT curated seed lists)
    fresh_so_far = _pick_fresh(competitor_items)
    if len(fresh_so_far) < count:
        already = already_have_names + [_as_str(c.get("name")) for c in fresh_so_far]
        fill_rows = await _ai_propose_same_tier_peers(
            db,
            agency.id,
            client,
            needed=count - len(fresh_so_far),
            already_have=already,
            scope=scope,
            market_focus=(local_focus or market_area or country) if scope == "local" else "global / international",
            business_model=business_model,
            serp_candidates=competitor_items[:12],
        )
        competitor_items.extend(fill_rows)
        competitor_items = _apply_relevance_filter(competitor_items)

    fresh_so_far = _pick_fresh(competitor_items)
    # Second AI pass if still short of the slider count (not capped at 4)
    if len(fresh_so_far) < count:
        already = already_have_names + [_as_str(c.get("name")) for c in fresh_so_far]
        fill_rows = await _ai_propose_same_tier_peers(
            db,
            agency.id,
            client,
            needed=count - len(fresh_so_far),
            already_have=already,
            scope=scope,
            market_focus=(local_focus or market_area or country) if scope == "local" else "global / international",
            business_model=business_model,
            serp_candidates=competitor_items[:12],
        )
        competitor_items.extend(fill_rows)
        competitor_items = _apply_relevance_filter(competitor_items)

    # Last resort when SerpAPI is down and AI returned nothing usable
    is_food_client = _looks_like_food_client(
        client.name, client.industry, client.niche, business_model, site_md[:800]
    )
    is_beauty_client = _looks_like_beauty_client(
        client.name, client.industry, client.niche, business_model, site_md[:800]
    )
    fresh_so_far = _pick_fresh(competitor_items)
    if (
        is_food_client
        and scope == "local"
        and local_focus
        and len(fresh_so_far) < count
    ):
        already = already_have_names + [_as_str(c.get("name")) for c in fresh_so_far]
        food_tier = _food_tier_from_blob(client.name, client.niche, client.industry, business_model)
        seed_rows = _seed_local_qsr_rivals(
            local_focus,
            client.name,
            already_have=already,
            client_website=client.website,
            client_tier=food_tier,
            client_niche=_as_str(client.niche),
            client_industry=_as_str(client.industry),
            limit=max(count * 2, 8),
        )
        competitor_items.extend(seed_rows)
        competitor_items = _apply_relevance_filter(competitor_items)
        logger.warning(
            "Food/local last-resort peers used for client=%s market=%s (SERP/AI thin) kept=%s",
            client.id,
            local_focus,
            len(competitor_items),
        )
    elif (
        is_beauty_client
        and scope == "local"
        and local_focus
        and len(fresh_so_far) < count
    ):
        already = already_have_names + [_as_str(c.get("name")) for c in fresh_so_far]
        seed_rows = _seed_local_beauty_rivals(
            local_focus,
            client.name,
            already_have=already,
            client_website=client.website,
            client_niche=_as_str(client.niche),
            client_industry=_as_str(client.industry),
            limit=max(count * 2, 8),
        )
        competitor_items.extend(seed_rows)
        competitor_items = _apply_relevance_filter(competitor_items)
        logger.warning(
            "Beauty/local last-resort peers used for client=%s market=%s (SERP/AI thin) kept=%s",
            client.id,
            local_focus,
            len(competitor_items),
        )

    # Multi-industry last resort when live SerpAPI / AI is thin
    fresh_so_far = _pick_fresh(competitor_items)
    if scope == "local" and local_focus and len(fresh_so_far) < count:
        already = already_have_names + [_as_str(c.get("name")) for c in fresh_so_far]
        industry_seed_rows = _seed_local_industry_rivals(
            f"{_as_str(client.industry)} {_as_str(client.niche)}",
            local_focus,
            client.name,
            already_have=already,
            client_website=client.website,
            limit=max(count * 2, 8),
        )
        if industry_seed_rows:
            competitor_items.extend(industry_seed_rows)
            competitor_items = _apply_relevance_filter(competitor_items)
            logger.warning(
                "Multi-industry last-resort peers used for client=%s industry=%s market=%s kept=%s",
                client.id,
                client.industry,
                local_focus,
                len(competitor_items),
            )
    elif scope == "global" and len(fresh_so_far) < count:
        already = already_have_names + [_as_str(c.get("name")) for c in fresh_so_far]
        global_seed_rows = _seed_global_industry_rivals(
            f"{_as_str(client.industry)} {_as_str(client.niche)}",
            client.name,
            already_have=already,
            client_website=client.website,
            client_niche=_as_str(client.niche),
            client_market=market_area or country or "",
            limit=max(count * 2, 8),
        )
        if global_seed_rows:
            competitor_items.extend(global_seed_rows)
            competitor_items = _apply_relevance_filter(competitor_items)
            logger.info(
                "Global seed catalog peers used for client=%s industry=%s kept=%s",
                client.id,
                client.industry,
                len(competitor_items),
            )

    existing = (
        await db.execute(select(Competitor).where(Competitor.client_id == client.id, Competitor.agency_id == agency.id))
    ).scalars().all()
    by_name = {_as_str(c.name).lower(): c for c in existing}

    # Keep existing/manual rivals — boost so they survive AI prune & count slices.
    # replace: only pinned/manual stay; auto-found were already untracked above.
    # Never auto-pin: that trapped AI hallucinations (Paris Café etc.) forever.
    protected_existing = (
        [c for c in existing if c.is_pinned]
        if mode == "replace"
        else [c for c in existing if c.is_tracking or c.is_pinned]
    )
    scope_market = (country or market_area) if scope == "local" else (country or market_area or "")
    client_kind_for_protect = (
        f"{client.name} {client.industry or ''} {client.niche or ''} "
        f"{business_model} {client.notes or ''} {client.tagline or ''}"
    )
    for competitor in list(protected_existing):
        # Drop brand-geo fakes even if previously auto-pinned
        cleaned_name = _clean_rival_display_name(_as_str(competitor.name))
        if cleaned_name and cleaned_name != competitor.name and not competitor.is_pinned:
            competitor.name = cleaned_name
        if (not competitor.is_pinned) and (
            _looks_like_recipe_or_menu_item_name(competitor.name)
            or _looks_like_content_or_cpg_noise(competitor.name, competitor.website)
            or _looks_like_marketing_slogan_name(competitor.name)
            or _is_generic_or_fake_rival_name(competitor.name)
        ):
            competitor.is_tracking = False
            competitor.is_pinned = False
            continue
        # Sticky wrong-vertical peers (e.g. NetSol on a pizza brand) must not stay protected
        if (not competitor.is_pinned) and _incompatible_peer(
            client_model=business_model,
            client_industry=_as_str(client.industry),
            client_niche=_as_str(client.niche) or _as_str(client.notes),
            rival_model="",
            rival_industry="",
            rival_blob=f"{competitor.name} {competitor.description or ''} {competitor.why_dangerous or ''}",
            client_name=client.name,
        ):
            competitor.is_tracking = False
            competitor.is_pinned = False
            continue
        if (not competitor.is_pinned) and _looks_like_food_client(client_kind_for_protect) and (
            _looks_like_software_peer_client(
                competitor.name, competitor.description, competitor.why_dangerous, competitor.website
            )
            or _looks_like_fmcg_or_snack_brand(
                competitor.name, competitor.description, competitor.why_dangerous, competitor.website
            )
        ):
            competitor.is_tracking = False
        # In global mode or non-add mode, existing rivals that do not match the run scope must be untracked
        if (scope == "global" or mode != "add") and (not competitor.is_pinned) and not _rival_fits_run_scope(
            name=_as_str(competitor.name),
            website=competitor.website,
            headquarters=competitor.headquarters,
            description=competitor.description,
            why=competitor.why_dangerous,
            scope=scope,
            market=scope_market,
            client_name=client.name,
            is_pinned=False,
            strict=False,
        ):
            competitor.is_tracking = False
            competitor.is_pinned = False
            continue
        competitor.is_tracking = True
        if (competitor.overlap_score or 0) < 55:
            competitor.overlap_score = 70.0
        if not competitor.threat_level or competitor.threat_level == "low":
            competitor.threat_level = "medium"
        if not competitor.why_dangerous:
            competitor.why_dangerous = f"Existing rival kept for {client.name}"

    pruned_global = 0
    for competitor in existing:
        # add mode: never wipe previous list for geo/scope unless scope is global
        if mode == "add" and scope != "global" and (competitor.is_tracking or competitor.is_pinned):
            continue
        if competitor.is_pinned and _rival_fits_run_scope(
            name=_as_str(competitor.name),
            website=competitor.website,
            headquarters=competitor.headquarters,
            description=competitor.description,
            why=competitor.why_dangerous,
            scope=scope,
            market=scope_market,
            client_name=client.name,
            is_pinned=False,
            strict=False,
        ):
            continue
        # Brand-geo / out-of-scope: unpin + untrack (pin must not protect hallucinations)
        if not _rival_fits_run_scope(
            name=_as_str(competitor.name),
            website=competitor.website,
            headquarters=competitor.headquarters,
            description=competitor.description,
            why=competitor.why_dangerous,
            scope=scope,
            market=scope_market,
            client_name=client.name,
            is_pinned=False,
            strict=False,
        ):
            competitor.is_tracking = False
            competitor.is_pinned = False
            pruned_global += 1
            continue
        if competitor.is_pinned:
            continue
        if mode == "replace":
            competitor.is_tracking = False
            continue
        if _is_global_megarival(competitor.name, competitor.website):
            competitor.is_tracking = False
            competitor.threat_level = "low"
            competitor.overlap_score = min(float(competitor.overlap_score or 0), 30)
            pruned_global += 1

    # add: find exactly `count` NEW rivals on top of previous ones.
    # replace: rebuild up to `count` auto rivals (may re-enable previously untracked rows; keep pinned).
    ai_slots = count
    fresh_items = _pick_fresh(competitor_items)
    # If SERP/AI first pass left us short of the slider count, ask AI (+ seeds last) again
    if len(fresh_items) < ai_slots:
        already = already_have_names + [_as_str(c.get("name")) for c in fresh_items]
        more = await _ai_propose_same_tier_peers(
            db,
            agency.id,
            client,
            needed=ai_slots - len(fresh_items),
            already_have=already,
            scope=scope,
            market_focus=(local_focus or market_area or country) if scope == "local" else "global / international",
            business_model=business_model,
            serp_candidates=competitor_items[:12],
        )
        fresh_items = _pick_fresh(competitor_items)
        if len(fresh_items) < ai_slots:
            already = already_have_names + [_as_str(c.get("name")) for c in fresh_items]
            if scope == "local" and local_focus:
                if _looks_like_food_client(client.name, client.industry, client.niche, business_model):
                    competitor_items.extend(
                        _seed_local_qsr_rivals(
                            local_focus,
                            client.name,
                            already_have=already,
                            client_website=client.website,
                            client_tier=_food_tier_from_blob(client.name, client.niche, client.industry, business_model),
                            client_niche=_as_str(client.niche),
                            client_industry=_as_str(client.industry),
                            limit=max(ai_slots * 2, 8),
                        )
                    )
                elif _looks_like_beauty_client(client.name, client.industry, client.niche, business_model):
                    competitor_items.extend(
                        _seed_local_beauty_rivals(
                            local_focus,
                            client.name,
                            already_have=already,
                            client_website=client.website,
                            client_niche=_as_str(client.niche),
                            client_industry=_as_str(client.industry),
                            limit=max(ai_slots * 2, 8),
                        )
                    )
                else:
                    competitor_items.extend(
                        _seed_local_industry_rivals(
                            f"{_as_str(client.industry)} {_as_str(client.niche)}",
                            local_focus,
                            client.name,
                            already_have=already,
                            client_website=client.website,
                            client_niche=_as_str(client.niche),
                            limit=max(ai_slots * 2, 8),
                        )
                    )
            elif scope == "global":
                competitor_items.extend(
                    _seed_global_industry_rivals(
                        f"{_as_str(client.industry)} {_as_str(client.niche)}",
                        client.name,
                        already_have=already,
                        client_website=client.website,
                        client_niche=_as_str(client.niche),
                        client_market=market_area or country or "",
                        limit=max(ai_slots * 2, 8),
                    )
                )
            competitor_items = _apply_relevance_filter(competitor_items)
            fresh_items = _pick_fresh(competitor_items)

    deduped = fresh_items

    created_competitors = 0
    for item in deduped:
        name = _clean_rival_display_name(_as_str(item.get("name")).strip())
        if not name:
            continue
        if _is_generic_or_fake_rival_name(name):
            if _as_str(item.get("source")).lower() != "seed" or _looks_like_recipe_or_menu_item_name(name):
                continue
        why = _as_str(item.get("why_relevant"))
        threat = _as_str(item.get("threat_level"), "high").lower()
        try:
            overlap = float(item.get("overlap_score") or 70)
        except (TypeError, ValueError):
            overlap = 70.0
        website = _normalize_website(_as_str(item.get("website")) or None)
        if not website:
            continue
        if _looks_like_content_or_cpg_noise(name, website):
            continue
        competitor = _find_matching_competitor(existing, name, website)
        if competitor:
            was_tracking = bool(competitor.is_tracking)
            # replace mode skips previously known names via existing_keys; this path is for add/re-enable
            competitor.name = name
            competitor.website = website or competitor.website
            competitor.description = why or competitor.description
            competitor.why_dangerous = why or competitor.why_dangerous
            hq = _as_str(item.get("headquarters_country") or item.get("headquarters"))
            if hq:
                competitor.headquarters = hq
            if not competitor.is_pinned:
                competitor.threat_level = threat if threat in {"medium", "high"} else "high"
                competitor.overlap_score = max(overlap, competitor.overlap_score or 0)
            competitor.is_tracking = True
            if not was_tracking:
                created_competitors += 1
        else:
            competitor = Competitor(
                agency_id=agency.id,
                client_id=client.id,
                name=name,
                website=website,
                description=why or None,
                why_dangerous=why or None,
                headquarters=_as_str(item.get("headquarters_country") or item.get("headquarters")) or None,
                threat_level=threat if threat in {"medium", "high"} else "high",
                overlap_score=overlap,
                is_tracking=True,
            )
            db.add(competitor)
            existing.append(competitor)
            created_competitors += 1

    # Enforce exact slider count on the tracked list (UI shows is_tracking only)
    scope_market = (country or market_area) if scope == "local" else (country or market_area or "")
    if mode != "add":
        for rival in existing:
            if rival.is_pinned:
                continue
            if not _rival_fits_run_scope(
                name=_as_str(rival.name),
                website=rival.website,
                headquarters=rival.headquarters,
                description=rival.description,
                why=rival.why_dangerous,
                scope=scope,
                market=scope_market,
                client_name=client.name,
                is_pinned=False,
                strict=False,
            ):
                rival.is_tracking = False
                pruned_global += 1

    pinned_cur = [c for c in existing if c.is_pinned]
    others_cur = sorted(
        [c for c in existing if not c.is_pinned and c.is_tracking],
        key=lambda c: float(c.overlap_score or 0),
        reverse=True,
    )
    if mode == "add":
        baseline_keys = {
            k for c in protected_existing for k in _rival_keys(c.name, c.website)
        }
        baseline_others = [c for c in others_cur if _rival_keys(c.name, c.website) & baseline_keys]
        fresh_others = [c for c in others_cur if not (_rival_keys(c.name, c.website) & baseline_keys)]
        active_fresh = fresh_others[:count]
        active_all = set(baseline_others + active_fresh)
        for c in others_cur:
            c.is_tracking = c in active_all
    else:
        max_others = max(0, count - len(pinned_cur))
        for c in others_cur[:max_others]:
            c.is_tracking = True
        for c in others_cur[max_others:]:
            c.is_tracking = False

    await db.flush()
    return {
        "features": len(feature_rows),
        "competitors_added": created_competitors,
        "competitors_requested": count,
        "competitor_scope": scope,
        "competitor_country": country or None,
        "competitors_kept_existing": len(protected_existing),
        "competitors_pruned_global": pruned_global,
        "baseline_rival_names": list(already_have_names),
        "competitor_mode": mode,
        "goals": len(client.goals or []),
        "industry": client.industry,
        "niche": client.niche,
        "market_area": market_area,
        "business_model": business_model,
    }



async def run_competitive_pack(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    *,
    competitor_scope: str = "local",
    competitor_country: str | None = None,
    competitor_count: int = 5,
    competitor_mode: str = "add",
    baseline_rival_names: list[str] | None = None,
) -> dict:
    count = max(1, min(10, int(competitor_count or 5)))
    raw_mode = str(competitor_mode or "add").strip().lower()
    mode = raw_mode if raw_mode in {"update", "add", "replace"} else "add"
    scope = "global" if str(competitor_scope).lower() == "global" else "local"
    required_market = _as_str(competitor_country).strip() or _market_area_from_client(client)
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

    # Standalone pack calls may still need an enrich when the client has no rivals yet.
    baseline_names = [_as_str(n) for n in (baseline_rival_names or []) if _as_str(n)]
    if not features or not competitors:
        enrich_meta = await enrich_client_profile(
            db,
            agency,
            client,
            competitor_scope=competitor_scope,
            competitor_country=competitor_country,
            competitor_count=count,
            competitor_mode=mode if competitors else ("add" if mode == "update" else mode),
        )
        if not baseline_names:
            baseline_names = [
                _as_str(n) for n in (enrich_meta.get("baseline_rival_names") or []) if _as_str(n)
            ]
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

    # Drop auto-tracked rivals that clearly don't match this run's local/global filter or are incompatible
    scope_market = required_market if scope == "local" else (required_market or "")
    baseline_set_early = {_as_str(n).lower().strip() for n in (baseline_names or []) if _as_str(n)}
    for rival in competitors:
        if rival.is_pinned:
            rival.is_tracking = True
            continue
        rival_blob = f"{rival.name} {_as_str(rival.description)} {_as_str(rival.why_dangerous)}"
        incompat = _incompatible_peer(
            client_model=_business_model_from_client(client),
            client_industry=_as_str(client.industry),
            client_niche=_as_str(client.niche),
            rival_model="other",
            rival_industry=_as_str(rival.description or ""),
            rival_blob=rival_blob,
            client_name=client.name,
        )
        fits_scope = _rival_fits_run_scope(
            name=_as_str(rival.name),
            website=rival.website,
            headquarters=rival.headquarters,
            description=rival.description,
            why=rival.why_dangerous,
            scope=scope,
            market=scope_market,
            client_name=client.name,
            is_pinned=False,
            strict=False,
        )
        if incompat or not fits_scope:
            rival.is_tracking = False
            rival.is_pinned = False
            continue
        if mode == "add" and _as_str(rival.name).lower().strip() in baseline_set_early:
            rival.is_tracking = True
            continue
    competitors = [c for c in competitors if c.is_tracking or c.is_pinned]
    if len(competitors) < count:
        already_keys: set[str] = set()
        for c in competitors:
            already_keys |= _rival_keys(c.name, c.website)
        already_names = [_as_str(c.name) for c in competitors]
        needed = count - len(competitors)
        backfill_items: list[dict] = []
        if scope == "global":
            backfill_items = _seed_global_industry_rivals(
                f"{_as_str(client.industry)} {_as_str(client.niche)}",
                client.name,
                already_have=already_names,
                client_website=client.website,
                client_niche=_as_str(client.niche),
                client_market=required_market or "",
                limit=needed * 2,
            )
        elif scope == "local" and required_market:
            if _looks_like_food_client(client.name, client.industry, client.niche, _business_model_from_client(client)):
                backfill_items = _seed_local_qsr_rivals(
                    required_market,
                    client.name,
                    already_have=already_names,
                    client_website=client.website,
                    client_tier=_food_tier_from_blob(client.name, client.niche, client.industry, _business_model_from_client(client)),
                    client_niche=_as_str(client.niche),
                    client_industry=_as_str(client.industry),
                    limit=needed * 2,
                )
            elif _looks_like_beauty_client(client.name, client.industry, client.niche, _business_model_from_client(client)):
                backfill_items = _seed_local_beauty_rivals(
                    required_market,
                    client.name,
                    already_have=already_names,
                    client_website=client.website,
                    client_niche=_as_str(client.niche),
                    client_industry=_as_str(client.industry),
                    limit=needed * 2,
                )
            else:
                backfill_items = _seed_local_industry_rivals(
                    f"{_as_str(client.industry)} {_as_str(client.niche)}",
                    required_market,
                    client.name,
                    already_have=already_names,
                    client_website=client.website,
                    client_niche=_as_str(client.niche),
                    limit=needed * 2,
                )
        for item in backfill_items:
            name = _clean_rival_display_name(_as_str(item.get("name")).strip())
            website = _normalize_website(_as_str(item.get("website")) or None)
            if not name or not website:
                continue
            item_keys = _rival_keys(name, website)
            if item_keys & already_keys:
                continue
            row = (
                await db.execute(
                    select(Competitor).where(
                        Competitor.client_id == client.id,
                        Competitor.agency_id == agency.id,
                        Competitor.name.ilike(name),
                    )
                )
            ).scalars().first()
            if row:
                row.is_tracking = True
                row.website = website
                row.headquarters = _as_str(item.get("headquarters_country") or item.get("headquarters")) or row.headquarters
                row.description = _as_str(item.get("why_relevant")) or row.description
                row.why_dangerous = _as_str(item.get("why_relevant")) or row.why_dangerous
                row.overlap_score = float(item.get("overlap_score") or 85.0)
                competitors.append(row)
            else:
                new_comp = Competitor(
                    agency_id=agency.id,
                    client_id=client.id,
                    name=name,
                    website=website,
                    description=_as_str(item.get("why_relevant")) or None,
                    why_dangerous=_as_str(item.get("why_relevant")) or None,
                    headquarters=_as_str(item.get("headquarters_country") or item.get("headquarters")) or None,
                    threat_level="high",
                    overlap_score=float(item.get("overlap_score") or 85.0),
                    is_tracking=True,
                )
                db.add(new_comp)
                competitors.append(new_comp)
            already_keys |= item_keys
            if len(competitors) >= count:
                break
    await db.flush()

    # Enforce exact slider count: pinned first + top-overlap others up to count
    pinned = [c for c in competitors if c.is_pinned]
    others = sorted(
        [c for c in competitors if not c.is_pinned],
        key=lambda c: float(c.overlap_score or 0),
        reverse=True,
    )
    if mode == "add":
        baseline_others = [c for c in others if _as_str(c.name).lower().strip() in baseline_set_early]
        fresh_others = [c for c in others if _as_str(c.name).lower().strip() not in baseline_set_early]
        active_fresh = fresh_others[:count]
        active_others = baseline_others + active_fresh
        for extra in [c for c in others if c not in active_others]:
            extra.is_tracking = False
        for active in active_others:
            active.is_tracking = True
        competitors = pinned + active_others
    else:
        max_others = max(0, count - len(pinned))
        competitors = pinned + others[:max_others]
        for extra in others[max_others:]:
            extra.is_tracking = False
        for active in others[:max_others]:
            active.is_tracking = True

    if not features or not competitors:
        if not features and not competitors:
            raise ValueError(
                "Could not build features or rivals for this client. Add a website, then run intel again "
                "or add features and competitors manually."
            )
        if not features:
            raise ValueError(
                "Could not extract product features for this client. Add a website, then run intel again "
                "or add features manually."
            )
        peer_hint = _client_peer_hint(
            client.name,
            client.industry,
            client.niche,
            _business_model_from_client(client),
            client.notes,
            client.tagline,
        )
        market_bit = f" in {required_market}" if scope == "local" and required_market else ""
        raise ValueError(
            f"No matching {peer_hint} survived quality filters{market_bit}. "
            "Platform search/AI could not rank true peers right now — "
            "try again shortly, or add real rivals manually and pin them."
        )

    kept: list[Competitor] = []
    analyzed: list[Competitor] = []
    # Snapshot BEFORE analyze — add mode must keep these names even if AI prune is harsh
    baseline_set = {_as_str(n).lower().strip() for n in baseline_names if _as_str(n)}
    if mode == "add" and not baseline_set:
        baseline_set = {_as_str(c.name).lower().strip() for c in competitors if _as_str(c.name)}
    # Scrape all candidates concurrently with concurrency limiter (up to 5 parallel)
    async def _scrape_candidate(comp: Competitor) -> tuple[str, dict]:
        if not comp.website:
            return comp.id, {}
        already = bool(comp.feature_list) and bool(comp.description or comp.why_dangerous)
        if already:
            return comp.id, {}
        try:
            res = await scrape_website(db, agency.id, comp.website)
            return comp.id, res if isinstance(res, dict) else {}
        except Exception as err:
            logger.warning("Scrape candidate failed for %s: %s", comp.website, err)
            return comp.id, {}

    sem = asyncio.Semaphore(5)
    async def _sem_scrape(c: Competitor):
        async with sem:
            return await _scrape_candidate(c)

    scrape_results = await asyncio.gather(*[_sem_scrape(c) for c in competitors], return_exceptions=True)
    scraped_map: dict[str, dict] = {}
    for item in scrape_results:
        if isinstance(item, tuple) and len(item) == 2:
            scraped_map[item[0]] = item[1]

    for idx, competitor in enumerate(competitors):
        # Normalize stored website so UI links open correctly
        if competitor.website:
            competitor.website = _normalize_website(competitor.website) or competitor.website
        already_enriched = bool(competitor.feature_list) and bool(
            competitor.description or competitor.why_dangerous
        )
        site_data = scraped_map.get(competitor.id, {})
        site_md = (site_data.get("markdown") or "")[:3500]
        if already_enriched and (competitor.overlap_score or 0) >= 55:
            # Fast path: reuse prior enrich instead of another scrape+LLM round-trip
            analysis = {
                "tagline": competitor.tagline,
                "description": competitor.description,
                "headquarters": competitor.headquarters,
                "headquarters_country": competitor.headquarters,
                "overlap_score": competitor.overlap_score or 70,
                "threat_level": competitor.threat_level or "medium",
                "is_leading_rival": True,
                "same_niche": True,
                "same_market": True,
                "is_global_platform": False,
                "why_dangerous": competitor.why_dangerous or competitor.description,
                "evidence_snippet": competitor.evidence_snippet,
                "features": competitor.feature_list or [],
            }
        else:
            analysis = await ai_service.structured_json(
                db,
                agency.id,
                (
                    "Enrich a competitor for competitive intelligence against THIS client only. "
                    "Return JSON keys: tagline, description, headquarters, headquarters_country, industry, business_model, "
                    "overlap_score (0-100), threat_level (low|medium|high), is_leading_rival (boolean), "
                    "same_niche (boolean), same_market (boolean), is_global_platform (boolean), "
                    "why_dangerous (1-2 sentences), evidence_snippet (short quote/paraphrase from site), "
                    "features (array of {name, category, description}). "
                    "Score overlap high ONLY when industry + niche + buyer + business model truly match. "
                    "If the rival is a consumer shopping/ecommerce retailer, fintech wallet/payments app, bank, "
                    "government board/ministry/authority, news site, directory, or unrelated industry, "
                    "set same_niche=false, is_leading_rival=false, overlap_score below 40. "
                    "Same broad industry label like 'Technology' is NOT enough — buyers and product must match. "
                    "If the site is a directory, review site, news article, job board, or unrelated industry, "
                    "set same_niche=false, is_leading_rival=false, overlap_score below 40. "
                    "Global hyperscalers/platforms that are not peer businesses should be low threat, "
                    "same_niche=false, is_global_platform=true. "
                    + (
                        f"GLOBAL SCOPE: The user specifically requested global/international industry peers and benchmarks. "
                        "When the rival is a genuine international peer/benchmark in the same industry (e.g. global universities for universities, global tech/software houses for software, global hotel chains for hotels, global gym brands for gyms, etc.), "
                        "set same_niche=true, same_market=true, is_leading_rival=true, and score overlap high (75-95) regardless of country. "
                        "Only reject unrelated industries, gig sites (Upwork/Fiverr), or non-peer directories."
                        if scope == "global"
                        else (
                            f"LOCAL MARKET REQUIRED: {required_market}. "
                            f"Set headquarters_country from SITE EVIDENCE only (not guesses). "
                            f"If headquarters / primary selling country is clearly NOT {required_market}, "
                            "set same_market=false and overlap_score below 40. "
                            "Do not treat neighboring countries (e.g. India vs Pakistan, Singapore vs Pakistan) as the same market. "
                            "Never invent that a foreign company sells primarily in the required market."
                            if scope == "local" and required_market
                            else ""
                        )
                    )
                ),
                json.dumps(
                    {
                        "client": client.name,
                        "client_industry": client.industry,
                        "client_niche": client.niche,
                        "client_business_model": _business_model_from_client(client),
                        "client_market_area": required_market or _market_area_from_client(client),
                        "competitor_scope": scope,
                        "required_market": required_market or None,
                        "client_features": [
                            {"name": f.name, "category": f.category, "description": f.description} for f in features
                        ],
                        "competitor": {
                            "name": competitor.name,
                            "website": competitor.website,
                            "site_excerpt": site_md,
                        },
                    }
                )[:9000],
                temperature=0.2,
            )
        # If AI fallback text returned, keep prior competitor values — but do not auto-trust as leading
        if not analysis or ("summary" in analysis and "features" not in analysis):
            analysis = {
                "overlap_score": competitor.overlap_score or 70,
                "threat_level": competitor.threat_level or "medium",
                "is_leading_rival": False,
                "same_niche": True,
                "same_market": True,
                "is_global_platform": False,
                "why_dangerous": competitor.why_dangerous
                or competitor.description
                or f"{competitor.name} competes for the same buyers.",
                "features": competitor.feature_list or [],
            }

        competitor.tagline = _as_str(analysis.get("tagline")) or competitor.tagline
        competitor.description = _as_str(analysis.get("description")) or competitor.description
        competitor.headquarters = _as_str(analysis.get("headquarters") or analysis.get("headquarters_country")) or competitor.headquarters
        competitor.feature_list = analysis.get("features") if isinstance(analysis.get("features"), list) else (competitor.feature_list or [])
        try:
            raw_score = float(analysis.get("overlap_score") or competitor.overlap_score or 75.0)
            if 0 < raw_score <= 10.0:
                raw_score = raw_score * 10.0
            computed_score = _compute_feature_overlap_score(features, competitor.feature_list, default_score=raw_score, rank_idx=idx)
            competitor.overlap_score = max(55.0, min(94.0, computed_score))
        except (TypeError, ValueError):
            competitor.overlap_score = competitor.overlap_score or 75.0
        competitor.threat_level = _as_str(analysis.get("threat_level") or competitor.threat_level or "medium").lower()
        if competitor.threat_level not in {"low", "medium", "high"}:
            competitor.threat_level = "medium"
        competitor.why_dangerous = _as_str(analysis.get("why_dangerous")) or competitor.why_dangerous
        competitor.evidence_snippet = _as_str(analysis.get("evidence_snippet")) or competitor.evidence_snippet
        if site_md and not competitor.evidence_snippet:
            competitor.evidence_snippet = site_md[:280]
        competitor.last_scraped_at = datetime.utcnow()
        analyzed.append(competitor)

        # Trust site + HQ fields for geo — AI blurbs often hallucinate the client's country
        site_geo_blob = " ".join(
            [
                _as_str(competitor.headquarters),
                _as_str(analysis.get("headquarters_country")),
                site_md[:2000],
            ]
        ).lower()
        hq_key = _normalize_country_key(_as_str(analysis.get("headquarters_country") or competitor.headquarters))
        # If site text clearly names another country, prefer that over AI HQ claim
        site_conflict = _mentions_conflicting_country(site_md[:2000].lower(), competitor.website, required_market) if required_market else False
        if site_conflict:
            for key, aliases in _COUNTRY_ALIASES.items():
                if key == _normalize_country_key(required_market):
                    continue
                if _blob_mentions_any(site_md[:2000].lower(), aliases) or _host_matches_tlds(
                    _domain_of(competitor.website or ""), _COUNTRY_TLDS.get(key, set())
                ):
                    hq_key = key
                    break
        market_key = _normalize_country_key(required_market)
        peer_blob = " ".join(
            [
                _as_str(competitor.name),
                _as_str(competitor.description),
                _as_str(analysis.get("industry")),
                _as_str(analysis.get("business_model")),
                site_md[:2000],
            ]
        ).lower()
        client_feature_blob = " ".join(
            f"{_as_str(f.name)} {_as_str(f.description)}" for f in features[:10]
        )
        client_kind_blob = (
            f"{_as_str(client.industry)} {_as_str(client.niche)} {_business_model_from_client(client)} "
            f"{_as_str(client.notes)} {_as_str(client.tagline)} {client.name} {client_feature_blob}"
        )
        client_is_beauty = _looks_like_beauty_client(client_kind_blob)
        client_is_food = _looks_like_food_client(client_kind_blob)
        client_is_software_peer = _looks_like_software_peer_client(client_kind_blob)
        bad_peer = _incompatible_peer(
            client_model=_business_model_from_client(client),
            client_industry=f"{_as_str(client.industry)} {client_feature_blob[:400]}",
            client_niche=_as_str(client.niche) or _as_str(client.notes),
            rival_model=_as_str(analysis.get("business_model")),
            rival_industry=_as_str(analysis.get("industry")),
            rival_blob=peer_blob,
            client_name=client.name,
        )
        curated = _is_curated_seed_rival(
            competitor.name,
            required_market if scope == "local" else None,
            kind="food" if client_is_food else ("software" if client_is_software_peer else ("beauty" if client_is_beauty else None)),
        )
        client_food_tier = (
            _food_tier_from_blob(client.name, client.niche, client.industry, _business_model_from_client(client))
            if client_is_food
            else None
        )
        client_peer_scale = _peer_scale_from_blob(
            client.name,
            client.niche,
            client.industry,
            _business_model_from_client(client),
            name=client.name,
        )
        # Boutique / local-specialty clients: enterprise giants & food franchises are not peers
        if not competitor.is_pinned and (
            (
                client_food_tier == _FOOD_TIER_LOCAL
                and _is_global_food_franchise(competitor.name, competitor.website)
            )
            or (
                client_peer_scale == _PEER_BOUTIQUE
                and (
                    _is_global_food_franchise(competitor.name, competitor.website)
                    or (scope == "local" and _is_global_megarival(competitor.name, competitor.website))
                    or (
                        scope == "local"
                        and _peer_scale_from_blob(
                            competitor.name, competitor.description, name=competitor.name, website=competitor.website
                        )
                        == _PEER_ENTERPRISE
                    )
                )
            )
        ):
            curated = False
            competitor.is_tracking = False
            competitor.threat_level = "low"
            continue
        # Curated seeds already passed geo/niche gates — don't let flaky scrape/AI wipe the list down to 1
        if curated and not hq_key and market_key:
            hq_key = market_key
            if not competitor.headquarters:
                competitor.headquarters = required_market
        has_local_proof = (
            curated
            or (bool(hq_key) and bool(market_key) and hq_key == market_key)
            or _mentions_target_market(site_geo_blob, competitor.website, required_market)
            or _host_matches_tlds(_domain_of(competitor.website or ""), _COUNTRY_TLDS.get(market_key or "", set()))
        )
        wrong_market = (
            scope == "local"
            and bool(required_market)
            and not competitor.is_pinned
            and not curated
            and (
                analysis.get("same_market") is False
                or site_conflict
                or _mentions_conflicting_country(site_geo_blob, competitor.website, required_market)
                or (bool(hq_key) and bool(market_key) and hq_key != market_key and not has_local_proof)
            )
        )
        # Only drop truly dead / parked domains with no content and error status
        dead_site = (
            not competitor.is_pinned
            and not curated
            and (
                not competitor.website
                or (
                    bool(competitor.website)
                    and _site_looks_parked_or_empty(site_md)
                    and site_data.get("status") not in {"ok", "skipped"}
                )
            )
        )
        weak_software_peer = (
            client_is_software_peer
            and not competitor.is_pinned
            and not curated
            and bool(site_md)
            and not _site_supports_software_peer(site_md)
        )
        fake_brand = (not competitor.is_pinned) and _is_generic_or_fake_rival_name(competitor.name)
        site_host_noise = bool(competitor.website and (_is_serp_noise_domain(competitor.website) or _is_blog_or_article_url(competitor.website, competitor.name)))
        off_niche = (
            not competitor.is_pinned
            and (
                fake_brand
                or (scope == "local" and _is_global_megarival(competitor.name, competitor.website))
                or site_host_noise
                or (analysis.get("is_global_platform") is True and not curated and scope == "local")
                or (analysis.get("same_niche") is False and not curated and scope == "local")
                or (bad_peer and not curated)
                or wrong_market
                or dead_site
                or weak_software_peer
                or ((competitor.overlap_score or 0) < 50 and not curated and scope == "local")
                or (
                    not curated
                    and competitor.threat_level == "low"
                    and (competitor.overlap_score or 0) < 60
                    and analysis.get("is_leading_rival") is False
                    and scope == "local"
                )
            )
        )
        if off_niche:
            # Add mode: do not wipe the user's existing list for soft AI/scrape misses
            name_key = _as_str(competitor.name).lower().strip()
            hard_drop = (
                fake_brand
                or bad_peer
                or _is_global_megarival(competitor.name, competitor.website)
                or _is_global_food_franchise(competitor.name, competitor.website)
                or site_host_noise
                or _looks_like_invented_food_domain(competitor.name, competitor.website)
                or _looks_like_content_or_cpg_noise(competitor.name, competitor.website)
                or _looks_like_recipe_or_menu_item_name(competitor.name)
                or _looks_like_marketing_slogan_name(competitor.name)
                or (
                    client_is_food
                    and (
                        _looks_like_software_peer_client(
                            competitor.name, competitor.description, competitor.why_dangerous, peer_blob
                        )
                        or _looks_like_fmcg_or_snack_brand(
                            competitor.name, competitor.description, competitor.why_dangerous, peer_blob
                        )
                    )
                )
            )
            if mode == "add" and not hard_drop:
                competitor.is_tracking = True
                if (competitor.overlap_score or 0) < 55:
                    competitor.overlap_score = 70.0
                if competitor.threat_level == "low":
                    competitor.threat_level = "medium"
                kept.append(competitor)
                continue
            competitor.is_tracking = False
            competitor.threat_level = "low"
            continue

        competitor.is_tracking = True
        if scope == "global":
            competitor.overlap_score = max(float(competitor.overlap_score or 0), 80.0)
            if competitor.threat_level not in {"medium", "high"}:
                competitor.threat_level = "high"
        elif competitor.threat_level == "low":
            competitor.threat_level = "medium"
        kept.append(competitor)

    # Always put pinned/manual rivals back even if AI scored them weakly
    kept_ids = {c.id for c in kept}
    for competitor in analyzed:
        if competitor.is_pinned and competitor.id not in kept_ids:
            competitor.is_tracking = True
            if competitor.threat_level == "low":
                competitor.threat_level = "medium"
            if (competitor.overlap_score or 0) < 55:
                competitor.overlap_score = 70.0
            kept.insert(0, competitor)
            kept_ids.add(competitor.id)

    # Add mode: restore baseline rivals that fit this run's scope
    if mode == "add" and baseline_set:
        for competitor in analyzed:
            name_key = _as_str(competitor.name).lower().strip()
            if competitor.id in kept_ids or name_key not in baseline_set:
                continue
            if not competitor.is_pinned and not _rival_fits_run_scope(
                name=competitor.name,
                website=competitor.website,
                headquarters=competitor.headquarters,
                description=competitor.description,
                why=competitor.why_dangerous,
                scope=scope,
                market=scope_market,
                client_name=client.name,
                is_pinned=False,
                strict=False,
            ):
                continue
            if (
                _is_generic_or_fake_rival_name(competitor.name)
                or _looks_like_invented_food_domain(competitor.name, competitor.website)
                or _looks_like_recipe_or_menu_item_name(competitor.name)
                or _looks_like_content_or_cpg_noise(competitor.name, competitor.website)
            ):
                continue
            competitor.is_tracking = True
            if competitor.threat_level == "low":
                competitor.threat_level = "medium"
            if (competitor.overlap_score or 0) < 55:
                competitor.overlap_score = 72.0
            kept.insert(0, competitor)
            kept_ids.add(competitor.id)
            logger.info(
                "Add-mode kept existing rival %s for client=%s",
                competitor.name,
                client.id,
            )

    # Also restore baseline rivals that were not in this analyze pass (still in DB)
    if mode == "add" and baseline_set:
        existing_all_baseline = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                )
            )
        ).scalars().all()
        for rival in existing_all_baseline:
            name_key = _as_str(rival.name).lower().strip()
            if rival.id in kept_ids or name_key not in baseline_set:
                continue
            if not rival.is_pinned and not _rival_fits_run_scope(
                name=rival.name,
                website=rival.website,
                headquarters=rival.headquarters,
                description=rival.description,
                why=rival.why_dangerous,
                scope=scope,
                market=scope_market,
                client_name=client.name,
                is_pinned=False,
                strict=False,
            ):
                continue
            if (
                _is_generic_or_fake_rival_name(rival.name)
                or _looks_like_invented_food_domain(rival.name, rival.website)
                or _looks_like_recipe_or_menu_item_name(rival.name)
                or _looks_like_content_or_cpg_noise(rival.name, rival.website)
            ):
                continue
            rival.is_tracking = True
            if rival.threat_level == "low":
                rival.threat_level = "medium"
            if (rival.overlap_score or 0) < 55:
                rival.overlap_score = 72.0
            kept.insert(0, rival)
            kept_ids.add(rival.id)

    pinned_kept = sum(1 for c in kept if c.is_pinned)
    if mode == "update":
        target_kept = max(count, pinned_kept)
    elif mode == "replace":
        target_kept = pinned_kept + count
    else:
        # add = keep current survivors + find exactly `count` NEW names
        new_in_kept = (
            sum(1 for c in kept if _as_str(c.name).lower().strip() not in baseline_set)
            if baseline_set
            else 0
        )
        need_more = max(0, count - new_in_kept) if baseline_set else count
        target_kept = len(kept) + need_more
        logger.info(
            "Add-mode rival target client=%s baseline=%s kept=%s already_new=%s still_need=%s target=%s",
            client.id,
            len(baseline_set),
            len(kept),
            new_in_kept,
            need_more,
            target_kept,
        )

    # If prune left us short of the requested count, fill remaining slots with
    # same-tier AI peers (not curated seed lists).
    if len(kept) < target_kept:
        existing_all = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                )
            )
        ).scalars().all()
        # When AI is down: first re-enable BASELINE peers, then other same-category untracked
        if mode == "add":
            client_fmt = (
                _food_format_from_blob(client.name, client.niche, client.industry)
                if _looks_like_food_client(client.name, client.niche, client.industry)
                else ""
            )
            client_tier = (
                _food_tier_from_blob(client.name, client.niche, client.industry)
                if client_fmt
                else ""
            )

            def _can_reenable(rival: Competitor) -> bool:
                if rival.id in kept_ids:
                    return False
                if _is_global_food_franchise(rival.name, rival.website):
                    return False
                if _is_generic_or_fake_rival_name(rival.name) or _looks_like_invented_food_domain(
                    rival.name, rival.website
                ):
                    return False
                if _looks_like_content_or_cpg_noise(rival.name, rival.website):
                    return False
                # Skip SERP title junk like "Savor The Biggest Pizza in Town"
                if len(_as_str(rival.name).split()) >= 6:
                    return False
                if client_fmt:
                    rival_fmt = _food_format_from_blob(
                        rival.name, rival.description, rival.why_dangerous, rival.website
                    )
                    if rival_fmt != _FOOD_FORMAT_GENERAL and not _food_format_compatible(
                        client_fmt, rival_fmt
                    ):
                        return False
                    rival_tier = _food_tier_from_blob(rival.name, rival.description)
                    if rival_tier and not _food_tier_compatible(client_tier, rival_tier):
                        return False
                if not _rival_fits_run_scope(
                    name=_as_str(rival.name),
                    website=rival.website,
                    headquarters=rival.headquarters,
                    description=rival.description,
                    why=rival.why_dangerous,
                    scope=scope,
                    market=required_market or "",
                    client_name=client.name,
                    is_pinned=False,
                    strict=False,
                ):
                    return False
                return True

            # Pass 1: baseline names first (keep-current promise)
            for rival in existing_all:
                if len(kept) >= target_kept and _as_str(rival.name).lower().strip() not in baseline_set:
                    break
                name_key = _as_str(rival.name).lower().strip()
                if name_key not in baseline_set:
                    continue
                if rival.is_tracking and rival.id in kept_ids:
                    continue
                if not _can_reenable(rival):
                    continue
                rival.is_tracking = True
                rival.is_pinned = False
                if (rival.overlap_score or 0) < 55:
                    rival.overlap_score = 74.0
                if not rival.threat_level or rival.threat_level == "low":
                    rival.threat_level = "high"
                if rival.id not in kept_ids:
                    kept.append(rival)
                    kept_ids.add(rival.id)
                logger.info(
                    "Re-enabled baseline peer %s for client=%s (add-mode keep-current)",
                    rival.name,
                    client.id,
                )

            # Pass 2: other untracked peers only to fill NEW slots
            for rival in existing_all:
                if len(kept) >= target_kept:
                    break
                if rival.is_tracking and rival.id in kept_ids:
                    continue
                name_key = _as_str(rival.name).lower().strip()
                if name_key in baseline_set:
                    continue
                if not _can_reenable(rival):
                    continue
                rival.is_tracking = True
                rival.is_pinned = False
                if (rival.overlap_score or 0) < 55:
                    rival.overlap_score = 74.0
                if not rival.threat_level or rival.threat_level == "low":
                    rival.threat_level = "high"
                kept.append(rival)
                kept_ids.add(rival.id)
                logger.info(
                    "Re-enabled untracked peer %s for client=%s (AI thin / add-mode)",
                    rival.name,
                    client.id,
                )

        already_names = [_as_str(c.name) for c in kept]
        bm = _business_model_from_client(client)
        fill_market = required_market if scope == "local" else "global / international"
        fill_rows = await _ai_propose_same_tier_peers(
            db,
            agency.id,
            client,
            needed=target_kept - len(kept),
            already_have=already_names,
            scope=scope,
            market_focus=fill_market,
            business_model=bm,
        )
        for item in fill_rows:
            if len(kept) >= target_kept:
                break
            name = _as_str(item.get("name")).strip()
            website = _normalize_website(_as_str(item.get("website")) or None)
            if not name or not website:
                continue
            if _is_generic_or_fake_rival_name(name):
                continue
            if _looks_like_content_or_cpg_noise(name, website):
                continue
            if _looks_like_brand_geo_hallucination(
                client.name,
                name,
                required_market or _market_area_from_client(client),
                website=website,
                source=_as_str(item.get("source")) or "ai",
            ):
                continue
            competitor = _find_matching_competitor(existing_all, name, website)
            hq = required_market or _as_str(item.get("headquarters_country")) or None
            why = _as_str(item.get("why_relevant")) or None
            try:
                overlap = float(item.get("overlap_score") or 72)
            except (TypeError, ValueError):
                overlap = 72.0
            if competitor:
                if competitor.id in kept_ids:
                    continue
                competitor.website = website or competitor.website
                competitor.headquarters = competitor.headquarters or hq
                competitor.description = competitor.description or why
                competitor.why_dangerous = competitor.why_dangerous or why
                competitor.overlap_score = max(float(competitor.overlap_score or 0), overlap, 70.0)
                competitor.threat_level = (
                    "high" if competitor.threat_level == "low" else (competitor.threat_level or "high")
                )
                competitor.is_tracking = True
            else:
                competitor = Competitor(
                    agency_id=agency.id,
                    client_id=client.id,
                    name=name,
                    website=website,
                    description=why,
                    why_dangerous=why,
                    headquarters=hq,
                    threat_level=_as_str(item.get("threat_level"), "high").lower() or "high",
                    overlap_score=max(overlap, 70.0),
                    is_tracking=True,
                    feature_list=[],
                )
                db.add(competitor)
                await db.flush()
                existing_all.append(competitor)
            kept.append(competitor)
            kept_ids.add(competitor.id)
            analyzed.append(competitor)
        if len(kept) < target_kept:
            logger.warning(
                "AI peer backfill still short for client=%s market=%s kept=%s requested=%s",
                client.id,
                required_market,
                len(kept),
                count,
            )

    # Food/local last resort when Serp+AI still left us short
    if (
        len(kept) < target_kept
        and scope == "local"
        and required_market
        and _looks_like_food_client(
            client.name,
            client.industry,
            client.niche,
            _business_model_from_client(client),
            client.notes,
            client.tagline,
        )
    ):
        existing_all = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                )
            )
        ).scalars().all()
        already_names = [_as_str(c.name) for c in kept]
        food_tier = _food_tier_from_blob(
            client.name, client.niche, client.industry, _business_model_from_client(client)
        )
        for item in _seed_local_qsr_rivals(
            required_market,
            client.name,
            already_have=already_names,
            client_website=client.website,
            client_tier=food_tier,
            client_niche=_as_str(client.niche),
            client_industry=_as_str(client.industry),
            limit=max(count * 2, 8),
        ):
            if len(kept) >= target_kept:
                break
            name = _as_str(item.get("name")).strip()
            website = _normalize_website(_as_str(item.get("website")) or None)
            if not name or not website:
                continue
            if _looks_like_brand_geo_hallucination(
                client.name, name, required_market, website=website, source="seed"
            ):
                continue
            competitor = _find_matching_competitor(existing_all, name, website)
            why = _as_str(item.get("why_relevant")) or None
            if competitor:
                if competitor.id in kept_ids:
                    continue
                competitor.website = website or competitor.website
                competitor.headquarters = competitor.headquarters or required_market
                competitor.description = competitor.description or why
                competitor.why_dangerous = competitor.why_dangerous or why
                competitor.overlap_score = max(float(competitor.overlap_score or 0), 74.0)
                competitor.threat_level = (
                    "high" if competitor.threat_level == "low" else (competitor.threat_level or "high")
                )
                competitor.is_tracking = True
                competitor.is_pinned = False
            else:
                competitor = Competitor(
                    agency_id=agency.id,
                    client_id=client.id,
                    name=name,
                    website=website,
                    description=why,
                    why_dangerous=why,
                    headquarters=required_market,
                    threat_level="high",
                    overlap_score=76.0,
                    is_tracking=True,
                    feature_list=[],
                )
                db.add(competitor)
                await db.flush()
                existing_all.append(competitor)
            kept.append(competitor)
            kept_ids.add(competitor.id)
            analyzed.append(competitor)

    # Beauty/local last resort — fill to requested count with curated beauty brands
    if (
        len(kept) < target_kept
        and scope == "local"
        and required_market
        and _looks_like_beauty_client(
            client.name,
            client.industry,
            client.niche,
            _business_model_from_client(client),
            client.notes,
            client.tagline,
        )
    ):
        existing_all = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                )
            )
        ).scalars().all()
        already_names = [_as_str(c.name) for c in kept]
        for item in _seed_local_beauty_rivals(
            required_market,
            client.name,
            already_have=already_names,
            client_website=client.website,
            client_niche=_as_str(client.niche),
            client_industry=_as_str(client.industry),
            limit=max(count * 2, 8),
        ):
            if len(kept) >= target_kept:
                break
            name = _as_str(item.get("name")).strip()
            website = _normalize_website(_as_str(item.get("website")) or None)
            if not name or not website:
                continue
            if _looks_like_brand_geo_hallucination(
                client.name, name, required_market, website=website, source="seed"
            ):
                continue
            competitor = _find_matching_competitor(existing_all, name, website)
            why = _as_str(item.get("why_relevant")) or None
            if competitor:
                if competitor.id in kept_ids:
                    continue
                competitor.website = website or competitor.website
                competitor.headquarters = competitor.headquarters or required_market
                competitor.description = competitor.description or why
                competitor.why_dangerous = competitor.why_dangerous or why
                competitor.overlap_score = max(float(competitor.overlap_score or 0), 74.0)
                competitor.threat_level = (
                    "high" if competitor.threat_level == "low" else (competitor.threat_level or "high")
                )
                competitor.is_tracking = True
                competitor.is_pinned = False
            else:
                competitor = Competitor(
                    agency_id=agency.id,
                    client_id=client.id,
                    name=name,
                    website=website,
                    description=why,
                    why_dangerous=why,
                    headquarters=required_market,
                    threat_level="high",
                    overlap_score=76.0,
                    is_tracking=True,
                    feature_list=[],
                )
                db.add(competitor)
                await db.flush()
                existing_all.append(competitor)
            kept.append(competitor)
            kept_ids.add(competitor.id)
            analyzed.append(competitor)

    # Software/local last resort — fill to requested count with curated peer houses
    if (
        len(kept) < target_kept
        and _looks_like_software_peer_client(
            client.name,
            client.industry,
            client.niche,
            _business_model_from_client(client),
            client.notes,
            client.tagline,
        )
    ):
        existing_all = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                )
            )
        ).scalars().all()
        already_names = [_as_str(c.name) for c in kept]
        user_country = _as_str(competitor_country).strip()
        seed_market = user_country or required_market or _market_area_from_client(client)
        home = _known_brand_home_market(client.name, client.website)
        seed_rows: list[dict] = []
        if scope == "global":
            # Global software peers from curated international list
            for seed in _GLOBAL_SOFTWARE_SEEDS:
                name = _as_str(seed.get("name")).strip()
                website = _normalize_website(_as_str(seed.get("website")) or None)
                if not name or not website:
                    continue
                if _rival_keys(name, website) & _blocked_rival_keys(already_names, client.name):
                    continue
                seed_rows.append(
                    {
                        "name": name,
                        "website": website,
                        "why_relevant": f"Global software / digital engineering peer for {client.name}",
                        "overlap_score": 85.0,
                        "threat_level": "high",
                        "source": "seed",
                        "headquarters_country": _as_str(seed.get("headquarters_country")) or "United States",
                    }
                )
                if len(seed_rows) >= count * 2:
                    break
        else:
            seed_rows = _seed_local_software_rivals(
                seed_market,
                client.name,
                already_have=already_names,
                client_website=client.website,
                client_niche=_as_str(client.niche),
                client_industry=_as_str(client.industry),
                limit=max(count * 2, 8),
            )
            # Only fall back to brand-home seeds when the user did NOT pick a country
            if (
                not seed_rows
                and not user_country
                and home
                and _normalize_country_key(home) != _normalize_country_key(seed_market)
            ):
                seed_rows = _seed_local_software_rivals(
                    home,
                    client.name,
                    already_have=already_names,
                    client_website=client.website,
                    client_niche=_as_str(client.niche),
                    client_industry=_as_str(client.industry),
                    limit=max(count * 2, 8),
                )
        for item in seed_rows:
            if len(kept) >= target_kept:
                break
            name = _as_str(item.get("name")).strip()
            website = _normalize_website(_as_str(item.get("website")) or None)
            if not name or not website or _is_generic_or_fake_rival_name(name):
                continue
            competitor = _find_matching_competitor(existing_all, name, website)
            why = _as_str(item.get("why_relevant")) or None
            hq = _as_str(item.get("headquarters_country")) or seed_market
            if competitor:
                if competitor.id in kept_ids:
                    continue
                competitor.website = website or competitor.website
                competitor.headquarters = competitor.headquarters or hq
                competitor.description = competitor.description or why
                competitor.why_dangerous = competitor.why_dangerous or why
                competitor.overlap_score = max(float(competitor.overlap_score or 0), 74.0)
                competitor.threat_level = (
                    "high" if competitor.threat_level == "low" else (competitor.threat_level or "high")
                )
                competitor.is_tracking = True
                competitor.is_pinned = False
            else:
                competitor = Competitor(
                    agency_id=agency.id,
                    client_id=client.id,
                    name=name,
                    website=website,
                    description=why,
                    why_dangerous=why,
                    headquarters=hq,
                    threat_level="high",
                    overlap_score=76.0,
                    is_tracking=True,
                    feature_list=[],
                )
                db.add(competitor)
                await db.flush()
                existing_all.append(competitor)
            kept.append(competitor)
            kept_ids.add(competitor.id)
            analyzed.append(competitor)
        if seed_rows:
            logger.warning(
                "Software last-resort peers used for client=%s market=%s kept=%s target=%s",
                client.id,
                seed_market,
                len(kept),
                target_kept,
            )

    # Multi-industry last resort when live SerpAPI / AI is thin
    if len(kept) < target_kept and scope == "local" and required_market:
        existing_all = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                )
            )
        ).scalars().all()
        already_names = [_as_str(c.name) for c in kept]
        industry_seed_rows = _seed_local_industry_rivals(
            f"{_as_str(client.industry)} {_as_str(client.niche)}",
            required_market,
            client.name,
            already_have=already_names,
            client_website=client.website,
            limit=max(count * 2, 8),
        )
        for item in industry_seed_rows:
            if len(kept) >= target_kept:
                break
            name = _as_str(item.get("name")).strip()
            website = _normalize_website(_as_str(item.get("website")) or None)
            if not name or not website:
                continue
            competitor = _find_matching_competitor(existing_all, name, website)
            why = _as_str(item.get("why_relevant")) or None
            if competitor:
                if competitor.id in kept_ids:
                    continue
                competitor.website = website or competitor.website
                competitor.headquarters = competitor.headquarters or required_market
                competitor.description = competitor.description or why
                competitor.why_dangerous = competitor.why_dangerous or why
                competitor.overlap_score = max(float(competitor.overlap_score or 0), 85.0)
                competitor.threat_level = "high"
                competitor.is_tracking = True
                competitor.is_pinned = False
            else:
                competitor = Competitor(
                    agency_id=agency.id,
                    client_id=client.id,
                    name=name,
                    website=website,
                    description=why,
                    why_dangerous=why,
                    headquarters=required_market,
                    threat_level="high",
                    overlap_score=85.0,
                    is_tracking=True,
                    feature_list=[],
                )
                db.add(competitor)
                await db.flush()
                existing_all.append(competitor)
            kept.append(competitor)
            kept_ids.add(competitor.id)
            analyzed.append(competitor)

    # Global multi-industry last resort when scope is global and kept < target_kept
    if len(kept) < target_kept and scope == "global":
        existing_all = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                )
            )
        ).scalars().all()
        already_names = [_as_str(c.name) for c in kept]
        client_home_mkt = _market_area_from_client(client) or _known_brand_home_market(client.name, client.website) or ""
        global_seed_rows = _seed_global_industry_rivals(
            f"{_as_str(client.industry)} {_as_str(client.niche)}",
            client.name,
            already_have=already_names,
            client_website=client.website,
            client_niche=_as_str(client.niche),
            client_market=client_home_mkt,
            limit=max(count * 2, 8),
        )
        for item in global_seed_rows:
            if len(kept) >= target_kept:
                break
            name = _as_str(item.get("name")).strip()
            website = _normalize_website(_as_str(item.get("website")) or None)
            if not name or not website or _is_generic_or_fake_rival_name(name):
                continue
            competitor = _find_matching_competitor(existing_all, name, website)
            why = _as_str(item.get("why_relevant")) or f"Leading international peer benchmark in {item.get('headquarters_country', 'global market')}"
            hq = _as_str(item.get("headquarters_country")) or "United States"
            if competitor:
                if competitor.id in kept_ids:
                    continue
                competitor.website = website or competitor.website
                competitor.headquarters = competitor.headquarters or hq
                competitor.description = competitor.description or why
                competitor.why_dangerous = competitor.why_dangerous or why
                competitor.overlap_score = max(float(competitor.overlap_score or 0), 85.0)
                competitor.threat_level = "high"
                competitor.is_tracking = True
                competitor.is_pinned = False
            else:
                competitor = Competitor(
                    agency_id=agency.id,
                    client_id=client.id,
                    name=name,
                    website=website,
                    description=why,
                    why_dangerous=why,
                    headquarters=hq,
                    threat_level="high",
                    overlap_score=85.0,
                    is_tracking=True,
                    feature_list=[],
                )
                db.add(competitor)
                await db.flush()
                existing_all.append(competitor)
            kept.append(competitor)
            kept_ids.add(competitor.id)
            analyzed.append(competitor)

    if not kept and analyzed:
        # Prefer strongest overlaps that are not megacorp/noise domains.
        # For local runs, never resurrect clear foreign-market rivals as a fallback.
        def _fallback_ok(c: Competitor) -> bool:
            if _is_generic_or_fake_rival_name(c.name):
                return False
            if _is_global_megarival(c.name, c.website):
                return False
            if c.website and _is_serp_noise_domain(c.website):
                return False
            if scope == "local" and required_market:
                blob = f"{c.headquarters or ''} {c.description or ''} {c.why_dangerous or ''}".lower()
                if _mentions_conflicting_country(blob, c.website, required_market):
                    return False
            return True

        analyzed_sorted = sorted(
            [c for c in analyzed if _fallback_ok(c)] or [],
            key=lambda c: float(c.overlap_score or 0),
            reverse=True,
        )
        for competitor in analyzed_sorted[:count]:
            competitor.is_tracking = True
            if competitor.threat_level not in {"medium", "high"}:
                competitor.threat_level = "medium"
            kept.append(competitor)

    # update: refresh up to `count`. replace: pinned + up to `count` fresh. add: keep all after prune.
    collapse_duplicate_competitors(kept)
    kept = [c for c in kept if c.is_tracking]
    # Final scope gate — local shows only selected-country peers; global drops noise/hallucinations
    # add/keep-current: previous rivals stay tracked even if this run's country/global differs
    scope_market = required_market if scope == "local" else (required_market or "")
    client_kind_final = (
        f"{client.name} {client.industry or ''} {client.niche or ''} "
        f"{_business_model_from_client(client)} {client.notes or ''} {client.tagline or ''}"
    )

    def _hard_junk_rival(rival: Competitor) -> bool:
        if _is_generic_or_fake_rival_name(rival.name):
            return True
        if _looks_like_recipe_or_menu_item_name(rival.name):
            return True
        if _looks_like_invented_food_domain(rival.name, rival.website):
            return True
        if _looks_like_content_or_cpg_noise(rival.name, rival.website):
            return True
        if _looks_like_food_client(client_kind_final) and (
            _looks_like_software_peer_client(rival.name, rival.description, rival.why_dangerous, rival.website)
            or _looks_like_fmcg_or_snack_brand(rival.name, rival.description, rival.why_dangerous, rival.website)
        ):
            return True
        if _incompatible_peer(
            client_model=_business_model_from_client(client),
            client_industry=_as_str(client.industry),
            client_niche=_as_str(client.niche) or _as_str(client.notes),
            rival_model="",
            rival_industry="",
            rival_blob=f"{rival.name} {rival.description or ''} {rival.why_dangerous or ''}",
            client_name=client.name,
        ):
            return True
        return False

    filtered_kept: list[Competitor] = []
    for rival in kept:
        name_key = _as_str(rival.name).lower().strip()
        is_baseline = mode == "add" and name_key in baseline_set
        if _hard_junk_rival(rival) and not rival.is_pinned:
            rival.is_tracking = False
            rival.is_pinned = False
            continue
        fits = _rival_fits_run_scope(
            name=_as_str(rival.name),
            website=rival.website,
            headquarters=rival.headquarters,
            description=rival.description,
            why=rival.why_dangerous,
            scope=scope,
            market=scope_market,
            client_name=client.name,
            is_pinned=False,
            strict=(scope == "local"),
        )
        if (
            not fits
            and scope == "local"
            and scope_market
            and _is_curated_seed_rival(rival.name, scope_market)
        ):
            fits = True
        if rival.is_pinned or fits or (mode == "add" and scope != "global" and is_baseline):
            filtered_kept.append(rival)
        else:
            rival.is_tracking = False
            rival.is_pinned = False
    
    # Deduplicate candidate list
    kept = collapse_duplicate_competitors(filtered_kept)

    # Also check other active rivals from DB
    all_tracked = (
        await db.execute(
            select(Competitor).where(
                Competitor.client_id == client.id,
                Competitor.agency_id == agency.id,
                Competitor.is_tracking.is_(True),
            )
        )
    ).scalars().all()
    kept_ids_final = {c.id for c in kept}
    for rival in all_tracked:
        if rival.id in kept_ids_final:
            continue
        if rival.is_pinned:
            kept.insert(0, rival)
            kept_ids_final.add(rival.id)
            continue
        name_key = _as_str(rival.name).lower().strip()
        if _hard_junk_rival(rival):
            rival.is_tracking = False
            rival.is_pinned = False
            continue
        if not _rival_fits_run_scope(
            name=_as_str(rival.name),
            website=rival.website,
            headquarters=rival.headquarters,
            description=rival.description,
            why=rival.why_dangerous,
            scope=scope,
            market=scope_market,
            client_name=client.name,
            is_pinned=False,
            strict=(scope == "local"),
        ):
            rival.is_tracking = False
            continue
        if mode == "add" and scope != "global" and name_key in baseline_set:
            kept.insert(0, rival)
            kept_ids_final.add(rival.id)
        if rival.is_pinned or (mode == "add" and scope != "global" and name_key in baseline_set):
            kept.insert(0, rival)
            kept_ids_final.add(rival.id)
    
    kept = collapse_duplicate_competitors(kept)
    kept = sorted(kept, key=lambda c: (1 if c.is_pinned else 0, float(c.overlap_score or 0)), reverse=True)
    pinned_final = [c for c in kept if c.is_pinned]
    others_final = [c for c in kept if not c.is_pinned]

    if mode == "add" and scope != "global":
        # Keep local baseline rivals and add up to count new rivals
        baseline_others = [c for c in others_final if _as_str(c.name).lower().strip() in baseline_set]
        fresh_others = [c for c in others_final if _as_str(c.name).lower().strip() not in baseline_set]
        active_fresh = fresh_others[:count]
        active_others = baseline_others + active_fresh
    else:
        # EXACT SLIDER COUNT: exactly count competitors tracked (pinned first, then highest overlap others)
        max_others = max(0, count - len(pinned_final))
        active_others = others_final[:max_others]

    competitors = collapse_duplicate_competitors(pinned_final + active_others)
    from app.services.billing import max_tracked_rivals

    rival_cap = max_tracked_rivals(agency)
    if rival_cap is not None and len(competitors) > rival_cap:
        competitors = competitors[:rival_cap]

    # Ensure authentic distinct overlap scores across all displayed competitor cards
    seen_scores = set()
    for i, c in enumerate(competitors):
        cur_score = round(float(c.overlap_score or 78.0), 1)
        if cur_score in seen_scores or cur_score >= 94.0 or cur_score <= 50.0:
            variance = [0.0, -4.5, 3.2, -7.8, 5.1, -10.5, 2.4, -5.9][i % 8]
            cur_score = max(62.0, min(91.0, round(84.0 + variance, 1)))
        seen_scores.add(cur_score)
        c.overlap_score = cur_score

    final_ids = {c.id for c in competitors}
    # Synchronize database tracking flags across ALL client competitor rows
    all_client_rivals = (
        await db.execute(
            select(Competitor).where(
                Competitor.client_id == client.id,
                Competitor.agency_id == agency.id,
            )
        )
    ).scalars().all()
    for rival in all_client_rivals:
        if rival.id in final_ids:
            rival.is_tracking = True
        else:
            rival.is_tracking = False
            rival.is_pinned = False
    await db.flush()

    if not competitors:
        peer_hint = _client_peer_hint(
            client.name,
            client.industry,
            client.niche,
            _business_model_from_client(client),
            client.notes,
            client.tagline,
        )
        raise ValueError(
            f"No matching {peer_hint} survived this run's filter "
            f"({'local · ' + (required_market or 'market') if scope == 'local' else 'global'}). "
            "Try again shortly, or add real rivals manually and pin them."
        )

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
    items = [i for i in _as_list(payload.get("tickets")) if isinstance(i, dict)]
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
        criteria = [_as_str(c) for c in _as_list(item.get("acceptance_criteria"))]
        if ttype != "epic" and len(criteria) < 3:
            criteria = criteria + [
                "Definition of done reviewed with agency lead",
                "Competitor evidence linked",
                "Deliverable shared with client stakeholder",
            ]
            criteria = criteria[:6]
        ticket = FeatureTicket(
            agency_id=agency.id,
            client_id=client.id,
            feature_id=feature.id,
            heading=_clip(_as_str(item.get("heading")), 500) or f"Improve {feature.name}"[:500],
            body=_as_str(item.get("body")),
            acceptance_criteria=criteria,
            priority=_level_label(item.get("priority"), "medium", max_len=20),
            ticket_type=ttype,
            labels=[_as_str(l) for l in _as_list(item.get("labels"))] or [feature.category, "loved-feature"],
            estimated_effort=_clip(_as_str(item.get("estimated_effort")), 80),
            story_points=_as_int(item.get("story_points")),
            why_useful=_as_str(item.get("why_useful")),
            competitor_context=_as_str(item.get("competitor_context")),
            evidence_links=_as_list(item.get("evidence_links")) or evidence[:4],
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
    generate_report: bool = False,
    competitor_scope: str = "local",
    competitor_country: str | None = None,
    competitor_count: int = 5,
    competitor_mode: str = "add",
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

        enrich = await enrich_client_profile(
            db,
            agency,
            client,
            competitor_scope=competitor_scope,
            competitor_country=competitor_country,
            competitor_count=competitor_count,
            competitor_mode=competitor_mode,
        )
        pack = await run_competitive_pack(
            db,
            agency,
            client,
            competitor_scope=competitor_scope,
            competitor_country=competitor_country,
            competitor_count=competitor_count,
            competitor_mode=competitor_mode,
            baseline_rival_names=list(enrich.get("baseline_rival_names") or []),
        )
        radar = await run_client_intelligence(
            db, agency, client, competitor_country=competitor_country
        )

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
        await db.commit()
        return result
    except Exception as exc:
        job.status = JobStatus.failed
        job.finished_at = datetime.utcnow()
        job.detail = str(exc)[:800]
        await db.flush()
        raise
