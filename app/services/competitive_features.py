"""Feature comparison matrices, descriptions clarification, and Jira ticketing."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

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
    ProductFeature,
)
from app.services import ai as ai_service
from app.services import jira as jira_service
from app.services.competitive_identities import (
    _as_int,
    _as_list,
    _as_str,
    _clip,
    _is_generic_text,
    _level_label,
)

logger = logging.getLogger("marketbiqs.competitive.features")

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

_FEATURE_DESC_PROMPT = (
    "For each feature, write a plain-English description a non-technical agency user can understand. "
    "Rules for every description:\n"
    "1) Exactly 2 to 3 short sentences.\n"
    "2) Explain what the customer gets / what problem it solves, not buzzwords.\n"
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
        ("production-grade ai, not demoware", "AI that is ready for real day-to-day business use, not just a flashy demo"),
        ("production grade ai, not demoware", "AI that is ready for real day-to-day business use, not just a flashy demo"),
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
        f"This is part of their current offering (not a future idea)."
    )


def _normalize_comparison_row(row: dict | str, client_name: str, competitor_name: str) -> dict | None:
    if not isinstance(row, dict):
        if isinstance(row, str) and row.strip():
            row = {"feature_name": row.strip()}
        else:
            return None

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
            "You are an elite competitive intelligence director at Marketbiqs. "
            "Analyze the client vs competitor to produce a sharp, high-signal Feature Comparison Matrix.\n"
            "Return JSON: {\"competitor_name\": string, \"rows\": [{\"feature_name\": string, \"category\": string, \"our_status\": \"leading\"|\"parity\"|\"lagging\", \"competitor_status\": \"leading\"|\"parity\"|\"lagging\", \"note\": string, \"how_competitor_leads\": string, \"how_to_improve\": string, \"confidence_score\": float, \"evidence_strength\": \"low\"|\"medium\"|\"high\", \"citations\": [{\"url\": string, \"snippet\": string, \"source\": string}]}]}\n\n"
            "Strict Rules:\n"
            "1) feature_name must be a concise capability or specialty (2-5 words max, e.g. 'Online Ordering', 'Real-time Tracking', 'Loyalty Program'), NOT a sentence.\n"
            "2) Produce 4 to 6 focused comparison rows: prioritize key competitive vulnerabilities and parity risks, plus at least 1 strength where our client holds parity or leads.\n"
            "3) Adapt to the client's industry: for food/hospitality focus on menu variety, speed, dining experience, or ordering tech; for software/services focus on capability depth, SLA, pricing model, or proof; never use software jargon (like 'demo narrative' or 'workflow fit') for non-tech businesses.\n"
            "4) our_status and competitor_status must each be strictly one of: 'leading', 'parity', 'lagging'.\n"
            "5) 'how_to_improve' must be a tangible counter-move tailored to the client's industry (e.g., bundle offer, guarantee, proof point, staff training, specific campaign), NEVER generic fluff like 'continue to monitor' or 'improve marketing'.\n"
            "6) Citations: only cite the competitor's website domain or snippets provided in the prompt; DO NOT fabricate non-existent subpage URLs.\n"
            "7) BANNED terms: 'None', 'N/A', 'does not have a similar feature', 'both companies have similar features', 'continue to enhance and promote', or 'continue to monitor and improve'."
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
                "Rewrite competitive comparison rows for Marketbiqs. Prioritize where the rival threatens the client. "
                "Return JSON {\"rows\": [...] } with the same schema and strict anti-filler rules. "
                "Ensure every how_competitor_leads and how_to_improve uses concrete, industry-specific strategic language tailored to the client's business."
            ),
            json.dumps(
                {
                    "competitor_name": competitor.name,
                    "client_name": client.name,
                    "client_industry": client.industry,
                    "client_niche": client.niche,
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
                        "how_to_improve": f"Strengthen {name} positioning, proof points, and market offering against {competitor.name}.",
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
                        "how_competitor_leads": f"{competitor.name} presents strong public proof around {feat.name}.",
                        "how_to_improve": f"Highlight and reinforce {feat.name} value proposition against {competitor.name}.",
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
    )
    comparisons_list = comparisons.scalars().all()
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
    for c in comparisons_list:
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
                            for c in comparisons_list[:6]
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
        rival_names = sorted({c.competitor_name for c in comparisons_list}) or ["top rival"]
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

__all__ = [
    "_WEAK_COMPARISON_MARKERS",
    "_FEATURE_DESC_PROMPT",
    "_feature_description_is_thin",
    "_fallback_plain_feature_description",
    "_normalize_comparison_row",
    "_generate_competitor_comparisons",
    "clarify_feature_descriptions",
    "love_feature_and_build_tickets",
    "create_all_feature_tickets_in_jira"
]
