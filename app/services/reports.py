import json
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from xml.sax.saxutils import escape

from app.models import (
    Agency,
    ClientBrand,
    Competitor,
    FeatureComparison,
    GapReport,
    GoalAlert,
    Insight,
    ProductFeature,
    Report,
    SentimentRecord,
    TrendSignal,
)
from app.services import ai as ai_service

logger = logging.getLogger("marketbiqs.reports")

REPORTS_DIR = Path(__file__).resolve().parents[2] / "storage" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _pdf_text(value: object) -> str:
    """Escape text for ReportLab Paragraph (XML) while keeping line breaks."""
    text = "" if value is None else str(value)
    return escape(text).replace("\n", "<br/>")


def _fallback_report(
    client: ClientBrand,
    competitors: list[Competitor],
    features: list[ProductFeature],
    gaps: list[GapReport],
    alerts: list[GoalAlert],
    trends: list[TrendSignal],
    period_label: str,
) -> dict:
    owned = [f for f in features if not f.is_wishlisted]
    wishlist = [f for f in features if f.is_wishlisted]
    rival_names = [c.name for c in competitors[:8]]
    lagging_items: list[str] = []
    for gap in gaps[:5]:
        for item in (gap.lagging or [])[:3]:
            if isinstance(item, str):
                lagging_items.append(f"{gap.competitor_name}: {item}")
            elif isinstance(item, dict):
                lagging_items.append(f"{gap.competitor_name}: {item.get('feature') or item.get('name') or item}")

    summary = (
        f"{period_label} for {client.name}: tracking {len(competitors)} rivals with "
        f"{len(owned)} owned features and {len(wishlist)} wishlist items. "
        f"{'Open alerts need attention now.' if alerts else 'No open specialty alerts this cycle.'}"
    )

    sections = [
        {
            "heading": "Competitive landscape",
            "bullets": [
                f"Priority rivals: {', '.join(rival_names) or 'none tagged yet'}.",
                f"Industry focus: {client.industry or 'not set'} · Website: {client.website or 'not set'}.",
                "Re-run intel after major product or pricing changes to keep this board fresh.",
            ],
        },
        {
            "heading": "Product posture",
            "bullets": [
                f"{len(owned)} shipped capabilities inventoried for {client.name}.",
                f"{len(wishlist)} wishlist items ready for roadmap / Jira push.",
                *(
                    [f"Top owned: {', '.join(f.name for f in owned[:5])}."]
                    if owned
                    else ["Add or refresh features so gap analysis has a baseline."]
                ),
            ],
        },
        {
            "heading": "Gaps & moves",
            "bullets": (
                lagging_items[:6]
                or [
                    "No structured gap rows yet — run intel again once rival pages are reachable.",
                ]
            )
            + [
                *(
                    [f"Alert: {a.title} — {a.action}" for a in alerts[:4]]
                    if alerts
                    else ["No open specialty alerts this run."]
                )
            ],
        },
        {
            "heading": "What to do next",
            "bullets": [
                "Pin the 3 rivals that matter most to your current pitch.",
                "Wishlist the highest-impact missing feature and draft tickets.",
                "Ask the AI assistant for a pre-call brief on the lead rival.",
                *(
                    [f"Watch trend: {t.topic}" for t in trends[:3]]
                    if trends
                    else ["Schedule the next intel refresh within 7 days."]
                ),
            ],
        },
    ]
    return {"summary": summary, "sections": sections}


async def generate_client_report(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    period_label: str = "Weekly",
    created_by: str | None = None,
) -> Report:
    competitors = list(
        (
            await db.execute(
                select(Competitor).where(Competitor.client_id == client.id, Competitor.agency_id == agency.id)
            )
        )
        .scalars()
        .all()
    )
    trends = list(
        (
            await db.execute(
                select(TrendSignal)
                .where(TrendSignal.client_id == client.id, TrendSignal.agency_id == agency.id)
                .order_by(TrendSignal.detected_at.desc())
                .limit(12)
            )
        )
        .scalars()
        .all()
    )
    sentiments = list(
        (
            await db.execute(
                select(SentimentRecord)
                .where(SentimentRecord.client_id == client.id, SentimentRecord.agency_id == agency.id)
                .order_by(SentimentRecord.created_at.desc())
                .limit(12)
            )
        )
        .scalars()
        .all()
    )
    insights = list(
        (
            await db.execute(
                select(Insight)
                .where(Insight.client_id == client.id, Insight.agency_id == agency.id)
                .order_by(Insight.created_at.desc())
                .limit(12)
            )
        )
        .scalars()
        .all()
    )
    features = list(
        (
            await db.execute(
                select(ProductFeature).where(
                    ProductFeature.client_id == client.id, ProductFeature.agency_id == agency.id
                )
            )
        )
        .scalars()
        .all()
    )
    gaps = list(
        (
            await db.execute(
                select(GapReport)
                .where(GapReport.client_id == client.id, GapReport.agency_id == agency.id)
                .order_by(GapReport.created_at.desc())
                .limit(8)
            )
        )
        .scalars()
        .all()
    )
    comparisons = list(
        (
            await db.execute(
                select(FeatureComparison)
                .where(FeatureComparison.client_id == client.id, FeatureComparison.agency_id == agency.id)
                .order_by(FeatureComparison.created_at.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )
    alerts = list(
        (
            await db.execute(
                select(GoalAlert)
                .where(GoalAlert.client_id == client.id, GoalAlert.agency_id == agency.id)
                .order_by(GoalAlert.created_at.desc())
                .limit(10)
            )
        )
        .scalars()
        .all()
    )

    owned = [f for f in features if not f.is_wishlisted]
    wishlist = [f for f in features if f.is_wishlisted]

    context = {
        "client": {
            "name": client.name,
            "industry": client.industry,
            "website": client.website,
            "niche": getattr(client, "niche", None),
        },
        "competitors": [
            {
                "name": c.name,
                "website": c.website,
                "description": (c.description or "")[:280],
                "pinned": bool(getattr(c, "is_pinned", False)),
            }
            for c in competitors[:12]
        ],
        "owned_features": [{"name": f.name, "category": f.category} for f in owned[:20]],
        "wishlist_features": [{"name": f.name, "category": f.category} for f in wishlist[:12]],
        "gaps": [
            {
                "competitor": g.competitor_name,
                "summary": (g.summary or "")[:300],
                "lagging": (g.lagging or [])[:5],
                "opportunities": (g.opportunities or [])[:5],
            }
            for g in gaps[:6]
        ],
        "comparisons": [
            {
                "competitor": row.competitor_name,
                "feature": row.feature_name,
                "our_status": row.our_status,
                "competitor_status": row.competitor_status,
                "improve": (row.how_to_improve or "")[:180],
            }
            for row in comparisons[:15]
        ],
        "alerts": [
            {"title": a.title, "impact": a.impact, "action": (a.action or "")[:200]} for a in alerts[:8]
        ],
        "trends": [{"topic": t.topic, "platform": t.platform, "summary": t.summary} for t in trends],
        "sentiments": [
            {"subject": s.subject, "label": s.label, "score": s.score, "themes": s.themes} for s in sentiments
        ],
        "insights": [{"title": i.title, "body": i.body, "priority": i.priority} for i in insights],
    }

    structured = await ai_service.structured_json(
        db,
        agency.id,
        (
            "You are an agency competitive intelligence analyst writing a client-ready white-label brief. "
            "Return JSON with keys: summary (2-4 friendly sentences), sections (array of "
            "{heading, bullets[]} with 3-6 actionable bullets each). "
            "Use only the provided workspace data. Prefer concrete rival/feature names. "
            "Tone: clear, confident, useful for an account team."
        ),
        json.dumps(context)[:14000],
        temperature=0.35,
    )

    summary = structured.get("summary") if isinstance(structured, dict) else None
    sections = structured.get("sections") if isinstance(structured, dict) else None

    if (
        not summary
        or not isinstance(sections, list)
        or not sections
        or ai_service.is_fallback_text(str(summary))
    ):
        logger.warning(
            "Using deterministic report fallback for client=%s (AI summary missing/invalid)",
            client.id,
        )
        fallback = _fallback_report(client, competitors, features, gaps, alerts, trends, period_label)
        summary = fallback["summary"]
        sections = fallback["sections"]
    else:
        # Normalize section shape
        clean_sections = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            heading = str(section.get("heading") or "Insight").strip()
            bullets = section.get("bullets") or []
            if not isinstance(bullets, list):
                bullets = [str(bullets)]
            bullets = [str(b).strip() for b in bullets if str(b).strip()]
            if heading and bullets:
                clean_sections.append({"heading": heading, "bullets": bullets})
        sections = clean_sections or _fallback_report(
            client, competitors, features, gaps, alerts, trends, period_label
        )["sections"]

    report = Report(
        agency_id=agency.id,
        client_id=client.id,
        title=f"{client.name} — {period_label} Intelligence Report",
        period_label=period_label,
        status="ready",
        summary=str(summary).strip(),
        sections=sections,
        white_labeled=True,
        created_by=created_by,
    )
    db.add(report)
    await db.flush()

    pdf_path = await _write_pdf(agency, client, report)
    report.pdf_path = str(pdf_path)
    try:
        from app.services.supabase_client import upload_report_pdf

        remote = await upload_report_pdf(report.id, Path(pdf_path).read_bytes())
        if remote:
            report.pdf_path = remote
    except Exception:
        pass
    agency.reports_used += 1
    await db.flush()
    return report


async def _write_pdf(agency: Agency, client: ClientBrand, report: Report) -> Path:
    path = REPORTS_DIR / f"{report.id}.pdf"
    styles = getSampleStyleSheet()
    brand = ParagraphStyle(
        "Brand",
        parent=styles["Heading1"],
        textColor=colors.HexColor(agency.brand_color or "#0F766E"),
        spaceAfter=8,
    )
    heading = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        textColor=colors.HexColor(agency.brand_secondary or "#134E4A"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body = styles["BodyText"]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    story = [
        Paragraph(_pdf_text(agency.name), brand),
        Paragraph(_pdf_text(report.title), styles["Heading2"]),
        Paragraph(
            _pdf_text(f"Prepared for {client.name} · {report.period_label} · {datetime.utcnow():%Y-%m-%d}"),
            body,
        ),
        Spacer(1, 12),
        Paragraph("Executive Summary", heading),
        Paragraph(_pdf_text(report.summary or "No summary available."), body),
    ]
    for section in report.sections or []:
        if not isinstance(section, dict):
            continue
        story.append(Paragraph(_pdf_text(section.get("heading") or "Section"), heading))
        for bullet in section.get("bullets") or []:
            story.append(Paragraph(f"• {_pdf_text(bullet)}", body))
            story.append(Spacer(1, 4))
    if agency.report_footer:
        story.append(Spacer(1, 20))
        story.append(Paragraph(_pdf_text(agency.report_footer), body))
    else:
        story.append(Spacer(1, 20))
        story.append(Paragraph(_pdf_text(f"Confidential · Prepared by {agency.name}"), body))

    meta = [
        ["Agency", _pdf_text(agency.name)],
        ["Client", _pdf_text(client.name)],
        ["Report ID", _pdf_text(report.id)],
        ["Generated", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
    ]
    table = Table(meta, colWidths=[1.5 * inch, 5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(agency.brand_color or "#0F766E")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(Spacer(1, 18))
    story.append(table)
    doc.build(story)
    return path


async def ensure_report_pdf_bytes(
    db: AsyncSession,
    agency: Agency,
    report: Report,
) -> bytes:
    """
    Return PDF bytes for a report.

    Prefer Supabase Storage, then local disk. If neither exists (common on Railway
    after redeploy), regenerate from stored report content and re-upload.
    """
    # 1) Supabase
    if report.pdf_path and report.pdf_path.startswith("supabase://"):
        try:
            from app.services.supabase_client import download_report_pdf_bytes

            data = await download_report_pdf_bytes(report.pdf_path)
            if data:
                return data
        except Exception as exc:
            logger.warning("Supabase PDF download failed for %s: %s", report.id, exc)

    # 2) Local path recorded on the report
    candidates: list[Path] = []
    if report.pdf_path and not report.pdf_path.startswith("supabase://"):
        candidates.append(Path(report.pdf_path))
    candidates.append(REPORTS_DIR / f"{report.id}.pdf")
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                return path.read_bytes()
        except OSError:
            continue

    # 3) Regenerate from DB content
    client = await db.get(ClientBrand, report.client_id)
    if not client:
        raise FileNotFoundError("Client missing for report PDF regeneration")
    path = await _write_pdf(agency, client, report)
    data = path.read_bytes()
    report.pdf_path = str(path)
    try:
        from app.services.supabase_client import upload_report_pdf

        remote = await upload_report_pdf(report.id, data)
        if remote:
            report.pdf_path = remote
    except Exception as exc:
        logger.warning("Could not re-upload regenerated PDF %s: %s", report.id, exc)
    await db.flush()
    return data
