import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Agency, ClientBrand, Competitor, Insight, Report, SentimentRecord, TrendSignal
from app.services import ai as ai_service

REPORTS_DIR = Path(__file__).resolve().parents[2] / "storage" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


async def generate_client_report(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    period_label: str = "Weekly",
    created_by: str | None = None,
) -> Report:
    competitors = (
        await db.execute(select(Competitor).where(Competitor.client_id == client.id, Competitor.agency_id == agency.id))
    ).scalars().all()
    trends = (
        await db.execute(
            select(TrendSignal)
            .where(TrendSignal.client_id == client.id, TrendSignal.agency_id == agency.id)
            .order_by(TrendSignal.detected_at.desc())
            .limit(12)
        )
    ).scalars().all()
    sentiments = (
        await db.execute(
            select(SentimentRecord)
            .where(SentimentRecord.client_id == client.id, SentimentRecord.agency_id == agency.id)
            .order_by(SentimentRecord.created_at.desc())
            .limit(12)
        )
    ).scalars().all()
    insights = (
        await db.execute(
            select(Insight)
            .where(Insight.client_id == client.id, Insight.agency_id == agency.id)
            .order_by(Insight.created_at.desc())
            .limit(12)
        )
    ).scalars().all()

    context = {
        "client": client.name,
        "competitors": [c.name for c in competitors],
        "trends": [{"topic": t.topic, "platform": t.platform, "summary": t.summary} for t in trends],
        "sentiments": [{"subject": s.subject, "label": s.label, "score": s.score, "themes": s.themes} for s in sentiments],
        "insights": [{"title": i.title, "body": i.body, "priority": i.priority} for i in insights],
    }

    structured = await ai_service.structured_json(
        db,
        agency.id,
        (
            "You are an agency competitive intelligence analyst. Produce a client-ready white-label report "
            "JSON with keys: summary (string), sections (array of {heading, bullets[]} ). Keep bullets actionable."
        ),
        json.dumps(context)[:12000],
    )

    summary = structured.get("summary") or f"{period_label} competitive intelligence summary for {client.name}."
    sections = structured.get("sections") or [
        {
            "heading": "Executive Overview",
            "bullets": [
                f"Tracked {len(competitors)} competitors for {client.name}.",
                f"Identified {len(trends)} trend signals and {len(sentiments)} sentiment themes.",
            ],
        }
    ]

    report = Report(
        agency_id=agency.id,
        client_id=client.id,
        title=f"{client.name} — {period_label} Intelligence Report",
        period_label=period_label,
        status="ready",
        summary=summary,
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
        # Local PDF path remains the fallback
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
        Paragraph(agency.name, brand),
        Paragraph(report.title, styles["Heading2"]),
        Paragraph(f"Prepared for {client.name} · {report.period_label} · {datetime.utcnow():%Y-%m-%d}", body),
        Spacer(1, 12),
        Paragraph("Executive Summary", heading),
        Paragraph(report.summary.replace("\n", "<br/>"), body),
    ]
    for section in report.sections or []:
        story.append(Paragraph(section.get("heading", "Section"), heading))
        for bullet in section.get("bullets", []):
            story.append(Paragraph(f"• {bullet}", body))
            story.append(Spacer(1, 4))
    if agency.report_footer:
        story.append(Spacer(1, 20))
        story.append(Paragraph(agency.report_footer, body))
    else:
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Confidential · Prepared by {agency.name}", body))

    meta = [
        ["Agency", agency.name],
        ["Client", client.name],
        ["Report ID", report.id],
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
