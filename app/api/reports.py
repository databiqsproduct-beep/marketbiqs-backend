from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import AuthContext, get_auth_context, get_tenant_client
from app.models import ClientBrand, Report
from app.schemas import ReportGenerateRequest, ReportOut
from app.services.reports import ensure_report_pdf_bytes, generate_client_report

router = APIRouter(tags=["reports"])


@router.get("/clients/{client_id}/reports", response_model=list[ReportOut])
async def list_reports(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Report)
        .where(Report.client_id == client.id, Report.agency_id == ctx.agency.id)
        .order_by(Report.created_at.desc())
    )
    return list(result.scalars().all())


@router.post("/clients/{client_id}/reports", response_model=ReportOut)
async def create_report(
    payload: ReportGenerateRequest,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    from app.services.billing import is_payg

    if not is_payg(ctx.agency) and ctx.agency.reports_used >= ctx.agency.reports_quota:
        raise HTTPException(status_code=402, detail="Report quota exceeded. Add usage packs.")
    report = await generate_client_report(
        db,
        ctx.agency,
        client,
        period_label=payload.period_label,
        created_by=ctx.user.id,
    )
    return report


@router.get("/reports/{report_id}", response_model=ReportOut)
async def get_report(
    report_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    report = await db.get(Report, report_id)
    if not report or report.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/reports/{report_id}/pdf")
async def download_pdf(
    report_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    report = await db.get(Report, report_id)
    if not report or report.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Report not found")

    try:
        data = await ensure_report_pdf_bytes(db, ctx.agency, report)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"PDF not available ({exc})") from exc

    # ASCII-safe filename for Content-Disposition (titles often include em dashes)
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in (report.title or "report"))
    filename = f"{safe[:80] or 'report'}.pdf"
    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )
