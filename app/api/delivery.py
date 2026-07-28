from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import AuthContext, get_auth_context, get_tenant_client
from app.models import ClientBrand, DeliveryLog, Report
from app.schemas import DeliveryLogOut, DeliveryRequest
from app.services.delivery import deliver_update, draft_delivery_copy

router = APIRouter(tags=["delivery"])


@router.post("/clients/{client_id}/deliver", response_model=list[DeliveryLogOut])
async def deliver(
    payload: DeliveryRequest,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    report = None
    if payload.report_id:
        report = await db.get(Report, payload.report_id)
        if not report or report.agency_id != ctx.agency.id or report.client_id != client.id:
            raise HTTPException(status_code=404, detail="Report not found")
    message = payload.message
    if not message and report:
        message = await draft_delivery_copy(db, ctx.agency.id, client.name, report.summary)
    logs = await deliver_update(
        db,
        ctx.agency.id,
        ctx.agency.name,
        client,
        report=report,
        channel=payload.channel,
        custom_message=message,
    )
    return logs


@router.get("/clients/{client_id}/deliveries", response_model=list[DeliveryLogOut])
async def delivery_history(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DeliveryLog)
        .where(DeliveryLog.client_id == client.id, DeliveryLog.agency_id == ctx.agency.id)
        .order_by(DeliveryLog.created_at.desc())
        .limit(50)
    )
    return list(result.scalars().all())
