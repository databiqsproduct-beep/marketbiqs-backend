from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import AuthContext, get_auth_context, get_tenant_client
from app.models import ClientBrand, Integration, JiraTicket
from app.schemas import JiraConnectRequest, JiraTicketCreate, JiraTicketOut
from app.services import jira as jira_service

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.get("/jira")
async def jira_status(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Integration).where(Integration.agency_id == ctx.agency.id, Integration.provider == "jira")
    )
    row = result.scalar_one_or_none()
    if not row:
        return {"connected": False}
    return {
        "connected": row.is_connected,
        "base_url": row.config.get("base_url"),
        "project_key": row.config.get("project_key"),
        "email": row.config.get("email"),
    }


@router.post("/jira/connect")
async def jira_connect(
    payload: JiraConnectRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    try:
        integration = await jira_service.connect_jira(
            db,
            ctx.agency.id,
            str(payload.base_url),
            payload.email,
            payload.api_token,
            payload.project_key,
            payload.epic_name_field,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"connected": True, "project_key": integration.config.get("project_key")}


@router.post("/jira/disconnect")
async def jira_disconnect(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    ok = await jira_service.disconnect_jira(db, ctx.agency.id)
    return {"connected": False, "disconnected": ok}


@router.post("/clients/{client_id}/jira/tickets", response_model=JiraTicketOut)
async def create_ticket(
    payload: JiraTicketCreate,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    try:
        ticket = await jira_service.create_jira_ticket(
            db,
            ctx.agency.id,
            client.id,
            payload.title,
            payload.description,
            payload.insight_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ticket


@router.get("/clients/{client_id}/jira/tickets", response_model=list[JiraTicketOut])
async def list_tickets(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(JiraTicket)
        .where(JiraTicket.client_id == client.id, JiraTicket.agency_id == ctx.agency.id)
        .order_by(JiraTicket.created_at.desc())
    )
    return list(result.scalars().all())
