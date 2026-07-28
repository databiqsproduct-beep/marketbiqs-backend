from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import AuthContext, get_auth_context, get_tenant_client
from app.models import ChatMessage, ClientBrand
from app.schemas import ChatMessageOut, ChatRequest
from app.services.intelligence import answer_client_question

router = APIRouter(tags=["assistant"])


@router.get("/clients/{client_id}/chat", response_model=list[ChatMessageOut])
async def chat_history(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.client_id == client.id, ChatMessage.agency_id == ctx.agency.id)
        .order_by(ChatMessage.created_at.asc())
        .limit(100)
    )
    return list(result.scalars().all())


@router.post("/clients/{client_id}/chat", response_model=list[ChatMessageOut])
async def chat(
    payload: ChatRequest,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    user_msg = ChatMessage(
        agency_id=ctx.agency.id,
        client_id=client.id,
        user_id=ctx.user.id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)
    answer = await answer_client_question(db, ctx.agency, client, payload.message)
    assistant_msg = ChatMessage(
        agency_id=ctx.agency.id,
        client_id=client.id,
        user_id=ctx.user.id,
        role="assistant",
        content=answer,
    )
    db.add(assistant_msg)
    await db.flush()
    return [user_msg, assistant_msg]
