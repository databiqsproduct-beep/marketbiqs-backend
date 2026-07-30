import json
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal, get_db
from app.deps import AuthContext, get_auth_context, get_tenant_client
from app.models import Agency, ChatMessage, ClientBrand
from app.schemas import ChatMessageOut, ChatRequest
from app.services.intelligence import answer_client_question, stream_client_question

router = APIRouter(tags=["assistant"])


def _message_payload(msg: ChatMessage) -> dict:
    created = msg.created_at or datetime.utcnow()
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "created_at": created.isoformat(),
    }


async def _recent_history(
    db: AsyncSession,
    agency_id: str,
    client_id: str,
    *,
    exclude_id: str | None = None,
    limit: int = 12,
) -> list[dict[str, str]]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.client_id == client_id, ChatMessage.agency_id == agency_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit + (1 if exclude_id else 0))
    )
    rows = list(reversed(result.scalars().all()))
    history: list[dict[str, str]] = []
    for row in rows:
        if exclude_id and row.id == exclude_id:
            continue
        if row.role in {"user", "assistant"} and row.content:
            history.append({"role": row.role, "content": row.content})
    return history[-limit:]


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
    await db.flush()
    history = await _recent_history(db, ctx.agency.id, client.id, exclude_id=user_msg.id)
    answer = await answer_client_question(db, ctx.agency, client, payload.message, history=history)
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


@router.post("/clients/{client_id}/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    """SSE stream: user → delta* → done (persisted assistant message)."""
    question = payload.message.strip()
    user_msg = ChatMessage(
        agency_id=ctx.agency.id,
        client_id=client.id,
        user_id=ctx.user.id,
        role="user",
        content=question,
    )
    db.add(user_msg)
    await db.flush()
    await db.commit()
    await db.refresh(user_msg)

    history = await _recent_history(db, ctx.agency.id, client.id, exclude_id=user_msg.id)
    agency_id = ctx.agency.id
    client_id = client.id
    user_id = ctx.user.id
    user_payload = _message_payload(user_msg)

    async def event_gen():
        yield f"data: {json.dumps({'type': 'user', 'message': user_payload})}\n\n"
        chunks: list[str] = []
        try:
            async with AsyncSessionLocal() as stream_db:
                agency = await stream_db.get(Agency, agency_id)
                brand = await stream_db.get(ClientBrand, client_id)
                if not agency or not brand:
                    yield f"data: {json.dumps({'type': 'error', 'detail': 'Workspace not found'})}\n\n"
                    return
                async for delta in stream_client_question(
                    stream_db, agency, brand, question, history=history
                ):
                    chunks.append(delta)
                    yield f"data: {json.dumps({'type': 'delta', 'content': delta})}\n\n"

                full = "".join(chunks).strip() or (
                    "I couldn’t put an answer together from the current workspace data. "
                    "Try running an intelligence refresh, then ask again."
                )
                assistant_msg = ChatMessage(
                    agency_id=agency_id,
                    client_id=client_id,
                    user_id=user_id,
                    role="assistant",
                    content=full,
                )
                stream_db.add(assistant_msg)
                await stream_db.commit()
                await stream_db.refresh(assistant_msg)
                yield f"data: {json.dumps({'type': 'done', 'message': _message_payload(assistant_msg)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)[:400]})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
