from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Agency, AgencyMember, ClientBrand, User, WhiteLabelApiKey
from app.security import decode_token, hash_api_key

bearer = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    user: User
    agency: Agency
    membership: AgencyMember


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = decode_token(credentials.credentials)
        user_id = payload.get("sub")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    user = await db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


async def get_auth_context(
    user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: AsyncSession = Depends(get_db),
    x_agency_id: str | None = Header(default=None, alias="X-Agency-Id"),
) -> AuthContext:
    stmt = (
        select(AgencyMember)
        .options(selectinload(AgencyMember.agency), selectinload(AgencyMember.user))
        .where(AgencyMember.user_id == user.id, AgencyMember.is_active.is_(True))
    )
    result = await db.execute(stmt)
    memberships = list(result.scalars().all())
    if not memberships:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No agency membership")

    # Prefer JWT claim, then header — ignore stale X-Agency-Id from a previous account/session.
    jwt_agency_id: str | None = None
    if credentials:
        try:
            jwt_agency_id = decode_token(credentials.credentials).get("agency_id")
        except ValueError:
            jwt_agency_id = None

    preferred = jwt_agency_id or x_agency_id
    membership = memberships[0]
    if preferred:
        match = next((m for m in memberships if m.agency_id == preferred), None)
        if match:
            membership = match
        # Stale header that doesn't match this user → fall back to first membership (do not 403)
    return AuthContext(user=user, agency=membership.agency, membership=membership)


async def require_roles(*roles: str):
    async def checker(ctx: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if ctx.membership.role.value not in roles and ctx.membership.role.value != "owner":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return ctx

    return checker


async def get_tenant_client(
    client_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> ClientBrand:
    client = await db.get(ClientBrand, client_id)
    if not client or client.agency_id != ctx.agency.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    return client


async def get_white_label_agency(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Agency:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")
    hashed = hash_api_key(x_api_key)
    stmt = select(WhiteLabelApiKey).where(
        WhiteLabelApiKey.hashed_key == hashed,
        WhiteLabelApiKey.is_active.is_(True),
    )
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if record.requests_used >= record.monthly_quota:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Quota exceeded")
    agency = await db.get(Agency, record.agency_id)
    if not agency:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Agency missing")
    record.requests_used += 1
    await db.flush()
    return agency
