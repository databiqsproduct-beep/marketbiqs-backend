from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.deps import AuthContext, get_auth_context, get_current_user
from app.models import Agency, AgencyMember, MemberRole, PlanType, User, WorkspaceMode
from app.schemas import (
    AgencyOut,
    LoginRequest,
    MemberOut,
    OnboardingRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)
from app.security import create_access_token, hash_password, slugify, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = (await db.execute(select(User).where(User.email == payload.email.lower()))).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=payload.email.lower(),
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.flush()

    base_slug = slugify(payload.agency_name)
    slug = base_slug
    i = 1
    while (await db.execute(select(Agency).where(Agency.slug == slug))).scalar_one_or_none():
        slug = f"{base_slug}-{i}"
        i += 1

    mode = WorkspaceMode.agency if payload.workspace_mode != "creator" else WorkspaceMode.creator
    plan = PlanType.agency if mode == WorkspaceMode.agency else PlanType.creator
    agency = Agency(
        name=payload.agency_name,
        slug=slug,
        workspace_mode=mode,
        plan=plan,
        included_clients=10 if mode == WorkspaceMode.agency else 1,
        reports_quota=40 if mode == WorkspaceMode.agency else 10,
        scrape_quota=5000 if mode == WorkspaceMode.agency else 500,
        budget_remaining_cents=45000 if mode == WorkspaceMode.agency else 3000,
    )
    db.add(agency)
    await db.flush()
    db.add(
        AgencyMember(
            agency_id=agency.id,
            user_id=user.id,
            role=MemberRole.owner,
        )
    )
    await db.flush()
    token = create_access_token(user.id, {"agency_id": agency.id})
    return TokenResponse(access_token=token, agency_id=agency.id)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.email == payload.email.lower()))).scalar_one_or_none()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    membership = (
        await db.execute(
            select(AgencyMember).where(AgencyMember.user_id == user.id, AgencyMember.is_active.is_(True))
        )
    ).scalars().first()
    agency_id = membership.agency_id if membership else None
    extra = {"agency_id": agency_id} if agency_id else {}
    return TokenResponse(access_token=create_access_token(user.id, extra), agency_id=agency_id)


@router.get("/me", response_model=dict)
async def me(ctx: AuthContext = Depends(get_auth_context)):
    return {
        "user": UserOut.model_validate(ctx.user),
        "agency": AgencyOut.model_validate(ctx.agency),
        "role": ctx.membership.role.value,
    }


@router.post("/onboarding", response_model=AgencyOut)
async def complete_onboarding(
    payload: OnboardingRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    from app.models import ClientBrand, Competitor

    agency = ctx.agency
    if payload.brand_color:
        agency.brand_color = payload.brand_color
    if payload.logo_url:
        agency.logo_url = payload.logo_url
    if payload.report_footer:
        agency.report_footer = payload.report_footer

    if payload.first_client_name:
        client = ClientBrand(
            agency_id=agency.id,
            name=payload.first_client_name,
            website=payload.first_client_website,
            delivery_emails=[ctx.user.email],
        )
        db.add(client)
        await db.flush()
        if payload.first_competitor_name:
            db.add(
                Competitor(
                    agency_id=agency.id,
                    client_id=client.id,
                    name=payload.first_competitor_name,
                    website=payload.first_competitor_website,
                )
            )

    agency.onboarding_completed = True
    await db.flush()
    return AgencyOut.model_validate(agency)
