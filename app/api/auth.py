from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.config import get_settings
from app.deps import AuthContext, get_auth_context, get_current_user
from app.models import Agency, AgencyMember, MemberRole, PlanType, User, WorkspaceMode
from app.schemas import (
    AgencyOut,
    BootstrapRequest,
    OnboardingRequest,
    UserOut,
)
from app.security import slugify

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


def _agency_for_mode(name: str, slug: str, workspace_mode: str) -> Agency:
    mode = WorkspaceMode.agency if workspace_mode != "creator" else WorkspaceMode.creator
    plan = PlanType.agency if mode == WorkspaceMode.agency else PlanType.creator
    return Agency(
        name=name,
        slug=slug,
        workspace_mode=mode,
        plan=plan,
        included_clients=settings.included_clients if mode == WorkspaceMode.agency else settings.creator_included_clients,
        reports_quota=(
            settings.included_reports_per_month
            if mode == WorkspaceMode.agency
            else settings.creator_included_reports_per_month
        ),
        scrape_quota=(
            settings.included_scrape_units
            if mode == WorkspaceMode.agency
            else settings.creator_included_scrape_units
        ),
        budget_remaining_cents=(
            settings.agency_base_price if mode == WorkspaceMode.agency else settings.creator_base_price
        ),
    )


async def _unique_agency_slug(db: AsyncSession, agency_name: str) -> str:
    base_slug = slugify(agency_name)
    slug = base_slug
    i = 1
    while (await db.execute(select(Agency).where(Agency.slug == slug))).scalar_one_or_none():
        slug = f"{base_slug}-{i}"
        i += 1
    return slug


@router.post("/register", status_code=status.HTTP_410_GONE)
async def register_gone():
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Use Supabase Auth signUp, then POST /api/auth/bootstrap",
    )


@router.post("/login", status_code=status.HTTP_410_GONE)
async def login_gone():
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Use Supabase Auth signIn; send the access token as Bearer",
    )


@router.post("/bootstrap", response_model=dict)
async def bootstrap_agency(
    payload: BootstrapRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create Agency + owner membership for a Supabase-authenticated user (idempotent)."""
    if payload.full_name and payload.full_name.strip():
        user.full_name = payload.full_name.strip()[:255]

    existing = (
        await db.execute(
            select(AgencyMember)
            .options(selectinload(AgencyMember.agency))
            .where(AgencyMember.user_id == user.id, AgencyMember.is_active.is_(True))
        )
    ).scalars().first()
    if existing:
        return {
            "user": UserOut.model_validate(user),
            "agency": AgencyOut.model_validate(existing.agency),
            "role": existing.role.value,
            "needs_bootstrap": False,
            "created": False,
        }

    slug = await _unique_agency_slug(db, payload.agency_name)
    agency = _agency_for_mode(payload.agency_name.strip(), slug, payload.workspace_mode)
    db.add(agency)
    await db.flush()
    membership = AgencyMember(
        agency_id=agency.id,
        user_id=user.id,
        role=MemberRole.owner,
    )
    db.add(membership)
    await db.flush()
    return {
        "user": UserOut.model_validate(user),
        "agency": AgencyOut.model_validate(agency),
        "role": membership.role.value,
        "needs_bootstrap": False,
        "created": True,
    }


@router.get("/me", response_model=dict)
async def me(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    membership = (
        await db.execute(
            select(AgencyMember)
            .options(selectinload(AgencyMember.agency))
            .where(AgencyMember.user_id == user.id, AgencyMember.is_active.is_(True))
        )
    ).scalars().first()
    if not membership:
        return {
            "user": UserOut.model_validate(user),
            "agency": None,
            "role": None,
            "needs_bootstrap": True,
        }
    return {
        "user": UserOut.model_validate(user),
        "agency": AgencyOut.model_validate(membership.agency),
        "role": membership.role.value,
        "needs_bootstrap": False,
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
