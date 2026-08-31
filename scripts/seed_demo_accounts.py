"""Seed local agency and client demo login credentials directly into the database."""
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.database import init_db, AsyncSessionLocal
from app.models import (
    Agency,
    AgencyMember,
    ClientBrand,
    Competitor,
    DeliveryChannel,
    Insight,
    MemberRole,
    PlanType,
    Report,
    User,
    WorkspaceMode,
)
from app.security import hash_password

AGENCY_EMAIL = "agency@marketbiqs.com"
AGENCY_PASSWORD = "AgencyPass123!"
AGENCY_NAME = "Apex Growth Agency"
AGENCY_USER_NAME = "Maarij Agency Admin"

CLIENT_EMAIL = "client@acmeretail.com"
CLIENT_PASSWORD = "ClientPass123!"
CLIENT_NAME = "Acme Retail Co."
CLIENT_USER_NAME = "Sarah Client Lead"


async def seed():
    print("Initializing database schema and applying patches...")
    await init_db()

    async with AsyncSessionLocal() as session:
        # 1. AGENCY USER
        agency_user = (
            await session.execute(select(User).where(User.email == AGENCY_EMAIL))
        ).scalar_one_or_none()

        if not agency_user:
            agency_user = User(
                id=str(uuid.uuid4()),
                email=AGENCY_EMAIL,
                full_name=AGENCY_USER_NAME,
                hashed_password=hash_password(AGENCY_PASSWORD),
                is_active=True,
            )
            session.add(agency_user)
            await session.flush()
        else:
            agency_user.full_name = AGENCY_USER_NAME
            agency_user.hashed_password = hash_password(AGENCY_PASSWORD)
            agency_user.is_active = True

        # 2. AGENCY WORKSPACE
        agency_slug = "apex-growth-agency"
        agency = (
            await session.execute(select(Agency).where(Agency.slug == agency_slug))
        ).scalar_one_or_none()

        if not agency:
            agency = Agency(
                id=str(uuid.uuid4()),
                name=AGENCY_NAME,
                slug=agency_slug,
                workspace_mode=WorkspaceMode.agency,
                plan=PlanType.agency,
                brand_color="#0f766e",
                brand_secondary="#134e4a",
                onboarding_completed=True,
                included_clients=20,
                reports_quota=100,
                scrape_quota=1000,
                budget_remaining_cents=24900,
            )
            session.add(agency)
            await session.flush()
        else:
            agency.name = AGENCY_NAME
            agency.onboarding_completed = True

        # 3. AGENCY OWNER MEMBERSHIP
        owner_member = (
            await session.execute(
                select(AgencyMember).where(
                    AgencyMember.agency_id == agency.id,
                    AgencyMember.user_id == agency_user.id,
                )
            )
        ).scalar_one_or_none()

        if not owner_member:
            session.add(
                AgencyMember(
                    id=str(uuid.uuid4()),
                    agency_id=agency.id,
                    user_id=agency_user.id,
                    role=MemberRole.owner,
                    is_active=True,
                )
            )

        # 4. CLIENT USER
        client_user = (
            await session.execute(select(User).where(User.email == CLIENT_EMAIL))
        ).scalar_one_or_none()

        if not client_user:
            client_user = User(
                id=str(uuid.uuid4()),
                email=CLIENT_EMAIL,
                full_name=CLIENT_USER_NAME,
                hashed_password=hash_password(CLIENT_PASSWORD),
                is_active=True,
            )
            session.add(client_user)
            await session.flush()
        else:
            client_user.full_name = CLIENT_USER_NAME
            client_user.hashed_password = hash_password(CLIENT_PASSWORD)
            client_user.is_active = True

        # 5. CLIENT USER MEMBERSHIP (Analyst)
        client_member = (
            await session.execute(
                select(AgencyMember).where(
                    AgencyMember.agency_id == agency.id,
                    AgencyMember.user_id == client_user.id,
                )
            )
        ).scalar_one_or_none()

        if not client_member:
            session.add(
                AgencyMember(
                    id=str(uuid.uuid4()),
                    agency_id=agency.id,
                    user_id=client_user.id,
                    role=MemberRole.analyst,
                    is_active=True,
                )
            )

        # 6. CLIENT BRAND 1: Acme Retail Co.
        client_brand = (
            await session.execute(
                select(ClientBrand).where(
                    ClientBrand.agency_id == agency.id,
                    ClientBrand.name == CLIENT_NAME,
                )
            )
        ).scalar_one_or_none()

        if not client_brand:
            client_brand = ClientBrand(
                id=str(uuid.uuid4()),
                agency_id=agency.id,
                name=CLIENT_NAME,
                industry="E-Commerce & Direct-to-Consumer",
                niche="Sustainable Apparel & Activewear",
                website="https://acmeretail.example.com",
                tagline="High-performance eco-activewear for modern athletes.",
                delivery_channel=DeliveryChannel.email,
                delivery_emails=[CLIENT_EMAIL],
                delivery_schedule_cron="0 9 * * 1",
                is_active=True,
            )
            session.add(client_brand)
            await session.flush()

        # Competitors for Acme Retail
        comp_data = [
            ("Rival Athletic Co.", "https://rivalathletic.example.com", "Main direct rival focusing on budget gym wear"),
            ("EcoFlex Performance", "https://ecoflex.example.com", "Eco-friendly premium yoga apparel"),
            ("Stride Athletics", "https://stride.example.com", "Fast shipping, subscription gym apparel"),
        ]
        for c_name, c_url, c_desc in comp_data:
            existing_comp = (
                await session.execute(
                    select(Competitor).where(
                        Competitor.client_id == client_brand.id,
                        Competitor.name == c_name,
                    )
                )
            ).scalar_one_or_none()
            if not existing_comp:
                session.add(
                    Competitor(
                        id=str(uuid.uuid4()),
                        agency_id=agency.id,
                        client_id=client_brand.id,
                        name=c_name,
                        website=c_url,
                        description=c_desc,
                        threat_level="high" if "Rival" in c_name else "medium",
                        is_tracking=True,
                    )
                )

        # Insights for Acme Retail
        existing_insight = (
            await session.execute(
                select(Insight).where(Insight.client_id == client_brand.id)
            )
        ).first()

        if not existing_insight:
            insights = [
                (
                    "Competitor Pricing Shift",
                    "Rival Athletic dropped starter tights pricing by 18% for clearance.",
                    "pricing",
                    "high",
                ),
                (
                    "Untapped Ad Angle Opportunity",
                    "72% of competitor reviews complain about fabric durability. Recommend launching comparison campaign.",
                    "positioning",
                    "high",
                ),
                (
                    "Landing Page Headline Change",
                    "EcoFlex updated headline to 'Carbon Neutral Leggings in 48 Hours'.",
                    "feature_gap",
                    "medium",
                ),
            ]
            for title, desc, cat, priority in insights:
                session.add(
                    Insight(
                        id=str(uuid.uuid4()),
                        agency_id=agency.id,
                        client_id=client_brand.id,
                        title=title,
                        body=desc,
                        category=cat,
                        priority=priority,
                    )
                )

        # Executive Report for Acme Retail
        existing_report = (
            await session.execute(
                select(Report).where(Report.client_id == client_brand.id)
            )
        ).first()

        if not existing_report:
            session.add(
                Report(
                    id=str(uuid.uuid4()),
                    agency_id=agency.id,
                    client_id=client_brand.id,
                    title="Q3 Competitive Intelligence & Positioning Deck",
                    summary="Comprehensive evaluation of Rival Athletic, EcoFlex, and Stride Athletics across pricing, ad creative, and positioning.",
                    sections=[
                        {
                            "title": "Key Executive Takeaways",
                            "content": "Rival Athletic reduced entry-tier pricing by 18%. Acme Retail maintains 24% higher customer sentiment on product durability.",
                        },
                        {
                            "title": "Action Plan",
                            "content": "Counter price pressure with 2-pack value bundles. Scale creator video ads showcasing gym stress tests.",
                        },
                    ],
                    status="ready",
                )
            )

        # 7. CLIENT BRAND 2: Databiqs Analytics
        brand2 = (
            await session.execute(
                select(ClientBrand).where(
                    ClientBrand.agency_id == agency.id,
                    ClientBrand.name == "Databiqs Analytics",
                )
            )
        ).scalar_one_or_none()

        if not brand2:
            brand2 = ClientBrand(
                id=str(uuid.uuid4()),
                agency_id=agency.id,
                name="Databiqs Analytics",
                industry="SaaS & Big Data",
                niche="Product Intelligence & Behavioral Analytics",
                website="https://www.databiqs.com",
                tagline="Real-time product analytics for modern product teams.",
                is_active=True,
            )
            session.add(brand2)
            await session.flush()

        await session.commit()
        print("\n" + "=" * 70)
        print("SEEDING COMPLETE — DEMO CREDENTIALS READY")
        print("=" * 70)
        print("1. AGENCY OWNER LOGIN:")
        print(f"   Email:     {AGENCY_EMAIL}")
        print(f"   Password:  {AGENCY_PASSWORD}")
        print(f"   Name:      {AGENCY_USER_NAME}")
        print(f"   Agency:    {AGENCY_NAME} (Role: Owner)")
        print(f"   Dashboard: http://localhost:3000/dashboard")
        print("-" * 70)
        print("2. CLIENT USER LOGIN:")
        print(f"   Email:     {CLIENT_EMAIL}")
        print(f"   Password:  {CLIENT_PASSWORD}")
        print(f"   Name:      {CLIENT_USER_NAME}")
        print(f"   Client:    {CLIENT_NAME}")
        print(f"   Portal:    http://localhost:3000/portal/{client_brand.id}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(seed())
