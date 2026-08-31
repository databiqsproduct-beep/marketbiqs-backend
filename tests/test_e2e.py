import os
import unittest
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.database import Base, get_db
from app.main import app
from app.models import (
    Agency,
    AgencyMember,
    ClientBrand,
    Competitor,
    ChatMessage,
    FeatureTicket,
    MemberRole,
    PlanType,
    Report,
    User,
    WhiteLabelApiKey,
    WorkspaceMode,
)
from app.security import hash_password

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture(autouse=True)
async def prepare_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.asyncio
async def test_full_e2e_flow():
    """End-to-End testing of all core functional modules of MarketBiqs."""

    # 1. Setup User and Agency in DB
    async with TestSessionLocal() as db:
        hashed_pwd = hash_password("TestPassword123!")
        user = User(
            id="usr_e2e_001",
            email="owner@e2eagency.com",
            full_name="E2E Owner",
            hashed_password=hashed_pwd,
        )
        agency = Agency(
            id="agc_e2e_001",
            name="E2E Marketing Agency",
            slug="e2e-marketing-agency",
            workspace_mode=WorkspaceMode.agency,
            plan=PlanType.agency,
            included_clients=10,
            reports_quota=40,
            scrape_quota=5000,
        )
        member = AgencyMember(
            id="mem_e2e_001",
            agency_id=agency.id,
            user_id=user.id,
            role=MemberRole.owner,
        )
        db.add_all([user, agency, member])
        await db.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # --- 2. AUTHENTICATION ---
        # 2a. Login Local Success
        res = await ac.post("/api/auth/login-local", json={"email": "owner@e2eagency.com", "password": "TestPassword123!"})
        assert res.status_code == 200, f"Login failed: {res.text}"
        auth_data = res.json()
        assert "access_token" in auth_data
        token = auth_data["access_token"]
        assert auth_data["agency"]["id"] == "agc_e2e_001"
        assert auth_data["role"] == "owner"

        headers = {
            "Authorization": f"Bearer {token}",
            "X-Agency-Id": "agc_e2e_001",
        }

        # 2b. Login Local Invalid Credentials
        res_fail = await ac.post("/api/auth/login-local", json={"email": "owner@e2eagency.com", "password": "WrongPassword"})
        assert res_fail.status_code == 401

        # 2c. Auth /me
        res_me = await ac.get("/api/auth/me", headers=headers)
        assert res_me.status_code == 200
        assert res_me.json()["user"]["email"] == "owner@e2eagency.com"
        assert res_me.json()["agency"]["name"] == "E2E Marketing Agency"

        # 2d. Deprecated 410 endpoints check
        res_reg = await ac.post("/api/auth/register", json={})
        assert res_reg.status_code == 410

        # --- 3. AGENCY & TEAM MANAGEMENT ---
        # 3a. Get agency profile via auth me
        res_agency = await ac.get("/api/auth/me", headers=headers)
        assert res_agency.status_code == 200
        assert res_agency.json()["agency"]["id"] == "agc_e2e_001"

        # 3b. Update agency brand
        res_brand = await ac.patch(
            "/api/agency/branding",
            headers=headers,
            json={
                "brand_color": "#FF5722",
                "report_footer": "E2E Custom Footer",
            },
        )
        assert res_brand.status_code == 200
        assert res_brand.json()["brand_color"] == "#ff5722"

        # 3c. Dashboard stats
        res_dash = await ac.get("/api/agency/dashboard", headers=headers)
        assert res_dash.status_code == 200
        dash = res_dash.json()
        assert "clients_count" in dash
        assert dash["clients_count"] == 0

        # 3d. Team members list
        res_members = await ac.get("/api/agency/members", headers=headers)
        assert res_members.status_code == 200
        assert len(res_members.json()) >= 1
        assert res_members.json()[0]["user"]["email"] == "owner@e2eagency.com"

        # --- 4. CLIENT MANAGEMENT ---
        # 4a. Create client brand
        client_payload = {
            "name": "Cheezious Pizza",
            "industry": "fast food",
            "niche": "pizza delivery",
            "website": "https://www.cheezious.com",
            "market_area": "Pakistan",
            "tagline": "Real Cheese Real Taste",
        }
        res_client = await ac.post("/api/clients", headers=headers, json=client_payload)
        assert res_client.status_code == 200, f"Create client failed: {res_client.text}"
        client_data = res_client.json()
        client_id = client_data["id"]
        assert client_data["name"] == "Cheezious Pizza"
        assert client_data["rivals_count"] == 0

        # 4b. List clients
        res_list_clients = await ac.get("/api/clients", headers=headers)
        assert res_list_clients.status_code == 200
        assert len(res_list_clients.json()) == 1

        # 4c. Get client by ID
        res_get_client = await ac.get(f"/api/clients/{client_id}", headers=headers)
        assert res_get_client.status_code == 200
        assert res_get_client.json()["id"] == client_id

        # 4d. Update client
        res_up_client = await ac.patch(
            f"/api/clients/{client_id}",
            headers=headers,
            json={"tagline": "Updated Tagline"},
        )
        assert res_up_client.status_code == 200
        assert res_up_client.json()["tagline"] == "Updated Tagline"

        # --- 5. COMPETITOR DISCOVERY & MANAGEMENT ---
        # 5a. Add competitor manually
        comp_payload = {
            "name": "Broadway Pizza",
            "website": "https://broadwaypizza.com.pk",
            "industry": "fast food",
            "niche": "pizza delivery",
        }
        res_add_comp = await ac.post(f"/api/clients/{client_id}/competitors", headers=headers, json=comp_payload)
        assert res_add_comp.status_code == 200
        comp_data = res_add_comp.json()
        comp_id = comp_data["id"]
        assert comp_data["name"] == "Broadway Pizza"

        # 5b. List competitors
        res_comps = await ac.get(f"/api/clients/{client_id}/competitors", headers=headers)
        assert res_comps.status_code == 200
        assert len(res_comps.json()) == 1

        # 5c. Get client workspace (combines rivals, features, reports, radar)
        res_ws = await ac.get(f"/api/clients/{client_id}/workspace", headers=headers)
        assert res_ws.status_code == 200

        # 5d. Trigger auto-run intel with competitor_mode="add" ("add new" dropdown selection)
        res_run = await ac.post(
            f"/api/clients/{client_id}/auto-run",
            headers=headers,
            json={
                "competitor_scope": "local",
                "competitor_country": "Pakistan",
                "competitor_count": 3,
                "competitor_mode": "add",
                "generate_report": False,
            },
        )
        assert res_run.status_code == 202
        assert res_run.json()["status"] == "queued"
        assert res_run.json()["competitor_mode"] == "add"
        assert res_run.json()["competitor_count"] == 3

        # 5e. Check client features list
        res_feat = await ac.get(f"/api/clients/{client_id}/features", headers=headers)
        assert res_feat.status_code == 200

        # --- 6. AI ASSISTANT & CHAT ---
        # 6a. Post chat message to client
        res_chat = await ac.post(
            f"/api/clients/{client_id}/chat",
            headers=headers,
            json={"message": "What are our competitive advantages over Broadway Pizza?"},
        )
        assert res_chat.status_code == 200
        msgs = res_chat.json()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

        # 6b. Get chat history
        res_history = await ac.get(f"/api/clients/{client_id}/chat", headers=headers)
        assert res_history.status_code == 200
        assert len(res_history.json()) == 2

        # --- 7. REPORTS & EXPORTS ---
        # 7a. Generate client report
        res_gen_report = await ac.post(
            f"/api/clients/{client_id}/reports",
            headers=headers,
            json={"period_label": "Monthly E2E Intel"},
        )
        assert res_gen_report.status_code == 200
        report_data = res_gen_report.json()
        report_id = report_data["id"]

        # 7b. List reports
        res_reports = await ac.get(f"/api/clients/{client_id}/reports", headers=headers)
        assert res_reports.status_code == 200
        assert len(res_reports.json()) >= 1

        # 7c. Get report by ID
        res_rep_get = await ac.get(f"/api/reports/{report_id}", headers=headers)
        assert res_rep_get.status_code == 200
        assert res_rep_get.json()["id"] == report_id

        # 7d. Get PDF report
        res_pdf = await ac.get(f"/api/reports/{report_id}/pdf", headers=headers)
        assert res_pdf.status_code == 200
        assert res_pdf.headers["content-type"] == "application/pdf"

        # --- 8. DELIVERY & MESSAGING ---
        # 8a. Get delivery history
        res_logs = await ac.get(f"/api/clients/{client_id}/deliveries", headers=headers)
        assert res_logs.status_code == 200
        assert isinstance(res_logs.json(), list)

        # 8b. Manual send report delivery
        res_send = await ac.post(
            f"/api/clients/{client_id}/deliver",
            headers=headers,
            json={
                "report_id": report_id,
                "channel": "email",
                "message": "Here is your monthly report update.",
            },
        )
        assert res_send.status_code == 200
        assert isinstance(res_send.json(), list)

        # --- 9. BILLING & QUOTAS ---
        # 9a. Catalog endpoint
        res_cat = await ac.get("/api/billing/catalog", headers=headers)
        assert res_cat.status_code == 200
        assert "plans" in res_cat.json()

        # 9b. Budget endpoint
        res_bud = await ac.get("/api/billing/budget", headers=headers)
        assert res_bud.status_code == 200
        assert res_bud.json()["plan_name"] in ("Agency", "PAYG", "Agency / Growth")

        # --- 10. WHITELABEL API KEYS ---
        # 10a. Create API Key
        res_key = await ac.post(
            "/api/agency/white-label-keys",
            headers=headers,
            json={"name": "Partner Integration Key", "monthly_quota": 5000},
        )
        assert res_key.status_code == 200
        api_key = res_key.json()["api_key"]

        # 10b. List API Keys
        res_keys = await ac.get("/api/agency/white-label-keys", headers=headers)
        assert res_keys.status_code == 200
        assert len(res_keys.json()) == 1

        # 10c. Embed API call with X-API-Key
        res_embed_clients = await ac.get(
            "/api/v1/clients",
            headers={"X-API-Key": api_key},
        )
        assert res_embed_clients.status_code == 200
        assert len(res_embed_clients.json()) == 1

        # --- 11. HELPDESK & SUPPORT ---
        # 11a. Get helpdesk knowledge FAQs and issues
        res_topics = await ac.get("/api/helpdesk/knowledge", headers=headers)
        assert res_topics.status_code == 200
        assert "faqs" in res_topics.json()

        # --- 12. INTEGRATIONS (Jira) ---
        # 12a. Get Jira status
        res_jira = await ac.get("/api/integrations/jira", headers=headers)
        assert res_jira.status_code == 200

        # 12b. Connect Jira integration (mocking external Atlassian verification HTTP call)
        from unittest.mock import patch, AsyncMock
        from app.models import Integration

        mock_integration = Integration(
            agency_id="agc_e2e_001",
            provider="jira",
            is_connected=True,
            config={"base_url": "https://e2eagency.atlassian.net", "project_key": "INTEL", "email": "jira@e2eagency.com"},
        )
        with patch("app.services.jira.connect_jira", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_integration
            res_jira_up = await ac.post(
                "/api/integrations/jira/connect",
                headers=headers,
                json={
                    "base_url": "https://e2eagency.atlassian.net",
                    "email": "jira@e2eagency.com",
                    "api_token": "mock-token-12345",
                    "project_key": "INTEL",
                },
            )
            assert res_jira_up.status_code == 200
            assert res_jira_up.json()["project_key"] == "INTEL"

        # --- 13. SUPABASE CONFIG & INTELLIGENCE DATA ENDPOINTS ---
        # 13a. Supabase public config
        res_sb_cfg = await ac.get("/api/supabase/config")
        assert res_sb_cfg.status_code == 200
        assert "configured" in res_sb_cfg.json()

        # 13b. Supabase health status
        res_sb_st = await ac.get("/api/supabase/status")
        assert res_sb_st.status_code == 200

        # 13c. Client Trends, Sentiment, Insights & Snapshots
        res_tr = await ac.get(f"/api/clients/{client_id}/trends", headers=headers)
        assert res_tr.status_code == 200
        res_st = await ac.get(f"/api/clients/{client_id}/sentiment", headers=headers)
        assert res_st.status_code == 200
        res_in = await ac.get(f"/api/clients/{client_id}/insights", headers=headers)
        assert res_in.status_code == 200
        res_snp = await ac.get(f"/api/clients/{client_id}/snapshots", headers=headers)
        assert res_snp.status_code == 200

        # --- 14. CLEANUP & DELETE CLIENT ---
        # 14a. Delete competitor
        res_del_comp = await ac.delete(f"/api/clients/{client_id}/competitors/{comp_id}", headers=headers)
        assert res_del_comp.status_code == 200

        # 14b. Delete client (deactivates)
        res_del_client = await ac.delete(f"/api/clients/{client_id}", headers=headers)
        assert res_del_client.status_code == 200
        assert res_del_client.json()["ok"] is True
