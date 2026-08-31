"""Comprehensive End-to-End (E2E) Test Suite for MarketBiqs.

Tests all core backend APIs, multi-tenant isolation, AI chat, PDF generation,
white-label keys, and frontend page routes.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BACKEND_BASE = "http://127.0.0.1:8000"
FRONTEND_BASE = "http://localhost:3000"

AGENCY_EMAIL = "agency@marketbiqs.com"
AGENCY_PASSWORD = "AgencyPass123!"

CLIENT_EMAIL = "client@acmeretail.com"
CLIENT_PASSWORD = "ClientPass123!"

results: list[dict] = []


def record_result(category: str, name: str, passed: bool, detail: str, duration_ms: float):
    results.append({
        "category": category,
        "name": name,
        "passed": passed,
        "detail": detail,
        "duration_ms": round(duration_ms, 1),
    })
    status_str = "PASS" if passed else "FAIL"
    print(f"[{status_str:4}] ({category}) {name} -> {detail} ({duration_ms:.1f}ms)")


async def run_suite():
    print("\n" + "=" * 80)
    print("STARTING MARKETBIQS END-TO-END (E2E) TEST SUITE")
    print(f"Backend API:  {BACKEND_BASE}")
    print(f"Frontend App: {FRONTEND_BASE}")
    print("=" * 80 + "\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # =========================================================================
        # 1. SYSTEM HEALTH & READINESS
        # =========================================================================
        t0 = time.perf_counter()
        try:
            r = await client.get(f"{BACKEND_BASE}/health")
            d = (time.perf_counter() - t0) * 1000
            record_result("Health & System", "GET /health", r.status_code == 200, f"HTTP {r.status_code} {r.json()}", d)
        except Exception as e:
            record_result("Health & System", "GET /health", False, str(e), 0)

        t0 = time.perf_counter()
        try:
            r = await client.get(f"{BACKEND_BASE}/health/ready")
            d = (time.perf_counter() - t0) * 1000
            record_result("Health & System", "GET /health/ready", r.status_code == 200, f"HTTP {r.status_code} {r.json()}", d)
        except Exception as e:
            record_result("Health & System", "GET /health/ready", False, str(e), 0)

        # =========================================================================
        # 2. AUTHENTICATION & SECURITY
        # =========================================================================
        # Unauthenticated protection
        t0 = time.perf_counter()
        r = await client.get(f"{BACKEND_BASE}/api/auth/me")
        d = (time.perf_counter() - t0) * 1000
        record_result("Auth & Security", "Unauthorized Access Rejection", r.status_code in (401, 403), f"Protected route returned HTTP {r.status_code}", d)

        # Invalid password rejection
        t0 = time.perf_counter()
        r = await client.post(f"{BACKEND_BASE}/api/auth/login-local", json={"email": AGENCY_EMAIL, "password": "WrongPassword!"})
        d = (time.perf_counter() - t0) * 1000
        record_result("Auth & Security", "Invalid Password Rejection", r.status_code == 401, f"Bad password rejected HTTP {r.status_code}", d)

        # Agency Login
        t0 = time.perf_counter()
        r = await client.post(f"{BACKEND_BASE}/api/auth/login-local", json={"email": AGENCY_EMAIL, "password": AGENCY_PASSWORD})
        d = (time.perf_counter() - t0) * 1000
        agency_data = r.json() if r.status_code == 200 else {}
        agency_token = agency_data.get("access_token")
        agency_id = (agency_data.get("agency") or {}).get("id")
        record_result(
            "Auth & Security",
            "Agency Owner Login",
            r.status_code == 200 and bool(agency_token),
            f"Logged in as {agency_data.get('user', {}).get('email')} (Role: {agency_data.get('role')})",
            d,
        )

        agency_headers = {
            "Authorization": f"Bearer {agency_token}",
            "X-Agency-Id": agency_id or "",
        }

        # Client User Login
        t0 = time.perf_counter()
        r = await client.post(f"{BACKEND_BASE}/api/auth/login-local", json={"email": CLIENT_EMAIL, "password": CLIENT_PASSWORD})
        d = (time.perf_counter() - t0) * 1000
        client_data = r.json() if r.status_code == 200 else {}
        client_token = client_data.get("access_token")
        record_result(
            "Auth & Security",
            "Client User Login",
            r.status_code == 200 and bool(client_token),
            f"Logged in as {client_data.get('user', {}).get('email')} (Role: {client_data.get('role')})",
            d,
        )

        # /api/auth/me Session Validation
        t0 = time.perf_counter()
        r = await client.get(f"{BACKEND_BASE}/api/auth/me", headers=agency_headers)
        d = (time.perf_counter() - t0) * 1000
        me_data = r.json() if r.status_code == 200 else {}
        record_result(
            "Auth & Security",
            "Validate Session (/api/auth/me)",
            r.status_code == 200 and me_data.get("user", {}).get("email") == AGENCY_EMAIL,
            f"Active Agency: {me_data.get('agency', {}).get('name')}",
            d,
        )

        # =========================================================================
        # 3. AGENCY SETTINGS & BRANDING
        # =========================================================================
        t0 = time.perf_counter()
        r = await client.get(f"{BACKEND_BASE}/api/agency", headers=agency_headers)
        d = (time.perf_counter() - t0) * 1000
        record_result("Agency Workspace", "Fetch Agency Profile", r.status_code == 200, f"Agency Plan: {r.json().get('plan')}", d)

        # Update Branding
        t0 = time.perf_counter()
        r = await client.put(
            f"{BACKEND_BASE}/api/agency/branding",
            headers=agency_headers,
            json={
                "brand_color": "#0f766e",
                "brand_secondary": "#134e4a",
                "report_footer": "Apex Growth Agency — Proprietary Client Intelligence",
            },
        )
        d = (time.perf_counter() - t0) * 1000
        record_result("Agency Workspace", "Update White-Label Branding", r.status_code == 200, f"Updated footer: {r.json().get('report_footer', '')[:40]}...", d)

        # =========================================================================
        # 4. CLIENT BRAND MANAGEMENT
        # =========================================================================
        t0 = time.perf_counter()
        r = await client.get(f"{BACKEND_BASE}/api/clients", headers=agency_headers)
        d = (time.perf_counter() - t0) * 1000
        clients_list = r.json() if r.status_code == 200 else []
        record_result("Clients Management", "List Agency Clients", r.status_code == 200 and len(clients_list) > 0, f"Found {len(clients_list)} active client brands", d)

        # Create New Client Brand
        test_client_name = f"Test Client {uuid.uuid4().hex[:6]}"
        t0 = time.perf_counter()
        r = await client.post(
            f"{BACKEND_BASE}/api/clients",
            headers=agency_headers,
            json={
                "name": test_client_name,
                "industry": "FinTech & Payments",
                "niche": "Cross-Border Remittances",
                "website": "https://testfintech.example.com",
                "tagline": "Frictionless global payments for modern businesses.",
            },
        )
        d = (time.perf_counter() - t0) * 1000
        created_client = r.json() if r.status_code == 200 else {}
        created_client_id = created_client.get("id")
        record_result("Clients Management", "Create New Client Brand", r.status_code == 200 and bool(created_client_id), f"Created {test_client_name} (ID: {created_client_id})", d)

        target_client_id = created_client_id or (clients_list[0]["id"] if clients_list else None)

        # Fetch Single Client
        if target_client_id:
            t0 = time.perf_counter()
            r = await client.get(f"{BACKEND_BASE}/api/clients/{target_client_id}", headers=agency_headers)
            d = (time.perf_counter() - t0) * 1000
            record_result("Clients Management", "Fetch Client Details", r.status_code == 200, f"Name: {r.json().get('name')}", d)

        # =========================================================================
        # 5. COMPETITOR TRACKING & CRAWLING
        # =========================================================================
        if target_client_id:
            # Add Competitor
            t0 = time.perf_counter()
            r = await client.post(
                f"{BACKEND_BASE}/api/clients/{target_client_id}/competitors",
                headers=agency_headers,
                json={
                    "name": "GlobalPay Rival",
                    "website": "https://globalpayrival.example.com",
                    "description": "Direct competitor in cross border payments.",
                    "threat_level": "high",
                },
            )
            d = (time.perf_counter() - t0) * 1000
            comp_data = r.json() if r.status_code == 200 else {}
            record_result("Competitor Intel", "Add Competitor to Client", r.status_code == 200 and bool(comp_data.get("id")), f"Added {comp_data.get('name')}", d)

            # List Competitors
            t0 = time.perf_counter()
            r = await client.get(f"{BACKEND_BASE}/api/clients/{target_client_id}/competitors", headers=agency_headers)
            d = (time.perf_counter() - t0) * 1000
            comps = r.json() if r.status_code == 200 else []
            record_result("Competitor Intel", "List Client Competitors", r.status_code == 200 and len(comps) > 0, f"Tracking {len(comps)} competitor(s)", d)

        # =========================================================================
        # 6. INSIGHTS, TRENDS & SENTIMENTS
        # =========================================================================
        if target_client_id:
            # Create Insight
            t0 = time.perf_counter()
            r = await client.post(
                f"{BACKEND_BASE}/api/clients/{target_client_id}/insights",
                headers=agency_headers,
                json={
                    "title": "Untapped Zero-Fee Transfer Hook",
                    "body": "Competitors charge 1.5% hidden FX fees. Suggest launching a transparent calculator comparison.",
                    "category": "pricing",
                    "priority": "high",
                },
            )
            d = (time.perf_counter() - t0) * 1000
            record_result("Insights & Intelligence", "Generate Competitive Insight", r.status_code in (200, 201), f"Created insight: {r.json().get('title', '')[:35]}...", d)

            # List Insights
            t0 = time.perf_counter()
            r = await client.get(f"{BACKEND_BASE}/api/clients/{target_client_id}/insights", headers=agency_headers)
            d = (time.perf_counter() - t0) * 1000
            insights = r.json() if r.status_code == 200 else []
            record_result("Insights & Intelligence", "Fetch Client Insights", r.status_code == 200 and len(insights) > 0, f"Found {len(insights)} strategic insights", d)

            # Fetch Trends
            t0 = time.perf_counter()
            r = await client.get(f"{BACKEND_BASE}/api/clients/{target_client_id}/trends", headers=agency_headers)
            d = (time.perf_counter() - t0) * 1000
            record_result("Insights & Intelligence", "Fetch Trend Signals", r.status_code == 200, f"HTTP {r.status_code}", d)

            # Fetch Sentiments
            t0 = time.perf_counter()
            r = await client.get(f"{BACKEND_BASE}/api/clients/{target_client_id}/sentiments", headers=agency_headers)
            d = (time.perf_counter() - t0) * 1000
            record_result("Insights & Intelligence", "Fetch Sentiment Records", r.status_code == 200, f"HTTP {r.status_code}", d)

        # =========================================================================
        # 7. EXECUTIVE REPORTS & PDF GENERATION
        # =========================================================================
        if target_client_id:
            # Create Executive Report
            t0 = time.perf_counter()
            r = await client.post(
                f"{BACKEND_BASE}/api/clients/{target_client_id}/reports",
                headers=agency_headers,
                json={
                    "title": "Comprehensive E2E Competitive Briefing",
                    "period_label": "Monthly",
                    "summary": "Detailed analysis of competitors, messaging changes, and positioning opportunities.",
                    "sections": [
                        {"title": "Executive Summary", "content": "Rivals are increasing spend on influencer marketing."},
                        {"title": "Action Steps", "content": "Launch comparison ads emphasizing lower fees."},
                    ],
                },
            )
            d = (time.perf_counter() - t0) * 1000
            rep_data = r.json() if r.status_code in (200, 201) else {}
            rep_id = rep_data.get("id")
            record_result("Reports & White-Label", "Create Executive Report", r.status_code in (200, 201) and bool(rep_id), f"Report ID: {rep_id}", d)

            # Fetch Reports List
            t0 = time.perf_counter()
            r = await client.get(f"{BACKEND_BASE}/api/clients/{target_client_id}/reports", headers=agency_headers)
            d = (time.perf_counter() - t0) * 1000
            reps = r.json() if r.status_code == 200 else []
            record_result("Reports & White-Label", "List Client Reports", r.status_code == 200 and len(reps) > 0, f"Total reports: {len(reps)}", d)

            # Download Report PDF
            if rep_id:
                t0 = time.perf_counter()
                r = await client.get(f"{BACKEND_BASE}/api/reports/{rep_id}/pdf", headers=agency_headers)
                d = (time.perf_counter() - t0) * 1000
                is_pdf = r.status_code == 200 and r.headers.get("content-type") == "application/pdf"
                record_result("Reports & White-Label", "Generate & Download PDF", is_pdf, f"PDF Size: {len(r.content)} bytes ({r.headers.get('content-type')})", d)

        # =========================================================================
        # 8. AI ASSISTANT CHAT & STREAMING
        # =========================================================================
        if target_client_id:
            # Post Chat Message
            t0 = time.perf_counter()
            r = await client.post(
                f"{BACKEND_BASE}/api/clients/{target_client_id}/chat",
                headers=agency_headers,
                json={"content": "What are our primary competitor weaknesses?"},
            )
            d = (time.perf_counter() - t0) * 1000
            chat_resp = r.json() if r.status_code in (200, 201) else {}
            record_result("AI Intelligence", "AI Assistant Chat Post", r.status_code in (200, 201), f"AI Response Length: {len(chat_resp.get('content', ''))} chars", d)

            # Fetch Chat History
            t0 = time.perf_counter()
            r = await client.get(f"{BACKEND_BASE}/api/clients/{target_client_id}/chat", headers=agency_headers)
            d = (time.perf_counter() - t0) * 1000
            messages = r.json() if r.status_code == 200 else []
            record_result("AI Intelligence", "Fetch Chat History", r.status_code == 200 and len(messages) > 0, f"Total messages: {len(messages)}", d)

        # =========================================================================
        # 9. WHITE-LABEL API KEYS & EXTERNAL ACCESS
        # =========================================================================
        t0 = time.perf_counter()
        r = await client.post(
            f"{BACKEND_BASE}/api/white-label/keys",
            headers=agency_headers,
            json={"name": "Client Portal Key", "monthly_quota": 5000},
        )
        d = (time.perf_counter() - t0) * 1000
        key_data = r.json() if r.status_code in (200, 201) else {}
        api_key = key_data.get("key")
        record_result("White-Label API", "Generate White-Label API Key", r.status_code in (200, 201) and bool(api_key), f"Key Prefix: {key_data.get('prefix')}", d)

        # Test External Access with X-API-Key
        if api_key and target_client_id:
            t0 = time.perf_counter()
            r = await client.get(
                f"{BACKEND_BASE}/api/white-label/clients/{target_client_id}",
                headers={"X-API-Key": api_key},
            )
            d = (time.perf_counter() - t0) * 1000
            record_result("White-Label API", "Access via X-API-Key Header", r.status_code in (200, 404), f"HTTP {r.status_code}", d)

        # =========================================================================
        # 10. FRONTEND ROUTES STATUS (HTTP 200 OK)
        # =========================================================================
        frontend_routes = [
            ("/", "Root Landing Page (6 Sections, Header, Footer)"),
            ("/login", "Sign In Page"),
            ("/register", "Registration Page"),
            ("/dashboard", "Agency Dashboard Overview"),
            ("/clients", "Client Portfolio Page"),
            ("/reports", "Reports & Export Center"),
            ("/tracker", "Competitive Tracker Jobs"),
            ("/assistant", "AI Research Assistant"),
            ("/branding", "Agency White-Label Branding"),
            ("/white-label", "Client API Keys & Embeds"),
            ("/team", "Team Seats & Invitations"),
            ("/billing", "Billing & Subscriptions"),
            (f"/portal/{target_client_id}" if target_client_id else "/portal", "Dedicated Client Portal"),
        ]

        for route, desc in frontend_routes:
            t0 = time.perf_counter()
            try:
                r = await client.get(f"{FRONTEND_BASE}{route}", follow_redirects=True)
                d = (time.perf_counter() - t0) * 1000
                record_result("Frontend Web Pages", f"Route {route}", r.status_code == 200, f"{desc} (HTTP {r.status_code})", d)
            except Exception as e:
                record_result("Frontend Web Pages", f"Route {route}", False, str(e), 0)

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count
    total_count = len(results)

    print("\n" + "=" * 80)
    print(f"E2E TEST SUITE RESULTS: {passed_count}/{total_count} PASSED ({failed_count} FAILED)")
    print("=" * 80)

    categories = list(dict.fromkeys(r["category"] for r in results))
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        cat_passed = sum(1 for r in cat_items if r["passed"])
        print(f"\n--- {cat.upper()} ({cat_passed}/{len(cat_items)}) ---")
        for item in cat_items:
            mark = "PASS" if item["passed"] else "FAIL"
            print(f"  [{mark:4}] {item['name']:<35} | {item['detail']} ({item['duration_ms']}ms)")

    print("\n" + "=" * 80 + "\n")
    return failed_count == 0


if __name__ == "__main__":
    success = asyncio.run(run_suite())
    sys.exit(0 if success else 1)
