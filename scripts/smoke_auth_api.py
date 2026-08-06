"""Smoke + auth/API functional tests for Biqs after Supabase Auth merge."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = os.environ.get("API_BASE", "http://127.0.0.1:8000")
results: list[tuple[str, bool, str]] = []


def ok(name: str, passed: bool, detail: str = ""):
    results.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    line = f"[{mark}] {name}" + (f" -- {detail}" if detail else "")
    print(line.encode("ascii", "replace").decode("ascii"))


async def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from app.config import get_settings

    settings = get_settings()
    sb_url = (settings.supabase_url or "").rstrip("/")
    pub = settings.resolved_publishable_key()

    async with httpx.AsyncClient(base_url=API, timeout=60.0) as client:
        # Health
        r = await client.get("/health")
        ok("GET /health", r.status_code == 200, f"{r.status_code} {r.text[:80]}")

        r = await client.get("/health/ready")
        ok("GET /health/ready", r.status_code == 200, f"{r.status_code} {r.text[:120]}")

        # Legacy endpoints gone
        r = await client.post("/api/auth/login", json={"email": "a@b.com", "password": "x"})
        ok("POST /api/auth/login returns 410", r.status_code == 410, str(r.status_code))

        r = await client.post(
            "/api/auth/register",
            json={
                "email": "a@b.com",
                "password": "x",
                "full_name": "x",
                "agency_name": "x",
            },
        )
        ok("POST /api/auth/register returns 410", r.status_code == 410, str(r.status_code))

        # Unauthenticated protected routes
        r = await client.get("/api/auth/me")
        ok("GET /api/auth/me without token returns 401", r.status_code in (401, 403), str(r.status_code))

        r = await client.get("/api/clients")
        ok("GET /api/clients without token returns 401", r.status_code in (401, 403), str(r.status_code))

        # Supabase config endpoint if present
        r = await client.get("/api/supabase/config")
        ok(
            "GET /api/supabase/config",
            r.status_code in (200, 404),
            f"{r.status_code} {r.text[:100]}",
        )

        # Env alignment
        fe_url = "https://wcseztlbcajqegmzpzqm.supabase.co"
        ok(
            "Backend SUPABASE_URL matches frontend project",
            sb_url == fe_url,
            sb_url or "(empty)",
        )
        ok("Backend publishable key set", bool(pub), "missing" if not pub else "set")

        # JWKS
        try:
            jwks = await client.get(f"{sb_url}/auth/v1/.well-known/jwks.json")
            keys = jwks.json().get("keys", []) if jwks.status_code == 200 else []
            ok("Supabase JWKS reachable", jwks.status_code == 200 and len(keys) > 0, f"{jwks.status_code} keys={len(keys)}")
        except Exception as e:
            ok("Supabase JWKS reachable", False, str(e))

        # Full auth round-trip via Supabase Auth API
        if not (sb_url and pub):
            ok("Supabase signup+API me", False, "missing supabase config")
        else:
            email = f"biqs.test.{uuid.uuid4().hex[:10]}@mailinator.com"
            password = f"TestPass!{uuid.uuid4().hex[:8]}"
            secret = settings.resolved_secret_key()
            access = None

            # Prefer admin create (email confirmed) so local tests work with confirm-required projects
            if secret:
                admin_headers = {
                    "apikey": secret,
                    "Authorization": f"Bearer {secret}",
                    "Content-Type": "application/json",
                }
                created = await client.post(
                    f"{sb_url}/auth/v1/admin/users",
                    headers=admin_headers,
                    json={
                        "email": email,
                        "password": password,
                        "email_confirm": True,
                        "user_metadata": {"full_name": "Biqs Tester"},
                    },
                )
                ok(
                    "Supabase admin create user",
                    created.status_code in (200, 201),
                    f"{created.status_code} {created.text[:160]}",
                )
                if created.status_code in (200, 201):
                    login = await client.post(
                        f"{sb_url}/auth/v1/token?grant_type=password",
                        headers={
                            "apikey": pub,
                            "Authorization": f"Bearer {pub}",
                            "Content-Type": "application/json",
                        },
                        json={"email": email, "password": password},
                    )
                    login_body = login.json() if login.status_code == 200 else {}
                    access = login_body.get("access_token")
                    ok(
                        "Supabase password login",
                        login.status_code == 200 and bool(access),
                        f"{login.status_code} access={'yes' if access else 'no'} {str(login_body)[:160]}",
                    )
            else:
                headers = {
                    "apikey": pub,
                    "Authorization": f"Bearer {pub}",
                    "Content-Type": "application/json",
                }
                signup = await client.post(
                    f"{sb_url}/auth/v1/signup",
                    headers=headers,
                    json={"email": email, "password": password, "data": {"full_name": "Biqs Tester"}},
                )
                body = signup.json() if signup.headers.get("content-type", "").startswith("application/json") else {}
                access = body.get("access_token") or (body.get("session") or {}).get("access_token")
                ok(
                    "Supabase signup",
                    signup.status_code in (200, 201) and bool(access),
                    f"{signup.status_code} access={'yes' if access else 'no'} msg={str(body)[:160]}",
                )

            if access:
                # Invalid token alg / garbage
                bad = await client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-token"})
                ok("Invalid bearer rejected", bad.status_code in (401, 403), str(bad.status_code))

                me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {access}"})
                me_body = me.json() if me.status_code == 200 else {"raw": me.text[:200]}
                ok(
                    "GET /api/auth/me with Supabase JWT",
                    me.status_code == 200 and "user" in me_body,
                    f"{me.status_code} needs_bootstrap={me_body.get('needs_bootstrap')} agency={bool(me_body.get('agency'))}",
                )

                if me.status_code == 200 and me_body.get("needs_bootstrap"):
                    boot = await client.post(
                        "/api/auth/bootstrap",
                        headers={"Authorization": f"Bearer {access}"},
                        json={"agency_name": f"Test Agency {uuid.uuid4().hex[:6]}", "workspace_mode": "agency"},
                    )
                    boot_body = boot.json() if boot.status_code == 200 else {"raw": boot.text[:200]}
                    ok(
                        "POST /api/auth/bootstrap",
                        boot.status_code == 200 and boot_body.get("agency"),
                        f"{boot.status_code} {str(boot_body)[:160]}",
                    )
                    agency_id = (boot_body.get("agency") or {}).get("id")
                else:
                    agency_id = (me_body.get("agency") or {}).get("id")
                    ok("POST /api/auth/bootstrap", True, "skipped (already bootstrapped)")

                if agency_id:
                    auth_h = {
                        "Authorization": f"Bearer {access}",
                        "X-Agency-Id": agency_id,
                    }
                    # Dashboard
                    dash = await client.get("/api/agency/dashboard", headers=auth_h)
                    ok("GET /api/agency/dashboard", dash.status_code == 200, str(dash.status_code))

                    # Clients list
                    clients = await client.get("/api/clients", headers=auth_h)
                    ok("GET /api/clients", clients.status_code == 200, f"{clients.status_code} count={len(clients.json()) if clients.status_code==200 else '?'}")

                    # Create client
                    cname = f"TestCo {uuid.uuid4().hex[:6]}"
                    create = await client.post(
                        "/api/clients",
                        headers=auth_h,
                        json={"name": cname, "website": "https://example.com", "industry": "SaaS"},
                    )
                    created = create.json() if create.status_code < 300 else {"raw": create.text[:200]}
                    cid = created.get("id")
                    ok("POST /api/clients", create.status_code in (200, 201) and bool(cid), f"{create.status_code} {str(created)[:160]}")

                    if cid:
                        one = await client.get(f"/api/clients/{cid}", headers=auth_h)
                        ok("GET /api/clients/{id}", one.status_code == 200, str(one.status_code))

                        feats = await client.get(f"/api/clients/{cid}/features", headers=auth_h)
                        ok("GET /api/clients/{id}/features", feats.status_code == 200, str(feats.status_code))

                        comps = await client.get(f"/api/clients/{cid}/competitors", headers=auth_h)
                        ok("GET /api/clients/{id}/competitors", comps.status_code == 200, str(comps.status_code))

                        gaps = await client.get(f"/api/clients/{cid}/gaps", headers=auth_h)
                        ok("GET /api/clients/{id}/gaps", gaps.status_code == 200, str(gaps.status_code))

                        alerts = await client.get(f"/api/clients/{cid}/alerts", headers=auth_h)
                        ok("GET /api/clients/{id}/alerts", alerts.status_code == 200, str(alerts.status_code))

                        reports = await client.get(f"/api/clients/{cid}/reports", headers=auth_h)
                        ok("GET /api/clients/{id}/reports", reports.status_code == 200, str(reports.status_code))

                        wish = await client.get(f"/api/clients/{cid}/wishlist", headers=auth_h)
                        ok("GET /api/clients/{id}/wishlist", wish.status_code == 200, str(wish.status_code))

                        members = await client.get("/api/agency/members", headers=auth_h)
                        ok("GET /api/agency/members", members.status_code == 200, str(members.status_code))

                        budget = await client.get("/api/billing/budget", headers=auth_h)
                        ok("GET /api/billing/budget", budget.status_code in (200, 404), str(budget.status_code))

                        # Idempotent bootstrap
                        boot2 = await client.post(
                            "/api/auth/bootstrap",
                            headers={"Authorization": f"Bearer {access}"},
                            json={"agency_name": "Should Not Duplicate"},
                        )
                        ok(
                            "Bootstrap idempotent",
                            boot2.status_code == 200 and (boot2.json().get("agency") or {}).get("id") == agency_id,
                            f"{boot2.status_code}",
                        )

    print("\n=== SUMMARY ===")
    failed = [r for r in results if not r[1]]
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("Failures:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
