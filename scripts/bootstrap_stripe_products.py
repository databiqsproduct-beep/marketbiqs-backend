#!/usr/bin/env python3
"""One-time: create MarketBiqs Stripe products/prices and print env lines."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import stripe

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SECRET = os.getenv("STRIPE_SECRET_KEY", "").strip()
if not SECRET.startswith(("sk_test_", "sk_live_")):
    print("STRIPE_SECRET_KEY missing or invalid in .env", file=sys.stderr)
    sys.exit(1)

stripe.api_key = SECRET

PLANS = [
    {
        "env": "STRIPE_AGENCY_PRICE_ID",
        "name": "MarketBiqs Agency",
        "description": "10 clients, 40 reports, 5,000 scrapes / month",
        "unit_amount": 45000,
        "lookup_key": "marketbiqs_agency_monthly",
    },
    {
        "env": "STRIPE_INDIVIDUAL_PRICE_ID",
        "name": "MarketBiqs Individual",
        "description": "1 client, 10 reports, 500 scrapes / month",
        "unit_amount": 9900,
        "lookup_key": "marketbiqs_individual_monthly",
    },
    {
        "env": "STRIPE_CLIENT_PACK_PRICE_ID",
        "name": "MarketBiqs Client Add-on Pack",
        "description": "+1 client, +8 reports, +800 scrapes / month",
        "unit_amount": 4900,
        "lookup_key": "marketbiqs_client_pack_monthly",
    },
    {
        "env": "STRIPE_SCRAPE_PACK_PRICE_ID",
        "name": "MarketBiqs Scrape Units (100)",
        "description": "+100 scrape units / month",
        "unit_amount": 500,
        "lookup_key": "marketbiqs_scrape_units_100_monthly_v5",
    },
    {
        "env": "STRIPE_PAYG_PRICE_ID",
        "name": "MarketBiqs Agency PAYG",
        "description": "Card on file. Billed monthly for clients, intel runs, reports, and scrape units used.",
        "unit_amount": 0,
        "lookup_key": "marketbiqs_agency_payg_monthly",
    },
]


def find_or_create_price(plan: dict) -> str:
    existing = stripe.Price.list(lookup_keys=[plan["lookup_key"]], limit=1)
    if existing.data:
        price = existing.data[0]
        print(f"exists  {plan['name']}: {price.id}")
        return price.id

    product = stripe.Product.create(
        name=plan["name"],
        description=plan["description"],
        metadata={"marketbiqs_plan": plan["lookup_key"]},
    )
    price = stripe.Price.create(
        product=product.id,
        unit_amount=plan["unit_amount"],
        currency="usd",
        recurring={"interval": "month"},
        lookup_key=plan["lookup_key"],
        metadata={"marketbiqs_plan": plan["lookup_key"]},
    )
    print(f"created {plan['name']}: product={product.id} price={price.id}")
    return price.id


def upsert_env(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text().splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            out.append(line)
            continue
        key, _, _ = line.partition("=")
        key = key.strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    # Insert missing keys after publishable key block if needed
    missing = [k for k in updates if k not in seen]
    if missing:
        inserted = False
        final: list[str] = []
        for line in out:
            final.append(line)
            if line.startswith("STRIPE_PUBLISHABLE_KEY="):
                for k in missing:
                    final.append(f"{k}={updates[k]}")
                    seen.add(k)
                inserted = True
                missing = [k for k in missing if k not in seen]
        if not inserted:
            final.extend(f"{k}={updates[k]}" for k in missing)
        out = final
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    updates: dict[str, str] = {}
    for plan in PLANS:
        updates[plan["env"]] = find_or_create_price(plan)

    env_path = ROOT / ".env"
    upsert_env(env_path, updates)
    print("\nUpdated .env:")
    for k, v in updates.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
