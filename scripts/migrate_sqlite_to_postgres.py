"""Copy MarketBiqs data from local SQLite biqs.db into Supabase Postgres.

Usage (from backend/):
  1) Set DATABASE_URL or SUPABASE_DB_PASSWORD in .env to the Postgres target
  2) python -m scripts.migrate_sqlite_to_postgres
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from sqlalchemy import text

from app.database import DATABASE_BACKEND, DATABASE_URL, engine, init_db

SQLITE_PATH = Path(__file__).resolve().parents[1] / "biqs.db"


async def migrate() -> None:
    if DATABASE_BACKEND != "postgres":
        raise SystemExit(
            "Target DATABASE_URL must be Supabase Postgres. "
            f"Current backend={DATABASE_BACKEND} url={DATABASE_URL[:48]}..."
        )
    if not SQLITE_PATH.exists():
        raise SystemExit(f"Missing SQLite source at {SQLITE_PATH}")

    await init_db()
    src = sqlite3.connect(SQLITE_PATH)
    src.row_factory = sqlite3.Row
    tables = [
        r[0]
        for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]

    async with engine.begin() as conn:
        for table in tables:
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                print(f"skip empty {table}")
                continue
            cols = list(rows[0].keys())
            placeholders = ", ".join(f":{c}" for c in cols)
            col_sql = ", ".join(cols)
            inserted = 0
            for row in rows:
                payload = {c: row[c] for c in cols}
                try:
                    await conn.execute(
                        text(
                            f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
                            f"ON CONFLICT DO NOTHING"
                        ),
                        payload,
                    )
                    inserted += 1
                except Exception as exc:
                    print(f"warn {table}: {exc}")
            print(f"{table}: attempted {len(rows)}, wrote ~{inserted}")

    src.close()
    print("Migration finished.")


if __name__ == "__main__":
    asyncio.run(migrate())
