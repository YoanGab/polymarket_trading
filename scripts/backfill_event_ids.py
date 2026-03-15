"""Backfill event_id on the markets table.

Phase 1: SQL join from market_events (instant, covers ~99% of markets).
Phase 2: For remaining NULLs, fetch from Gamma API in batches.

Usage:
    uv run python scripts/backfill_event_ids.py [--db data/polymarket_backtest_v2.sqlite]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import httpx

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"
GAMMA_BASE = "https://gamma-api.polymarket.com"
BATCH_DELAY = 0.1  # 100ms between API calls


def phase1_sql_backfill(conn: sqlite3.Connection) -> int:
    """Backfill event_id from market_events table. Returns rows updated."""
    cursor = conn.execute("""
        UPDATE markets
        SET event_id = (
            SELECT me.event_id
            FROM market_events me
            WHERE me.market_id = markets.market_id
            LIMIT 1
        )
        WHERE event_id IS NULL
          AND EXISTS (
              SELECT 1 FROM market_events me
              WHERE me.market_id = markets.market_id
          )
    """)
    conn.commit()
    return cursor.rowcount


def phase2_api_backfill(conn: sqlite3.Connection) -> tuple[int, int]:
    """Fetch event_id from Gamma API for remaining NULL markets.

    Returns:
        (updated, failed) counts.
    """
    null_ids = [row[0] for row in conn.execute("SELECT market_id FROM markets WHERE event_id IS NULL").fetchall()]

    if not null_ids:
        return 0, 0

    print(f"\nPhase 2: {len(null_ids)} markets still missing event_id, fetching from API...")

    updated = 0
    failed = 0

    client = httpx.Client(
        headers={"Accept": "application/json", "User-Agent": "polymarket-backtest/0.2"},
        timeout=30.0,
    )

    for i, market_id in enumerate(null_ids):
        try:
            # Gamma API uses condition_id to look up markets
            resp = client.get(
                f"{GAMMA_BASE}/markets",
                params={"condition_id": market_id},
            )
            resp.raise_for_status()
            data = resp.json()

            event_id = None
            if isinstance(data, list) and data:
                event_id = data[0].get("events", [{}])[0].get("id") if data[0].get("events") else None
                if event_id is None:
                    # Try top-level event_id
                    event_id = data[0].get("event_id")
            elif isinstance(data, dict):
                event_id = data.get("events", [{}])[0].get("id") if data.get("events") else None
                if event_id is None:
                    event_id = data.get("event_id")

            if event_id:
                conn.execute(
                    "UPDATE markets SET event_id = ? WHERE market_id = ?",
                    (str(event_id), market_id),
                )
                updated += 1

                # Also insert into market_events for consistency
                conn.execute(
                    "INSERT OR IGNORE INTO market_events (market_id, event_id) VALUES (?, ?)",
                    (market_id, str(event_id)),
                )
            else:
                failed += 1

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            print(f"  API error for {market_id[:20]}...: {e}")
            failed += 1

        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  [{i + 1}/{len(null_ids)}] updated={updated}, failed={failed}")

        time.sleep(BATCH_DELAY)

    conn.commit()
    client.close()
    return updated, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill event_id on markets table")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip Phase 2 (API calls), only do SQL backfill",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    # Pre-check
    total = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    null_before = conn.execute("SELECT COUNT(*) FROM markets WHERE event_id IS NULL").fetchone()[0]
    print(f"Markets: {total:,}")
    print(f"event_id IS NULL: {null_before:,}")

    if null_before == 0:
        print("Nothing to backfill -- all markets already have event_id.")
        conn.close()
        return

    # Phase 1: SQL join
    t0 = time.time()
    sql_updated = phase1_sql_backfill(conn)
    t1 = time.time()
    print(f"\nPhase 1 (SQL backfill from market_events): {sql_updated:,} rows updated in {t1 - t0:.2f}s")

    remaining = conn.execute("SELECT COUNT(*) FROM markets WHERE event_id IS NULL").fetchone()[0]
    print(f"Remaining NULL: {remaining:,}")

    # Phase 2: API
    api_updated = 0
    api_failed = 0
    if remaining > 0 and not args.skip_api:
        api_updated, api_failed = phase2_api_backfill(conn)
        t2 = time.time()
        print(f"\nPhase 2 (API backfill): {api_updated:,} updated, {api_failed:,} failed in {t2 - t1:.2f}s")

    # Final stats
    final_null = conn.execute("SELECT COUNT(*) FROM markets WHERE event_id IS NULL").fetchone()[0]
    final_set = total - final_null
    print(f"\n{'=' * 50}")
    print("BACKFILL COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Total markets: {total:,}")
    print(f"  event_id set:  {final_set:,} ({final_set / total * 100:.1f}%)")
    print(f"  event_id NULL: {final_null:,} ({final_null / total * 100:.1f}%)")
    print(f"  Phase 1 (SQL): {sql_updated:,}")
    print(f"  Phase 2 (API): {api_updated:,}")

    conn.close()


if __name__ == "__main__":
    main()
