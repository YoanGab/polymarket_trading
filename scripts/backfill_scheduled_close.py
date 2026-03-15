"""Backfill scheduled_close_ts for all markets WITHOUT a full re-scrape.

Two-phase approach:
1. Use event end_date from the already-backfilled events table (~99% coverage)
2. Fetch remaining markets from Gamma API individually (~1.5K markets)

Usage:
    uv run python scripts/backfill_scheduled_close.py [--db data/polymarket_backtest_v2.sqlite]
    uv run python scripts/backfill_scheduled_close.py --skip-api   # Phase 1 only
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
API_TIMEOUT = 20.0
API_DELAY = 0.03  # 30ms between requests
MAX_RETRIES = 3
BACKOFF_BASE = 0.5
BATCH_COMMIT_SIZE = 500


def phase1_from_events(conn: sqlite3.Connection) -> int:
    """Populate scheduled_close_ts from event end_date via market_events links.

    Returns:
        Number of markets updated.
    """
    # For each market, take the end_date from its linked event(s).
    # All events linked to a market have the same end_date (verified).
    result = conn.execute("""
        UPDATE markets
        SET scheduled_close_ts = (
            SELECT e.end_date
            FROM market_events me
            JOIN events e ON me.event_id = e.event_id
            WHERE me.market_id = markets.market_id
              AND e.end_date IS NOT NULL
            LIMIT 1
        )
        WHERE scheduled_close_ts IS NULL
          AND EXISTS (
            SELECT 1 FROM market_events me
            JOIN events e ON me.event_id = e.event_id
            WHERE me.market_id = markets.market_id
              AND e.end_date IS NOT NULL
          )
    """)
    conn.commit()
    return result.rowcount


def get_missing_market_ids(conn: sqlite3.Connection) -> list[str]:
    """Get market IDs that still have NULL scheduled_close_ts."""
    rows = conn.execute("SELECT market_id FROM markets WHERE scheduled_close_ts IS NULL").fetchall()
    return [r[0] for r in rows]


def fetch_end_date_from_gamma(client: httpx.Client, condition_id: str) -> str | None:
    """Fetch endDate for a single market from Gamma API."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(
                "/markets",
                params={"condition_ids": condition_id, "limit": "1"},
            )
            resp.raise_for_status()
            markets = resp.json()
            if markets and isinstance(markets, list):
                return markets[0].get("endDate") or markets[0].get("endDateIso")
            return None
        except (httpx.HTTPStatusError, httpx.RequestError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE * (2**attempt))
            else:
                return None
    return None


def phase2_from_api(conn: sqlite3.Connection) -> tuple[int, int]:
    """Fetch scheduled_close_ts from Gamma API for remaining markets.

    Returns:
        Tuple of (updated_count, not_found_count).
    """
    missing_ids = get_missing_market_ids(conn)
    if not missing_ids:
        return 0, 0

    client = httpx.Client(
        base_url=GAMMA_BASE,
        headers={
            "Accept": "application/json",
            "User-Agent": "polymarket-backtest/0.2",
        },
        follow_redirects=True,
        timeout=API_TIMEOUT,
    )

    updated = 0
    not_found = 0
    t0 = time.time()

    try:
        for i, market_id in enumerate(missing_ids):
            end_date = fetch_end_date_from_gamma(client, market_id)

            if end_date:
                conn.execute(
                    "UPDATE markets SET scheduled_close_ts = ? WHERE market_id = ?",
                    (end_date, market_id),
                )
                updated += 1
            else:
                not_found += 1

            if (i + 1) % BATCH_COMMIT_SIZE == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i + 1:,}/{len(missing_ids):,}] "
                    f"updated={updated:,} not_found={not_found:,} "
                    f"({rate:.0f} markets/s)"
                )

            time.sleep(API_DELAY)

        conn.commit()
    finally:
        client.close()

    return updated, not_found


def print_stats(conn: sqlite3.Connection) -> None:
    """Print final coverage statistics."""
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN scheduled_close_ts IS NOT NULL THEN 1 ELSE 0 END) as has_scheduled,
            SUM(CASE WHEN scheduled_close_ts IS NULL THEN 1 ELSE 0 END) as missing
        FROM markets
    """).fetchone()
    total, has_scheduled, missing = stats

    print(f"\n  Total markets: {total:,}")
    print(f"  With scheduled_close_ts: {has_scheduled:,} ({has_scheduled / total * 100:.1f}%)")
    print(f"  Still missing: {missing:,}")

    # Compare scheduled vs actual close
    comparison = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN DATE(scheduled_close_ts) >= DATE(close_ts) THEN 1 ELSE 0 END) as sched_after,
            SUM(CASE WHEN DATE(scheduled_close_ts) < DATE(close_ts) THEN 1 ELSE 0 END) as sched_before,
            SUM(CASE WHEN DATE(scheduled_close_ts) = DATE(close_ts) THEN 1 ELSE 0 END) as same_day
        FROM markets
        WHERE scheduled_close_ts IS NOT NULL AND close_ts IS NOT NULL
    """).fetchone()
    print("\n  Scheduled vs actual close (day-level):")
    print(f"    Same day: {comparison[3]:,}")
    print(f"    Scheduled >= actual: {comparison[1]:,}")
    print(f"    Scheduled < actual: {comparison[2]:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill scheduled_close_ts for markets")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip Phase 2 (API fetching), only use event data",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    total = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    already_filled = conn.execute("SELECT COUNT(*) FROM markets WHERE scheduled_close_ts IS NOT NULL").fetchone()[0]
    print(f"Markets: {total:,} total, {already_filled:,} already have scheduled_close_ts")

    # Phase 1: from events table
    print(f"\n{'=' * 60}")
    print("Phase 1: Populate from events table")
    print(f"{'=' * 60}")
    t0 = time.time()
    phase1_count = phase1_from_events(conn)
    elapsed = time.time() - t0
    print(f"  Updated {phase1_count:,} markets from event end_date ({elapsed:.1f}s)")
    print_stats(conn)

    # Phase 2: from Gamma API
    if not args.skip_api:
        remaining = conn.execute("SELECT COUNT(*) FROM markets WHERE scheduled_close_ts IS NULL").fetchone()[0]

        if remaining > 0:
            print(f"\n{'=' * 60}")
            print(f"Phase 2: Fetch {remaining:,} remaining from Gamma API")
            print(f"{'=' * 60}")
            t0 = time.time()
            api_updated, api_missing = phase2_from_api(conn)
            elapsed = time.time() - t0
            print(f"  Updated {api_updated:,} from API, {api_missing:,} not found ({elapsed:.1f}s)")
        else:
            print("\n  No remaining markets to fetch from API.")
    else:
        print("\n  Skipping Phase 2 (--skip-api)")

    # Final stats
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print_stats(conn)

    conn.close()


if __name__ == "__main__":
    main()
