"""Backfill event data and tags from Gamma /events endpoint.

Adds 3 tables to polymarket_backtest_v2.sqlite:
- events: event metadata (id, slug, title, category, dates, volume)
- event_tags: event_id → tag (normalized, many-to-many)
- market_events: market_id (condition_id) → event_id (link table)

Does NOT modify existing tables. Safe to run multiple times (uses INSERT OR IGNORE).

Usage:
    uv run python scripts/backfill_events.py [--db data/polymarket_backtest_v2.sqlite]
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
PAGE_SIZE = 100
PAGE_DELAY = 0.05  # 50ms between pages


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            slug TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT,
            start_date TEXT,
            end_date TEXT,
            volume REAL NOT NULL DEFAULT 0,
            liquidity REAL NOT NULL DEFAULT 0,
            comment_count INTEGER NOT NULL DEFAULT 0,
            closed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS event_tags (
            event_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (event_id, tag),
            FOREIGN KEY (event_id) REFERENCES events (event_id)
        );

        CREATE TABLE IF NOT EXISTS market_events (
            market_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            PRIMARY KEY (market_id, event_id),
            FOREIGN KEY (market_id) REFERENCES markets (market_id),
            FOREIGN KEY (event_id) REFERENCES events (event_id)
        );

        CREATE INDEX IF NOT EXISTS idx_event_tags_tag ON event_tags (tag);
        CREATE INDEX IF NOT EXISTS idx_market_events_event ON market_events (event_id);
    """)


def fetch_events_page(client: httpx.Client, offset: int) -> list[dict]:
    resp = client.get(
        f"{GAMMA_BASE}/events",
        params={"limit": PAGE_SIZE, "offset": offset, "closed": "true"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def ingest_event(conn: sqlite3.Connection, event: dict) -> tuple[int, int]:
    """Ingest one event. Returns (n_tags, n_market_links)."""
    event_id = str(event["id"])

    conn.execute(
        """INSERT OR IGNORE INTO events
           (event_id, slug, title, category, start_date, end_date,
            volume, liquidity, comment_count, closed, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event_id,
            event.get("slug", ""),
            event.get("title", ""),
            event.get("category"),
            event.get("startDate"),
            event.get("endDate"),
            event.get("volume", 0) or 0,
            event.get("liquidity", 0) or 0,
            event.get("commentCount", 0) or 0,
            1 if event.get("closed") else 0,
            event.get("createdAt"),
            event.get("updatedAt"),
        ),
    )

    # Tags
    n_tags = 0
    for tag_obj in event.get("tags") or []:
        label = tag_obj.get("label", "").strip()
        if label and label.lower() != "all":  # Skip generic "All" tag
            conn.execute(
                "INSERT OR IGNORE INTO event_tags (event_id, tag) VALUES (?, ?)",
                (event_id, label),
            )
            n_tags += 1

    # Market links
    n_links = 0
    for market in event.get("markets") or []:
        condition_id = market.get("conditionId")
        if condition_id:
            conn.execute(
                "INSERT OR IGNORE INTO market_events (market_id, event_id) VALUES (?, ?)",
                (condition_id, event_id),
            )
            n_links += 1

    return n_tags, n_links


def main():
    parser = argparse.ArgumentParser(description="Backfill events and tags")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    create_tables(conn)
    conn.commit()

    # Count existing markets for matching stats
    total_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    print(f"Markets in DB: {total_markets:,}")

    client = httpx.Client(headers={"Accept": "application/json", "User-Agent": "polymarket-backtest/0.2"})

    offset = 0
    total_events = 0
    total_tags = 0
    total_links = 0
    empty_pages = 0
    t0 = time.time()

    print(f"\nFetching events from {GAMMA_BASE}/events ...")

    while True:
        try:
            events = fetch_events_page(client, offset)
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            print(f"  ERROR at offset {offset}: {e}")
            time.sleep(2)
            continue

        if not events:
            empty_pages += 1
            if empty_pages >= 3:
                break
            offset += PAGE_SIZE
            continue

        empty_pages = 0

        for event in events:
            try:
                n_tags, n_links = ingest_event(conn, event)
                total_events += 1
                total_tags += n_tags
                total_links += n_links
            except Exception as e:
                print(f"  INGEST ERROR event {event.get('id')}: {e}")

        if total_events % 500 == 0 or total_events < 100:
            conn.commit()
            elapsed = time.time() - t0
            rate = total_events / elapsed if elapsed > 0 else 0
            print(f"  [{total_events:,} events] {total_tags:,} tags, {total_links:,} market links, {rate:.0f} events/s")

        offset += PAGE_SIZE
        time.sleep(PAGE_DELAY)

    conn.commit()

    # Final stats
    elapsed = time.time() - t0
    linked = conn.execute("SELECT COUNT(DISTINCT market_id) FROM market_events").fetchone()[0]
    unlinked = total_markets - linked
    unique_tags = conn.execute("SELECT COUNT(DISTINCT tag) FROM event_tags").fetchone()[0]

    # Top tags
    top_tags = conn.execute("""
        SELECT et.tag, COUNT(DISTINCT me.market_id) as market_count
        FROM event_tags et
        JOIN market_events me ON et.event_id = me.event_id
        GROUP BY et.tag ORDER BY market_count DESC LIMIT 20
    """).fetchall()

    print(f"\n{'=' * 60}")
    print(f"BACKFILL COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  Events fetched: {total_events:,}")
    print(f"  Total tags: {total_tags:,}")
    print(f"  Unique tags: {unique_tags}")
    print(f"  Market-event links: {total_links:,}")
    print(f"  Markets with events: {linked:,} / {total_markets:,} ({linked / total_markets * 100:.1f}%)")
    print(f"  Markets without events: {unlinked:,}")
    print(f"\nTop 20 tags by market count:")
    for tag, count in top_tags:
        print(f"  {tag:30s} {count:6,} markets")

    conn.close()
    client.close()


if __name__ == "__main__":
    main()
