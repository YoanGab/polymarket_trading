"""Polymarket orderbook depth scraper.

Fetches orderbook snapshots for top active markets from the CLOB API
and stores them in a SQLite database. Designed to run as a cron job
every minute.

Usage:
    uv run python scripts/scrape_orderbook.py          # one-shot scrape
    uv run python scripts/scrape_orderbook.py --limit 60
    uv run python scripts/scrape_orderbook.py --loop    # continuous (every 60s)
    uv run python scripts/scrape_orderbook.py --stats   # print stored stats

Cron example (every minute):
    * * * * * cd /path/to/trading_bot && uv run python scripts/scrape_orderbook.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "orderbook_snapshots.db"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
TOP_N_MARKETS = 10
HTTP_TIMEOUT = 15.0


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            market_slug TEXT NOT NULL,
            question TEXT NOT NULL,
            token_id TEXT NOT NULL,
            side TEXT NOT NULL,  -- 'YES' or 'NO'
            bid_levels INTEGER NOT NULL,
            ask_levels INTEGER NOT NULL,
            best_bid REAL,
            best_ask REAL,
            spread REAL,
            total_bid_shares REAL NOT NULL,
            total_ask_shares REAL NOT NULL,
            total_bid_dollar REAL NOT NULL,
            total_ask_dollar REAL NOT NULL,
            last_trade_price REAL,
            tick_size REAL,
            raw_bids TEXT,  -- JSON array of {price, size}
            raw_asks TEXT   -- JSON array of {price, size}
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(ts);
        CREATE INDEX IF NOT EXISTS idx_snapshots_market ON snapshots(market_slug, ts);

        CREATE TABLE IF NOT EXISTS markets (
            token_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            slug TEXT NOT NULL,
            side TEXT NOT NULL,
            volume_24h REAL,
            liquidity REAL,
            last_updated TEXT
        );
    """)
    conn.commit()
    return conn


def get_top_markets(client: httpx.Client, n: int | None = None) -> list[dict]:
    """Fetch top N active markets by 24h volume from Gamma API."""
    if n is None:
        n = TOP_N_MARKETS
    resp = client.get(
        f"{GAMMA_API}/markets",
        params={
            "limit": n,
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
        },
    )
    resp.raise_for_status()
    return resp.json()


def fetch_orderbook(client: httpx.Client, token_id: str) -> dict:
    """Fetch full orderbook for a token from the CLOB API."""
    resp = client.get(f"{CLOB_API}/book", params={"token_id": token_id})
    resp.raise_for_status()
    return resp.json()


def analyze_book(book: dict) -> dict:
    """Extract depth metrics from an orderbook response."""
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    bids_sorted = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
    asks_sorted = sorted(asks, key=lambda x: float(x["price"]))

    total_bid_shares = sum(float(b["size"]) for b in bids)
    total_ask_shares = sum(float(a["size"]) for a in asks)
    total_bid_dollar = sum(float(b["price"]) * float(b["size"]) for b in bids)
    total_ask_dollar = sum(float(a["price"]) * float(a["size"]) for a in asks)

    best_bid = float(bids_sorted[0]["price"]) if bids_sorted else None
    best_ask = float(asks_sorted[0]["price"]) if asks_sorted else None
    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

    return {
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "total_bid_shares": total_bid_shares,
        "total_ask_shares": total_ask_shares,
        "total_bid_dollar": total_bid_dollar,
        "total_ask_dollar": total_ask_dollar,
        "last_trade_price": book.get("last_trade_price"),
        "tick_size": book.get("tick_size"),
        "bids_sorted": bids_sorted,
        "asks_sorted": asks_sorted,
    }


def scrape_once(conn: sqlite3.Connection, *, market_limit: int | None = None) -> int:
    """Scrape orderbooks for top markets. Returns number of snapshots saved."""
    now = datetime.now(tz=UTC).isoformat()
    saved = 0

    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": "PolymarketBot/1.0"}) as client:
        markets = get_top_markets(client, n=market_limit)
        print(f"[{now}] Fetched {len(markets)} markets from Gamma API")

        for market in markets:
            question = market.get("question", "?")
            slug = market.get("slug", "?")
            raw_token_ids = market.get("clobTokenIds", [])
            if isinstance(raw_token_ids, str):
                token_ids = json.loads(raw_token_ids)
            elif isinstance(raw_token_ids, list):
                token_ids = raw_token_ids
            else:
                token_ids = []

            if not token_ids:
                continue

            for i, token_id in enumerate(token_ids[:2]):
                side = "YES" if i == 0 else "NO"
                try:
                    book = fetch_orderbook(client, token_id)
                    metrics = analyze_book(book)

                    raw_bids = json.dumps(metrics["bids_sorted"][:50])  # cap storage
                    raw_asks = json.dumps(metrics["asks_sorted"][:50])

                    conn.execute(
                        """INSERT INTO snapshots
                           (ts, market_slug, question, token_id, side,
                            bid_levels, ask_levels, best_bid, best_ask, spread,
                            total_bid_shares, total_ask_shares,
                            total_bid_dollar, total_ask_dollar,
                            last_trade_price, tick_size, raw_bids, raw_asks)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            now,
                            slug,
                            question,
                            token_id,
                            side,
                            metrics["bid_levels"],
                            metrics["ask_levels"],
                            metrics["best_bid"],
                            metrics["best_ask"],
                            metrics["spread"],
                            metrics["total_bid_shares"],
                            metrics["total_ask_shares"],
                            metrics["total_bid_dollar"],
                            metrics["total_ask_dollar"],
                            float(metrics["last_trade_price"]) if metrics["last_trade_price"] else None,
                            float(metrics["tick_size"]) if metrics["tick_size"] else None,
                            raw_bids,
                            raw_asks,
                        ),
                    )

                    # Upsert market metadata
                    conn.execute(
                        """INSERT INTO markets (token_id, question, slug, side, volume_24h, liquidity, last_updated)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(token_id) DO UPDATE SET
                             volume_24h=excluded.volume_24h,
                             liquidity=excluded.liquidity,
                             last_updated=excluded.last_updated""",
                        (
                            token_id,
                            question,
                            slug,
                            side,
                            float(market.get("volume24hr", 0)),
                            float(market.get("liquidity", 0)),
                            now,
                        ),
                    )

                    saved += 1
                    sp = metrics["spread"] or 0
                    print(
                        f"  {side:3s} {question[:50]:50s} | "
                        f"bids={metrics['bid_levels']:>3}, asks={metrics['ask_levels']:>3} | "
                        f"bid=${metrics['total_bid_dollar']:>10,.0f}, ask=${metrics['total_ask_dollar']:>10,.0f} | "
                        f"spread={sp:.3f}"
                    )

                except Exception as e:
                    print(f"  ERROR {side} {question[:40]}: {e}")

    conn.commit()
    print(f"[{now}] Saved {saved} snapshots")
    return saved


def print_stats(conn: sqlite3.Connection) -> None:
    """Print summary statistics from stored snapshots."""
    row = conn.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM snapshots").fetchone()
    print(f"Total snapshots: {row[0]}")
    print(f"Time range: {row[1]} to {row[2]}")
    print()

    print("Per-market stats (latest snapshot):")
    print(f"{'Market':50s} | {'Side':4s} | {'BidLvl':>6} | {'AskLvl':>6} | {'Bid$':>12} | {'Ask$':>12} | {'Spread':>7}")
    print("-" * 110)

    rows = conn.execute("""
        SELECT question, side, bid_levels, ask_levels,
               total_bid_dollar, total_ask_dollar, spread
        FROM snapshots
        WHERE ts = (SELECT MAX(ts) FROM snapshots)
        ORDER BY total_bid_dollar + total_ask_dollar DESC
    """).fetchall()

    for r in rows:
        q, side, bl, al, bd, ad, sp = r
        sp = sp or 0
        print(f"{q[:50]:50s} | {side:4s} | {bl:>6} | {al:>6} | ${bd:>10,.0f} | ${ad:>10,.0f} | {sp:>7.3f}")

    print()
    # Show time series for a specific market
    row = conn.execute("""
        SELECT market_slug FROM snapshots
        GROUP BY market_slug
        ORDER BY COUNT(*) DESC LIMIT 1
    """).fetchone()

    if row:
        slug = row[0]
        print(f"Time series for '{slug}' (YES side, last 10):")
        rows = conn.execute(
            """
            SELECT ts, best_bid, best_ask, spread,
                   total_bid_dollar, total_ask_dollar
            FROM snapshots
            WHERE market_slug = ? AND side = 'YES'
            ORDER BY ts DESC LIMIT 10
        """,
            (slug,),
        ).fetchall()
        for r in rows:
            ts, bb, ba, sp, bd, ad = r
            bb = bb or 0
            ba = ba or 0
            sp = sp or 0
            print(f"  {ts} | bid={bb:.3f} ask={ba:.3f} spread={sp:.3f} | bid${bd:>10,.0f} ask${ad:>10,.0f}")


def parse_market_limit(args: list[str]) -> int | None:
    """Parse optional --limit N argument."""
    if "--limit" not in args:
        return None

    idx = args.index("--limit")
    if idx + 1 >= len(args):
        raise SystemExit("--limit requires an integer value")

    try:
        limit = int(args[idx + 1])
    except ValueError as exc:
        raise SystemExit("--limit requires an integer value") from exc

    if limit <= 0:
        raise SystemExit("--limit must be greater than zero")

    return limit


def main() -> None:
    args = sys.argv[1:]
    conn = init_db(DB_PATH)
    market_limit = parse_market_limit(args)

    if "--stats" in args:
        print_stats(conn)
    elif "--loop" in args:
        print(f"Starting continuous scrape (every 60s). DB: {DB_PATH}")
        while True:
            try:
                scrape_once(conn, market_limit=market_limit)
            except Exception as e:
                print(f"Scrape error: {e}")
            time.sleep(60)
    else:
        scrape_once(conn, market_limit=market_limit)

    conn.close()


if __name__ == "__main__":
    main()
