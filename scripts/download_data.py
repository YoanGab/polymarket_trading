"""One-time data download: fetch all available Polymarket resolved markets and their CLOB price histories.

Usage:
    uv run python scripts/download_data.py [--max-markets 500] [--fresh]

This populates data/polymarket_backtest.sqlite with:
- Markets from Gamma API (resolved, high-volume)
- Hourly CLOB price histories (real data, not synthetic)
- Synthetic news placeholders (for replay engine compatibility)
- Orderbook snapshots derived from CLOB prices

Subsequent runs skip markets already in the DB unless --fresh is passed.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx

from polymarket_backtest import db
from polymarket_backtest.downloaders.clob import fetch_price_history
from polymarket_backtest.downloaders.gamma import parse_resolution
from polymarket_backtest.types import isoformat

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "polymarket_backtest.sqlite"
RNG = random.Random(2026)

# Only exclude pure sports markets
EXCLUDE_KEYWORDS = {
    "set 1 games",
    "set 2 games",
    "o/u 8.5",
    "o/u 9.5",
    "moneyline",
    "handicap",
    "parlay",
    "league of legends",
    "world series",
}

NEWS_INTERVAL_HOURS = 3


def fetch_all_resolved_markets(max_markets: int) -> list[dict]:
    """Fetch resolved markets from Gamma API, sorted by volume descending."""
    markets: list[dict] = []
    offset = 0
    page_size = 100

    print(f"Fetching up to {max_markets} resolved markets from Gamma API...")
    while len(markets) < max_markets:
        resp = httpx.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "closed": "true",
                "limit": str(page_size),
                "offset": str(offset),
                "order": "volume",
                "ascending": "false",
            },
            timeout=20,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        markets.extend(page)
        offset += len(page)
        print(f"  Fetched {len(markets)} markets so far...")
        if len(page) < page_size:
            break
        time.sleep(0.05)

    print(f"  Total raw markets: {len(markets)}")
    return markets[:max_markets]


def filter_and_parse_markets(raw_markets: list[dict], existing_ids: set[str]) -> list[dict]:
    """Filter for valid, non-sports, 2024+ markets that aren't already in the DB."""
    valid = []
    for m in raw_markets:
        try:
            parsed = parse_resolution(m)
        except (ValueError, KeyError):
            continue
        if parsed is None:
            continue

        # Must have clobTokenIds
        clob_tokens = m.get("clobTokenIds")
        if not clob_tokens:
            continue
        if isinstance(clob_tokens, str):
            try:
                clob_tokens = json.loads(clob_tokens)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(clob_tokens, list) or len(clob_tokens) < 1:
            continue

        # Filter out sports
        title = (m.get("question") or m.get("title") or "").lower()
        if any(kw in title for kw in EXCLUDE_KEYWORDS):
            continue

        # Only 2024+ markets
        res_ts = parsed.get("resolution_ts", "")
        if isinstance(res_ts, str) and res_ts < "2024":
            continue

        market_id = parsed.get("condition_id") or m.get("conditionId", "")
        if market_id in existing_ids:
            continue

        m["_parsed_resolution"] = parsed
        valid.append(m)

    return valid


def extract_token_id(market: dict) -> str | None:
    """Extract the first clobTokenId from a Gamma market dict."""
    clob_tokens = market.get("clobTokenIds")
    if isinstance(clob_tokens, str):
        try:
            clob_tokens = json.loads(clob_tokens)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(clob_tokens, list) and clob_tokens:
        return str(clob_tokens[0])
    return None


def fetch_clob_prices(token_id: str, resolution_ts_iso: str | None) -> list[dict] | None:
    """Fetch real hourly CLOB prices. Returns None if unavailable."""
    try:
        if resolution_ts_iso:
            res_dt = datetime.fromisoformat(resolution_ts_iso.replace("Z", "+00:00"))
            end_ts = int(res_dt.timestamp())
            start_ts = end_ts - 30 * 86400  # 30 days before resolution
            history = fetch_price_history(token_id=token_id, start_ts=start_ts, end_ts=end_ts, fidelity=60)
        else:
            history = fetch_price_history(token_id=token_id, interval="max", fidelity=60)
    except Exception as exc:
        print(f"    CLOB fetch failed for {token_id[:16]}...: {exc}")
        return None

    if not history or len(history) < 10:
        return None

    path = []
    for point in history:
        price = float(point.get("price", 0))
        if price <= 0 or price >= 1:
            continue
        if price <= 0.10 or price >= 0.90:
            half_spread = 0.01
        elif price <= 0.20 or price >= 0.80:
            half_spread = 0.015
        else:
            half_spread = 0.02
        bid = max(0.001, price - half_spread)
        ask = min(0.999, price + half_spread)
        path.append(
            {
                "timestamp": int(point.get("timestamp", 0)),
                "mid": round(price, 4),
                "best_bid": round(bid, 4),
                "best_ask": round(ask, 4),
                "volume": float("nan"),
            }
        )
    return path if len(path) >= 10 else None


def ingest_market(
    conn: sqlite3.Connection,
    market: dict,
    idx: int,
) -> dict[str, int]:
    """Ingest a single market into the DB. Returns stats."""
    stats = {"snapshots": 0, "real_prices": 0, "synthetic": 0, "news": 0}
    parsed = market["_parsed_resolution"]
    market_id = parsed.get("condition_id") or market.get("conditionId", f"market_{idx}")
    title = market.get("question") or market.get("title", f"Market {idx}")
    resolved_yes = parsed["resolved_outcome"] >= 0.5

    # Fetch real CLOB prices
    token_id = extract_token_id(market)
    real_path = None
    if token_id:
        real_path = fetch_clob_prices(token_id, parsed.get("resolution_ts"))
        time.sleep(0.12)

    if real_path:
        stats["real_prices"] = 1
        path = real_path
        first_ts = datetime.fromtimestamp(path[0]["timestamp"], tz=UTC)
        market_base = first_ts
        resolution_ts = datetime.fromtimestamp(path[-1]["timestamp"], tz=UTC) + timedelta(hours=1)

        # Enrich with Gamma volume
        gamma_vol_str = market.get("volume") or market.get("volume24hr") or "0"
        try:
            gamma_vol = float(gamma_vol_str)
        except (ValueError, TypeError):
            gamma_vol = 0.0
        if gamma_vol > 0:
            for snap in path:
                snap["volume_24h"] = gamma_vol
                snap["volume_1m"] = gamma_vol / 1440.0
    else:
        stats["synthetic"] = 1
        # Generate synthetic price path
        n_snapshots = 72
        if resolved_yes:
            start_mid, end_mid = RNG.uniform(0.55, 0.80), RNG.uniform(0.92, 0.98)
        else:
            start_mid, end_mid = RNG.uniform(0.20, 0.45), RNG.uniform(0.02, 0.08)
        path = []
        for i in range(n_snapshots):
            frac = i / max(1, n_snapshots - 1)
            base = start_mid + (end_mid - start_mid) * (frac**0.8)
            noise = RNG.gauss(0, 0.015) * (1 - frac * 0.5)
            mid = max(0.01, min(0.99, base + noise))
            spread = RNG.uniform(0.005, 0.02) * (1 + 0.5 * (1 - frac))
            bid = max(0.001, mid - spread / 2)
            ask = min(0.999, mid + spread / 2)
            if bid >= ask:
                ask = min(0.999, bid + 0.01)
            path.append(
                {
                    "mid": round(mid, 4),
                    "best_bid": round(bid, 4),
                    "best_ask": round(ask, 4),
                    "volume": round(RNG.uniform(500, 5000) * (1 + frac), 2),
                }
            )
        market_base = datetime(2025, 11, 1, tzinfo=UTC) + timedelta(minutes=idx * 37)
        resolution_ts = market_base + timedelta(hours=n_snapshots)

    # Insert market
    db.add_market(
        conn,
        market_id=market_id,
        title=title,
        domain="politics",
        market_type="binary",
        open_ts=market_base - timedelta(days=7),
        close_ts=resolution_ts,
        resolution_ts=resolution_ts,
        status="resolved",
    )
    db.add_resolution(
        conn,
        market_id=market_id,
        resolved_outcome=1.0 if resolved_yes else 0.0,
        resolution_ts=resolution_ts,
        status="resolved",
    )
    db.add_rule_revision(
        conn,
        market_id=market_id,
        rules_text=market.get("description", f"Rules for {title}"),
        effective_ts=market_base,
    )

    # Insert snapshots
    for snap_idx, snap in enumerate(path):
        if real_path:
            snap_ts = datetime.fromtimestamp(snap["timestamp"], tz=UTC)
        else:
            snap_ts = market_base + timedelta(hours=snap_idx)

        vol_24h = snap.get("volume_24h")
        vol_1m = snap.get("volume_1m")
        if vol_24h is not None and not (isinstance(vol_24h, float) and math.isnan(vol_24h)):
            volume_24h = vol_24h
            volume_1m = vol_1m if vol_1m is not None else vol_24h / 1440.0
        else:
            raw_vol = snap.get("volume", 0.0)
            if isinstance(raw_vol, float) and math.isnan(raw_vol):
                raw_vol = 0.0
            volume_val = raw_vol or RNG.uniform(500, 5000)
            volume_1m = volume_val / 60
            volume_24h = volume_val * 24

        db.add_snapshot(
            conn,
            market_id=market_id,
            ts=snap_ts,
            best_bid=snap["best_bid"],
            best_ask=snap["best_ask"],
            last_trade=snap["mid"],
            volume_1m=volume_1m,
            volume_24h=volume_24h,
            open_interest=RNG.uniform(10000, 100000),
            tick_size=0.01,
            status="active",
            orderbook=[
                ("bid", 0, snap["best_bid"], RNG.uniform(100, 500)),
                ("bid", 1, round(snap["best_bid"] - 0.01, 4), RNG.uniform(200, 800)),
                ("ask", 0, snap["best_ask"], RNG.uniform(100, 500)),
                ("ask", 1, round(snap["best_ask"] + 0.01, 4), RNG.uniform(200, 800)),
            ],
        )
        stats["snapshots"] += 1

    # Add news
    n_snapshots = len(path)
    n_news = max(1, n_snapshots // NEWS_INTERVAL_HOURS)
    for news_idx in range(n_news):
        news_ts = market_base + timedelta(hours=news_idx * NEWS_INTERVAL_HOURS + 1)
        db.add_news(
            conn,
            document_id=f"news_{market_id}_{news_idx}",
            source="reuters",
            url=f"https://reuters.com/article/{market_id}/{news_idx}",
            title=f"Update: {title}",
            published_ts=news_ts,
            first_seen_ts=news_ts,
            ingested_ts=news_ts,
            content=f"Market update for {title}: situation remains unchanged as of {isoformat(news_ts)}.",
            metadata={},
            market_ids=[market_id],
        )
        stats["news"] += 1

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Polymarket data for backtesting")
    parser.add_argument("--max-markets", type=int, default=500, help="Max markets to fetch")
    parser.add_argument("--fresh", action="store_true", help="Delete existing DB and start fresh")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.fresh and DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Deleted existing DB: {DB_PATH}")

    # Get existing market IDs
    existing_ids: set[str] = set()
    if DB_PATH.exists():
        conn = db.connect(DB_PATH)
        db.init_db(conn)
        rows = conn.execute("SELECT market_id FROM markets").fetchall()
        existing_ids = {str(row["market_id"]) for row in rows}
        conn.close()
        print(f"DB already has {len(existing_ids)} markets")

    # Fetch from Gamma
    raw_markets = fetch_all_resolved_markets(args.max_markets * 3)
    new_markets = filter_and_parse_markets(raw_markets, existing_ids)
    print(f"  {len(new_markets)} new valid markets to ingest")

    if not new_markets:
        print("Nothing to do — all markets already in DB")
        return

    # Take up to max_markets
    # Balance YES/NO
    yes_markets = [m for m in new_markets if m["_parsed_resolution"]["resolved_outcome"] >= 0.5]
    no_markets = [m for m in new_markets if m["_parsed_resolution"]["resolved_outcome"] < 0.5]
    RNG.shuffle(yes_markets)
    RNG.shuffle(no_markets)
    half = args.max_markets // 2
    selected = yes_markets[:half] + no_markets[:half]
    remaining = [m for m in new_markets if m not in selected]
    RNG.shuffle(remaining)
    selected.extend(remaining[: args.max_markets - len(selected)])

    n_yes = sum(1 for m in selected if m["_parsed_resolution"]["resolved_outcome"] >= 0.5)
    print(f"  Selected {len(selected)} markets ({n_yes} YES, {len(selected) - n_yes} NO)")

    # Ingest
    conn = db.connect(DB_PATH)
    db.init_db(conn)

    totals = {"markets": 0, "snapshots": 0, "real_prices": 0, "synthetic": 0, "news": 0}
    for idx, market in enumerate(selected):
        title = (market.get("question") or market.get("title", ""))[:60]
        print(f"  [{idx + 1}/{len(selected)}] {title}...", end=" ", flush=True)

        try:
            stats = ingest_market(conn, market, idx)
            for key in totals:
                if key in stats:
                    totals[key] += stats[key]
            totals["markets"] += 1
            src = "REAL" if stats["real_prices"] else "SYNTH"
            print(f"{src} ({stats['snapshots']} snapshots)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

        if (idx + 1) % 10 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"  Database: {DB_PATH}")
    print(f"  New markets: {totals['markets']}")
    print(f"  Total snapshots: {totals['snapshots']}")
    print(f"  Real CLOB prices: {totals['real_prices']}")
    print(f"  Synthetic fallback: {totals['synthetic']}")
    print(f"  News items: {totals['news']}")

    # Show total DB size
    conn = db.connect(DB_PATH)
    total_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    total_snapshots = conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
    conn.close()
    print(f"\n  Total in DB: {total_markets} markets, {total_snapshots} snapshots")
    db_size_mb = DB_PATH.stat().st_size / 1_048_576
    print(f"  DB size: {db_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
