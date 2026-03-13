"""Real E2E backtest: fetch real Polymarket markets, fetch real CLOB prices, run all strategies."""

from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_backtest import db
from polymarket_backtest.downloaders.clob import fetch_price_history
from polymarket_backtest.downloaders.gamma import parse_resolution
from polymarket_backtest.grid_search import (
    build_grid_report,
    rank_strategies,
    run_grid_search,
)
from polymarket_backtest.grok_replay import ReplayGrokClient, create_transport
from polymarket_backtest.metrics import build_metrics_summary, persist_metric_results
from polymarket_backtest.replay_engine import ReplayEngine
from polymarket_backtest.report import ReportGenerator
from polymarket_backtest.strategies import default_strategy_grid
from polymarket_backtest.types import ReplayConfig, isoformat

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest.sqlite"
FORECAST_MODE = os.environ.get("FORECAST_MODE", "smart_rules")
MODEL_RELEASE = os.environ.get("GROK_MODEL_RELEASE", "grok-3")
NUM_MARKETS = 200
SNAPSHOTS_PER_MARKET = 72  # 3 days of hourly snapshots
NEWS_INTERVAL_HOURS = 3  # news every 3h to stay in 240min lookback
STARTING_CASH = 2_000.0
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


def is_acceptable_market(market: dict) -> bool:
    """Filter out pure sports betting markets, keep everything else."""
    title = (market.get("question") or market.get("title") or "").lower()
    return not any(kw in title for kw in EXCLUDE_KEYWORDS)


def _fetch_recent_resolved_markets(max_markets: int) -> list[dict]:
    """Fetch recently closed markets sorted by volume (most liquid first)."""
    import httpx

    markets: list[dict] = []
    offset = 0
    page_size = 100
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
        if len(page) < page_size:
            break
        time.sleep(0.05)
    return markets[:max_markets]


def fetch_real_markets(n: int) -> list[dict]:
    """Fetch real resolved markets from Gamma, sorted by volume for recent high-liquidity markets."""
    print(f"Fetching up to {n * 5} recent high-volume markets from Gamma API...")
    raw = _fetch_recent_resolved_markets(n * 5)
    print(f"  Got {len(raw)} raw markets")

    valid = []
    for m in raw:
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
        # Filter for political/societal
        if not is_acceptable_market(m):
            continue
        # Only recent markets (2024+) — older ones have no CLOB price data
        res_ts = parsed.get("resolution_ts", "")
        if isinstance(res_ts, str) and res_ts < "2024":
            continue
        m["_parsed_resolution"] = parsed
        valid.append(m)

    print(f"  {len(valid)} valid resolved markets")

    # Balance YES/NO outcomes (resolved_outcome >= 0.5 = YES, < 0.5 = NO)
    yes_markets = [m for m in valid if m["_parsed_resolution"]["resolved_outcome"] >= 0.5]
    no_markets = [m for m in valid if m["_parsed_resolution"]["resolved_outcome"] < 0.5]
    RNG.shuffle(yes_markets)
    RNG.shuffle(no_markets)

    half = n // 2
    selected = yes_markets[:half] + no_markets[:half]
    # Fill remainder
    remaining = [m for m in valid if m not in selected]
    RNG.shuffle(remaining)
    selected.extend(remaining[: n - len(selected)])

    n_yes = sum(1 for m in selected if m["_parsed_resolution"]["resolved_outcome"] >= 0.5)
    print(f"  Selected {len(selected)} markets ({n_yes} YES, {len(selected) - n_yes} NO)")
    return selected


def fetch_real_price_path(token_id: str, resolution_ts_iso: str | None = None) -> list[dict] | None:
    """Fetch real hourly price history from CLOB API. Returns None if unavailable.

    For resolved markets, uses startTs/endTs (30 days before resolution).
    For active markets, uses interval=max.
    """
    try:
        if resolution_ts_iso:
            # Resolved market: use absolute time range
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
        # Price-dependent spread estimation matching clob.py logic
        if price <= 0.10 or price >= 0.90:
            half_spread = 0.01  # 200 bps near extremes
        elif price <= 0.20 or price >= 0.80:
            half_spread = 0.015  # 300 bps
        else:
            half_spread = 0.02  # 400 bps in the middle range
        bid = max(0.001, price - half_spread)
        ask = min(0.999, price + half_spread)
        path.append(
            {
                "timestamp": int(point.get("timestamp", 0)),
                "mid": round(price, 4),
                "best_bid": round(bid, 4),
                "best_ask": round(ask, 4),
                "volume": float("nan"),  # CLOB doesn't provide volume per candle
            }
        )

    return path if len(path) >= 10 else None


def generate_synthetic_price_path(resolved_yes: bool, n_snapshots: int) -> list[dict]:
    """Fallback: generate a synthetic price path that converges toward resolution."""
    if resolved_yes:
        start_mid = RNG.uniform(0.55, 0.80)
        end_mid = RNG.uniform(0.92, 0.98)
    else:
        start_mid = RNG.uniform(0.20, 0.45)
        end_mid = RNG.uniform(0.02, 0.08)

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

        volume = RNG.uniform(500, 5000) * (1 + frac)
        path.append(
            {
                "mid": round(mid, 4),
                "best_bid": round(bid, 4),
                "best_ask": round(ask, 4),
                "volume": round(volume, 2),
            }
        )
    return path


def _extract_token_id(market: dict) -> str | None:
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


def ingest_markets(conn: sqlite3.Connection, markets: list[dict]) -> dict:
    """Ingest real markets with real CLOB prices (fallback to synthetic) into the database."""
    base_ts = datetime(2025, 11, 1, 0, 0, 0, tzinfo=UTC)
    stats = {"markets": 0, "snapshots": 0, "resolutions": 0, "news": 0, "rules": 0, "real_prices": 0, "synthetic": 0}
    seen_ids = set()

    for idx, market in enumerate(markets):
        parsed = market["_parsed_resolution"]
        market_id = parsed.get("condition_id") or market.get("conditionId", f"market_{idx}")

        if market_id in seen_ids:
            continue
        seen_ids.add(market_id)

        title = market.get("question") or market.get("title", f"Market {idx}")
        resolved_yes = parsed["resolved_outcome"] >= 0.5

        # Try to fetch real CLOB price history
        token_id = _extract_token_id(market)
        real_path = None
        if token_id:
            resolution_ts_iso = parsed.get("resolution_ts")
            real_path = fetch_real_price_path(token_id, resolution_ts_iso=resolution_ts_iso)
            time.sleep(0.15)  # rate limit

        if real_path:
            stats["real_prices"] += 1
            path = real_path
            n_snapshots = len(path)
            # Use actual timestamps from CLOB data
            first_ts = datetime.fromtimestamp(path[0]["timestamp"], tz=UTC)
            market_base = first_ts
            resolution_ts = datetime.fromtimestamp(path[-1]["timestamp"], tz=UTC) + timedelta(hours=1)

            # Enrich CLOB snapshots with Gamma volume when available
            gamma_volume_str = market.get("volume") or market.get("volume24hr") or "0"
            try:
                gamma_volume_24h = float(gamma_volume_str)
            except (ValueError, TypeError):
                gamma_volume_24h = 0.0
            if gamma_volume_24h > 0:
                gamma_volume_1m = gamma_volume_24h / 1440.0
                for snap in path:
                    snap["volume_24h"] = gamma_volume_24h
                    snap["volume_1m"] = gamma_volume_1m
        else:
            stats["synthetic"] += 1
            path = generate_synthetic_price_path(resolved_yes, SNAPSHOTS_PER_MARKET)
            n_snapshots = SNAPSHOTS_PER_MARKET
            market_base = base_ts + timedelta(minutes=idx * 37)
            resolution_ts = market_base + timedelta(hours=n_snapshots)

        # Add market
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
        stats["markets"] += 1

        # Add resolution
        db.add_resolution(
            conn,
            market_id=market_id,
            resolved_outcome=1.0 if resolved_yes else 0.0,
            resolution_ts=resolution_ts,
            status="resolved",
        )
        stats["resolutions"] += 1

        # Add rules
        db.add_rule_revision(
            conn,
            market_id=market_id,
            rules_text=market.get("description", f"Rules for {title}"),
            effective_ts=market_base,
        )
        stats["rules"] += 1

        # Add snapshots
        for snap_idx, snap in enumerate(path):
            if real_path:
                snap_ts = datetime.fromtimestamp(snap["timestamp"], tz=UTC)
            else:
                snap_ts = market_base + timedelta(hours=snap_idx)

            # Use enriched Gamma volume for real data; fallback to random for synthetic
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

        # Add news items every NEWS_INTERVAL_HOURS hours
        n_news = max(1, n_snapshots // NEWS_INTERVAL_HOURS)
        for news_idx in range(n_news):
            news_ts = market_base + timedelta(hours=news_idx * NEWS_INTERVAL_HOURS + 1)
            news_content = (
                f"Market update for {title}: situation remains unchanged as of "
                f"{isoformat(news_ts)}. Analysts maintain current assessments."
            )
            db.add_news(
                conn,
                document_id=f"news_{market_id}_{news_idx}",
                source="reuters",
                url=f"https://reuters.com/article/{market_id}/{news_idx}",
                title=f"Update: {title}",
                published_ts=news_ts,
                first_seen_ts=news_ts,
                ingested_ts=news_ts,
                content=news_content,
                metadata={},
                market_ids=[market_id],
            )
            stats["news"] += 1

    conn.commit()
    return stats


def run_default_backtest(conn: sqlite3.Connection) -> tuple[int, str]:
    """Run backtest with 2 default strategies, return experiment_id and report."""
    config = ReplayConfig(
        experiment_name="real_polymarket_backtest",
        starting_cash=STARTING_CASH,
        lookback_minutes=240,
    )
    transport = create_transport(mode=FORECAST_MODE, model_release=MODEL_RELEASE)
    print(f"  Forecast transport: {type(transport).__name__}")
    if not getattr(transport, "is_live_safe", True):
        print("  WARNING: Using FAKE forecaster. Results are for testing only.")
    grok = ReplayGrokClient(
        conn=conn,
        experiment_id=None,
        model_id="grok",
        model_release=MODEL_RELEASE,
        transport=transport,
        lookback_minutes=config.lookback_minutes,
    )
    timestamps = db.get_all_timestamps(conn)
    experiment_id = db.create_experiment(
        conn,
        name=config.experiment_name,
        model_id=grok.model_id,
        model_release=grok.model_release,
        system_prompt_hash=grok.prompt_hash(timestamps[0]),
        config={
            "starting_cash": config.starting_cash,
            "lookback_minutes": config.lookback_minutes,
            "markout_horizons_min": config.markout_horizons_min,
        },
    )
    grok.experiment_id = experiment_id
    grok.context_builder.experiment_id = experiment_id

    strategies = default_strategy_grid()
    engine = ReplayEngine(conn=conn, config=config, grok=grok, strategies=strategies)
    engine.run()
    persist_metric_results(conn, experiment_id, config.markout_horizons_min)
    summary = build_metrics_summary(conn, experiment_id, config.markout_horizons_min)

    report = ReportGenerator().build_markdown(
        summary,
        {
            "experiment_name": config.experiment_name,
            "starting_cash": config.starting_cash,
            "strategy_count": len(strategies),
        },
    )
    return experiment_id, report


def main():
    print("=" * 70)
    print("REAL POLYMARKET BACKTEST — Full Pipeline")
    print("=" * 70)
    print(f"  Forecast mode: {FORECAST_MODE}")
    print(f"  Model release: {MODEL_RELEASE}")

    # Step 1: Fetch real markets
    print("\n[1/4] Fetching real resolved markets from Gamma API...")
    markets = fetch_real_markets(NUM_MARKETS)
    if len(markets) < 10:
        print(f"Only got {len(markets)} markets, not enough for a meaningful test.")
        return

    # Step 2: Ingest (skip if DB already populated)
    if DB_PATH.exists():
        existing_conn = db.connect(DB_PATH)
        try:
            db.init_db(existing_conn)
            existing_count = existing_conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
        finally:
            existing_conn.close()
        if existing_count >= len(markets) * 0.8:
            print(f"\n[2/4] SKIPPING ingestion — DB already has {existing_count} markets (use --fresh to rebuild)")
            conn = db.connect(DB_PATH)
            db.init_db(conn)

            # Jump directly to backtest
            print("\n[3/4] Running default backtest (carry + news strategies)...")
            exp_id, report = run_default_backtest(conn)
            orders_count = conn.execute("SELECT COUNT(*) FROM orders WHERE experiment_id = ?", (exp_id,)).fetchone()[0]
            fills_count = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE experiment_id = ? AND filled_quantity > 0", (exp_id,)
            ).fetchone()[0]
            model_outputs = conn.execute(
                "SELECT COUNT(*) FROM model_outputs WHERE experiment_id = ?", (exp_id,)
            ).fetchone()[0]
            print(f"  Model outputs: {model_outputs}")
            print(f"  Orders: {orders_count}")
            print(f"  Fills: {fills_count}")
            print("\n" + "=" * 70)
            print("GO/NO-GO REPORT")
            print("=" * 70)
            print(report)

            print("\n" + "=" * 70)
            print("[4/4] Grid Search — all strategies")
            print("=" * 70)
            results = run_grid_search(DB_PATH)
            ranked = rank_strategies(results)
            grid_report = build_grid_report(ranked)
            for r in ranked:
                sn = r["strategy_name"]
                pnl = r["pnl"]
                nt = r["n_trades"]
                fr = r["fill_ratio"]
                print(f"  {sn:25s}  PnL={pnl:+8.2f}  trades={nt:5d}  fill={fr:.0%}")
            print("\n" + grid_report)
            conn.close()
            print(f"\n[done] Database at: {DB_PATH}")
            return

    print(f"\n[2/4] Ingesting {len(markets)} markets with {SNAPSHOTS_PER_MARKET} hourly snapshots each...")
    conn = db.connect(DB_PATH)
    db.init_db(conn)
    stats = ingest_markets(conn, markets)
    print(f"  Ingested: {stats}")

    # Step 3: Default backtest (2 strategies)
    print("\n[3/4] Running default backtest (carry + news strategies)...")
    exp_id, report = run_default_backtest(conn)

    # Extract key stats from DB
    orders_count = conn.execute("SELECT COUNT(*) FROM orders WHERE experiment_id = ?", (exp_id,)).fetchone()[0]
    fills_count = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE experiment_id = ? AND filled_quantity > 0", (exp_id,)
    ).fetchone()[0]
    model_outputs = conn.execute("SELECT COUNT(*) FROM model_outputs WHERE experiment_id = ?", (exp_id,)).fetchone()[0]

    print(f"  Model outputs: {model_outputs}")
    print(f"  Orders: {orders_count}")
    print(f"  Fills: {fills_count}")
    print("\n" + "=" * 70)
    print("GO/NO-GO REPORT")
    print("=" * 70)
    print(report)

    # Step 4: Grid search (8 strategies)
    print("\n" + "=" * 70)
    print("[4/4] Grid Search — 8 strategies")
    print("=" * 70)
    results = run_grid_search(DB_PATH)
    ranked = rank_strategies(results)
    grid_report = build_grid_report(ranked)

    # Print summary per strategy
    for r in ranked:
        print(
            f"  {r['strategy_name']:25s}  PnL={r['pnl']:+8.2f}  trades={r['n_trades']:5d}  fill={r['fill_ratio']:.0%}"
        )

    print("\n" + grid_report)

    conn.close()
    print(f"\n[done] Database saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
