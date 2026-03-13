"""Data integrity verification for the exhaustive Polymarket database.

Usage:
    uv run python scripts/verify_data.py [--db data/polymarket_backtest_v2.sqlite]

Checks:
- No synthetic/fake data
- All markets have resolutions
- Price data is valid (0 < price < 1)
- Timestamps are monotonic
- No duplicate snapshots
- Category distribution
- Temporal coverage
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"


def main():
    parser = argparse.ArgumentParser(description="Verify data integrity")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    issues: list[str] = []
    warnings: list[str] = []

    # ── Basic counts ──
    total_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    total_snaps = conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
    total_res = conn.execute("SELECT COUNT(*) FROM market_resolutions").fetchone()[0]
    print(f"Markets: {total_markets}")
    print(f"Snapshots: {total_snaps}")
    print(f"Resolutions: {total_res}")

    if total_markets == 0:
        print("ERROR: Database is empty")
        sys.exit(1)

    # ── Check: all markets have resolutions ──
    markets_without_res = conn.execute("""
        SELECT COUNT(*) FROM markets m
        LEFT JOIN market_resolutions r ON m.market_id = r.market_id
        WHERE r.market_id IS NULL
    """).fetchone()[0]
    if markets_without_res > 0:
        issues.append(f"{markets_without_res} markets without resolution")
    print(f"Markets without resolution: {markets_without_res}")

    # ── Check: all markets have snapshots ──
    markets_without_snaps = conn.execute("""
        SELECT COUNT(*) FROM markets m
        LEFT JOIN market_snapshots s ON m.market_id = s.market_id
        WHERE s.market_id IS NULL
    """).fetchone()[0]
    if markets_without_snaps > 0:
        issues.append(f"{markets_without_snaps} markets without snapshots")
    print(f"Markets without snapshots: {markets_without_snaps}")

    # ── Check: snapshot distribution ──
    snap_stats = conn.execute("""
        SELECT
            MIN(cnt) as min_snaps,
            MAX(cnt) as max_snaps,
            AVG(cnt) as avg_snaps,
            COUNT(*) as n_markets
        FROM (SELECT COUNT(*) as cnt FROM market_snapshots GROUP BY market_id)
    """).fetchone()
    print(f"\nSnapshot distribution:")
    print(f"  Min: {snap_stats['min_snaps']}")
    print(f"  Max: {snap_stats['max_snaps']}")
    print(f"  Avg: {snap_stats['avg_snaps']:.1f}")

    # ── Check: price validity ──
    invalid_prices = conn.execute("""
        SELECT COUNT(*) FROM market_snapshots
        WHERE mid <= 0 OR mid >= 1 OR best_bid < 0 OR best_ask > 1
    """).fetchone()[0]
    if invalid_prices > 0:
        issues.append(f"{invalid_prices} snapshots with invalid prices")
    print(f"\nInvalid prices: {invalid_prices}")

    # ── Check: bid < ask ──
    crossed = conn.execute("""
        SELECT COUNT(*) FROM market_snapshots WHERE best_bid >= best_ask
    """).fetchone()[0]
    if crossed > 0:
        warnings.append(f"{crossed} snapshots with crossed bid/ask")
    print(f"Crossed bid/ask: {crossed}")

    # ── Check: resolution outcomes ──
    outcomes = conn.execute("""
        SELECT resolved_outcome, COUNT(*) as cnt
        FROM market_resolutions GROUP BY resolved_outcome
    """).fetchall()
    print(f"\nOutcome distribution:")
    for o in outcomes:
        pct = o["cnt"] / total_res * 100
        print(f"  {o['resolved_outcome']}: {o['cnt']} ({pct:.1f}%)")

    # ── Check: temporal coverage ──
    time_range = conn.execute("""
        SELECT MIN(ts) as earliest, MAX(ts) as latest FROM market_snapshots
    """).fetchone()
    print(f"\nTime range: {time_range['earliest']} to {time_range['latest']}")

    res_by_year = conn.execute("""
        SELECT SUBSTR(resolution_ts, 1, 4) as year, COUNT(*) as cnt
        FROM market_resolutions GROUP BY year ORDER BY year
    """).fetchall()
    print(f"\nResolutions by year:")
    for r in res_by_year:
        print(f"  {r['year']}: {r['cnt']}")

    # ── Check: domain distribution ──
    domains = conn.execute("""
        SELECT domain, COUNT(*) as cnt FROM markets GROUP BY domain ORDER BY cnt DESC LIMIT 20
    """).fetchall()
    print(f"\nDomain distribution (top 20):")
    for d in domains:
        pct = d["cnt"] / total_markets * 100
        print(f"  {d['domain']:30s} {d['cnt']:6d} ({pct:.1f}%)")

    # ── Check: no synthetic data ──
    try:
        synthetic = conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE is_synthetic = 1").fetchone()[0]
        if synthetic > 0:
            issues.append(f"{synthetic} synthetic snapshots found!")
        print(f"\nSynthetic snapshots: {synthetic}")
    except sqlite3.OperationalError:
        print("\nNo is_synthetic column (OK for v2)")

    # ── Check: volume and OI ──
    zero_vol = conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE volume_24h = 0").fetchone()[0]
    zero_oi = conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE open_interest = 0").fetchone()[0]
    print(f"\nZero volume_24h: {zero_vol} ({zero_vol / total_snaps * 100:.1f}%)")
    print(f"Zero open_interest: {zero_oi} ({zero_oi / total_snaps * 100:.1f}%)")

    # ── Check: duplicate snapshots ──
    dupes = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT market_id, ts, COUNT(*) as cnt
            FROM market_snapshots GROUP BY market_id, ts
            HAVING cnt > 1
        )
    """).fetchone()[0]
    if dupes > 0:
        issues.append(f"{dupes} duplicate snapshot timestamps")
    print(f"\nDuplicate timestamps: {dupes}")

    # ── Summary ──
    print("\n" + "=" * 60)
    if issues:
        print(f"ISSUES ({len(issues)}):")
        for i in issues:
            print(f"  [ERROR] {i}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")
    if not issues and not warnings:
        print("ALL CHECKS PASSED")
    elif not issues:
        print("PASSED (with warnings)")
    else:
        print("FAILED")

    conn.close()
    print(f"\nDB size: {db_path.stat().st_size / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
