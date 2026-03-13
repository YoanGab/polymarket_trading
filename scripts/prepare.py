"""Prepare datasets for autoresearch — pre-extract features from SQLite.

THIS FILE IS FIXED. The autoresearch agent must NOT modify it.

Reads the 19GB SQLite database once, extracts all features (including event tags),
splits into train/val/test, and saves as fast-loading .npz files.

Usage:
    uv run python scripts/prepare.py [--stride 6] [--top-tags 50]

Output:
    data/prepared/
        train.npz    — X, y, market_ids, timestamps
        val.npz      — X, y, market_ids, timestamps
        test.npz     — X, y, market_ids, timestamps
        meta.json    — feature_names, split sizes, tag list, build timestamp
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_backtest import db
from polymarket_backtest.features import extract_snapshot_features

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "prepared"

# Fixed date cutoffs (same as features.py walk_forward_split)
TRAIN_CUTOFF = "2025-10-01"
VAL_CUTOFF = "2026-01-01"


def get_top_tags(conn: sqlite3.Connection, top_n: int) -> list[str]:
    """Get the top N most frequent event tags across all markets."""
    rows = conn.execute(
        """
        SELECT et.tag, COUNT(DISTINCT me.market_id) AS market_count
        FROM event_tags et
        JOIN market_events me ON me.event_id = et.event_id
        GROUP BY et.tag
        ORDER BY market_count DESC
        LIMIT ?
        """,
        (top_n,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def get_market_tags(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Get all tags for each market_id."""
    rows = conn.execute(
        """
        SELECT me.market_id, et.tag
        FROM market_events me
        JOIN event_tags et ON et.event_id = me.event_id
        """
    ).fetchall()
    market_tags: dict[str, set[str]] = {}
    for row in rows:
        mid = str(row[0])
        tag = str(row[1])
        if mid not in market_tags:
            market_tags[mid] = set()
        market_tags[mid].add(tag)
    return market_tags


def build_and_split(
    conn: sqlite3.Connection,
    snapshot_stride: int,
    top_tags: list[str],
    market_tags: dict[str, set[str]],
) -> dict[str, dict]:
    """Build features and split into train/val/test in a single pass.

    Uses batch queries instead of per-market queries for speed.
    """
    # Pre-load ALL resolutions in one query
    print("  Loading resolutions...")
    resolutions: dict[str, float] = {}
    for row in conn.execute("SELECT market_id, resolved_outcome FROM market_resolutions"):
        resolutions[str(row[0])] = float(row[1])
    print(f"  {len(resolutions):,} resolutions loaded")

    # Accumulators per split
    splits: dict[str, dict] = {
        name: {"features": [], "labels": [], "market_ids": [], "timestamps": []} for name in ["train", "val", "test"]
    }

    tag_set = set(top_tags)
    processed = 0
    skipped_no_resolution = 0
    skipped_too_few = 0

    # Stream ALL snapshots ordered by (market_id, ts) — single table scan
    print("  Streaming snapshots (single query)...")
    t0 = time.monotonic()

    cursor = conn.execute(
        """
        SELECT s.market_id, s.ts, s.mid, s.best_bid, s.best_ask, s.last_trade,
               s.volume_1m, s.volume_24h, s.open_interest, m.resolution_ts
        FROM market_snapshots s
        JOIN markets m ON m.market_id = s.market_id
        ORDER BY s.market_id, s.ts ASC
        """
    )

    current_market_id: str | None = None
    market_rows: list[sqlite3.Row] = []
    markets_seen = 0

    def process_market(market_id: str, rows: list[sqlite3.Row]) -> None:
        nonlocal processed, skipped_no_resolution, skipped_too_few

        if market_id not in resolutions:
            skipped_no_resolution += 1
            return
        if len(rows) < 10:
            skipped_too_few += 1
            return

        resolved_outcome = resolutions[market_id]

        # Determine split by last snapshot timestamp
        last_ts = str(rows[-1]["ts"])
        if last_ts < TRAIN_CUTOFF:
            split_name = "train"
        elif last_ts < VAL_CUTOFF:
            split_name = "val"
        else:
            split_name = "test"

        split = splits[split_name]
        mtags = market_tags.get(market_id, set())

        # Sample with stride
        sampled_indices = list(range(0, len(rows), snapshot_stride))
        if sampled_indices[-1] != len(rows) - 1:
            sampled_indices.append(len(rows) - 1)

        for idx in sampled_indices:
            row = rows[idx]
            prev_rows = rows[max(0, idx - 24) : idx]
            features = extract_snapshot_features(row, prev_rows)

            for tag in top_tags:
                features[f"tag_{tag}"] = 1.0 if tag in mtags else 0.0
            features["n_tags"] = float(len(mtags & tag_set))

            split["features"].append(features)
            split["labels"].append(resolved_outcome)
            split["market_ids"].append(market_id)
            split["timestamps"].append(str(row["ts"]))

        processed += 1

    for row in cursor:
        mid = str(row["market_id"])
        if mid != current_market_id:
            # Process previous market
            if current_market_id is not None:
                process_market(current_market_id, market_rows)
            current_market_id = mid
            market_rows = []
            markets_seen += 1
            if markets_seen % 10000 == 0:
                elapsed = time.monotonic() - t0
                rate = markets_seen / max(elapsed, 0.1)
                print(
                    f"  [{markets_seen:,} markets] {rate:.0f}/s, "
                    f"processed {processed:,}, ETA {(181275 - markets_seen) / max(rate, 0.1) / 60:.1f}m",
                    flush=True,
                )
        market_rows.append(row)

    # Process last market
    if current_market_id is not None:
        process_market(current_market_id, market_rows)

    elapsed = time.monotonic() - t0
    print(f"\n  Processed {processed:,} markets in {elapsed:.1f}s")
    print(f"  Skipped: {skipped_no_resolution:,} no resolution, {skipped_too_few:,} too few snapshots")

    return splits


def save_split(split_data: dict, feature_names: list[str], path: Path) -> int:
    """Convert feature dicts to numpy and save as .npz."""
    if not split_data["features"]:
        np.savez_compressed(
            path,
            X=np.empty((0, len(feature_names)), dtype=np.float32),
            y=np.empty(0, dtype=np.float32),
            market_ids=np.array([], dtype=object),
            timestamps=np.array([], dtype=object),
        )
        return 0

    X = np.array(
        [[f.get(name, 0.0) for name in feature_names] for f in split_data["features"]],
        dtype=np.float32,
    )
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.array(split_data["labels"], dtype=np.float32)

    np.savez_compressed(
        path,
        X=X,
        y=y,
        market_ids=np.array(split_data["market_ids"], dtype=object),
        timestamps=np.array(split_data["timestamps"], dtype=object),
    )
    return len(y)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pre-extract features from SQLite")
    parser.add_argument("--stride", type=int, default=6, help="Snapshot stride (1=all, 6=every 6h)")
    parser.add_argument("--top-tags", type=int, default=50, help="Number of top event tags to include")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}", file=sys.stderr)
        print("Run: uv run python scripts/download_exhaustive.py --fresh --workers 4", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Database: {DB_PATH}")
    print(f"Output: {OUT_DIR}")
    print(f"Stride: {args.stride}, Top tags: {args.top_tags}")
    print(f"Split cutoffs: train < {TRAIN_CUTOFF}, val < {VAL_CUTOFF}, test >= {VAL_CUTOFF}")

    conn = db.connect(DB_PATH)

    # Get event tags
    print("\nLoading event tags...")
    top_tags = get_top_tags(conn, args.top_tags)
    print(f"  Top {len(top_tags)} tags: {top_tags[:10]}...")
    market_tags = get_market_tags(conn)
    print(f"  Markets with tags: {len(market_tags):,}")

    # Build features and split
    print("\nExtracting features...")
    start = time.monotonic()
    splits = build_and_split(conn, args.stride, top_tags, market_tags)
    total_time = time.monotonic() - start

    conn.close()

    # Get feature names from first non-empty split
    feature_names: list[str] = []
    for name in ["train", "val", "test"]:
        if splits[name]["features"]:
            feature_names = sorted(splits[name]["features"][0].keys())
            break

    if not feature_names:
        print("ERROR: No features extracted — empty dataset", file=sys.stderr)
        sys.exit(1)

    # Save splits
    print("\nSaving splits...")
    sizes = {}
    for name in ["train", "val", "test"]:
        path = OUT_DIR / f"{name}.npz"
        n = save_split(splits[name], feature_names, path)
        size_mb = path.stat().st_size / 1024 / 1024
        sizes[name] = n
        n_markets = len(set(splits[name]["market_ids"]))
        print(f"  {name}: {n:,} samples, {n_markets:,} markets ({size_mb:.1f} MB)")

    # Save metadata
    meta = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "top_tags": top_tags,
        "n_top_tags": len(top_tags),
        "snapshot_stride": args.stride,
        "train_cutoff": TRAIN_CUTOFF,
        "val_cutoff": VAL_CUTOFF,
        "train_samples": sizes["train"],
        "val_samples": sizes["val"],
        "test_samples": sizes["test"],
        "total_samples": sum(sizes.values()),
        "build_time_seconds": round(total_time, 1),
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "db_path": str(DB_PATH),
    }
    meta_path = OUT_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata: {meta_path}")

    print(f"\nDone in {total_time:.1f}s ({total_time / 60:.1f}m)")
    print(f"Total: {sum(sizes.values()):,} samples, {len(feature_names)} features")


if __name__ == "__main__":
    main()
