"""Feature extraction from market snapshots for ML models.

Extracts features from the SQLite database and creates train/val/test splits
using walk-forward validation (time-based, no leakage).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import db


@dataclass
class FeatureSet:
    """A labeled dataset ready for model training."""

    X: np.ndarray  # (n_samples, n_features)
    y: np.ndarray  # (n_samples,) — resolved outcome (0 or 1)
    market_ids: list[str]  # market_id for each sample
    timestamps: list[str]  # snapshot timestamp for each sample
    feature_names: list[str]  # column names


def extract_snapshot_features(row: sqlite3.Row, prev_rows: list[sqlite3.Row]) -> dict[str, float]:
    """Extract features from a single snapshot + history of prior snapshots for same market."""
    mid = float(row["mid"])
    best_bid = float(row["best_bid"])
    best_ask = float(row["best_ask"])
    last_trade = float(row["last_trade"])
    volume_1m = float(row["volume_1m"])
    volume_24h = float(row["volume_24h"])
    open_interest = float(row["open_interest"])

    spread = best_ask - best_bid
    spread_pct = spread / mid if mid > 0 else 0.0

    features: dict[str, float] = {
        # Price features
        "mid": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "last_trade": last_trade,
        "spread": spread,
        "spread_pct": spread_pct,
        "price_vs_half": abs(mid - 0.5),  # distance from maximum uncertainty
        "price_extreme": max(mid, 1 - mid),  # how extreme the price is
        # Volume features
        "volume_1m": volume_1m,
        "volume_24h": volume_24h,
        "open_interest": open_interest,
        "volume_oi_ratio": volume_24h / max(open_interest, 1.0),
        # Trend features (mid vs last_trade)
        "trend": mid - last_trade,
        "trend_pct": (mid - last_trade) / max(mid, 0.001),
        # Targeted interactions for extreme price scenarios
        "extreme_x_spread": abs(mid - 0.5) * spread_pct,  # extreme price + wide spread = uncertainty
        "extreme_x_volume": abs(mid - 0.5) * (volume_1m / max(volume_24h / 24.0, 1.0)),  # extreme + volume spike
    }

    # Resolution proximity
    resolution_ts_str = row["resolution_ts"] if "resolution_ts" in row.keys() else None
    ts_str = row["ts"]
    if resolution_ts_str and ts_str:
        try:
            res_dt = datetime.fromisoformat(str(resolution_ts_str).replace("Z", "+00:00"))
            snap_dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            hours_to_res = max(0, (res_dt - snap_dt).total_seconds() / 3600)
            features["hours_to_resolution"] = hours_to_res
            features["log_hours_to_resolution"] = np.log1p(hours_to_res)
            features["resolution_proximity"] = 1.0 / (1.0 + hours_to_res)
        except (ValueError, TypeError):
            features["hours_to_resolution"] = 720.0
            features["log_hours_to_resolution"] = np.log1p(720.0)
            features["resolution_proximity"] = 0.0
    else:
        features["hours_to_resolution"] = 720.0
        features["log_hours_to_resolution"] = np.log1p(720.0)
        features["resolution_proximity"] = 0.0

    # Momentum from history
    if prev_rows:
        prev_mids = [float(r["mid"]) for r in prev_rows]

        # Price change over last N snapshots
        for lookback in [3, 6, 12, 24]:
            if len(prev_mids) >= lookback:
                old_mid = prev_mids[-lookback]
                features[f"momentum_{lookback}h"] = mid - old_mid
                features[f"momentum_{lookback}h_pct"] = (mid - old_mid) / max(old_mid, 0.001)
            else:
                features[f"momentum_{lookback}h"] = 0.0
                features[f"momentum_{lookback}h_pct"] = 0.0

        # Volatility (std of recent mids)
        recent = prev_mids[-min(24, len(prev_mids)) :]
        features["volatility_24h"] = float(np.std(recent)) if len(recent) > 1 else 0.0

        # Volume trend
        prev_volumes = [float(r["volume_24h"]) for r in prev_rows]
        if len(prev_volumes) >= 6:
            recent_vol = np.mean(prev_volumes[-3:])
            older_vol = np.mean(prev_volumes[-6:-3])
            features["volume_trend"] = (recent_vol - older_vol) / max(older_vol, 1.0)
        else:
            features["volume_trend"] = 0.0

        # Price range (high-low over history)
        features["price_range_24h"] = max(prev_mids[-24:]) - min(prev_mids[-24:]) if len(prev_mids) >= 24 else 0.0
    else:
        for lookback in [3, 6, 12, 24]:
            features[f"momentum_{lookback}h"] = 0.0
            features[f"momentum_{lookback}h_pct"] = 0.0
        features["volatility_24h"] = 0.0
        features["volume_trend"] = 0.0
        features["price_range_24h"] = 0.0

    return features


def build_dataset(db_path: str | Path, snapshot_stride: int = 6) -> FeatureSet:
    """Build feature matrix from all markets in the database.

    Args:
        db_path: Path to SQLite database.
        snapshot_stride: Use every Nth snapshot per market (1=all, 6=every 6h).
            Reduces dataset size while keeping temporal coverage.

    Returns:
        FeatureSet with features, labels, and metadata.
    """
    conn = db.connect(db_path)
    market_ids = db.get_market_ids(conn)

    all_features: list[dict[str, float]] = []
    all_labels: list[float] = []
    all_market_ids: list[str] = []
    all_timestamps: list[str] = []

    for market_id in market_ids:
        # Get resolution
        resolution = db.get_resolution(conn, market_id)
        if resolution is None:
            continue
        resolved_outcome = float(resolution["resolved_outcome"])

        # Get all snapshots for this market
        rows = conn.execute(
            """
            SELECT s.*, m.resolution_ts
            FROM market_snapshots s
            JOIN markets m ON m.market_id = s.market_id
            WHERE s.market_id = ?
            ORDER BY s.ts ASC
            """,
            (market_id,),
        ).fetchall()

        if len(rows) < 10:
            continue

        # Sample snapshots with stride, keeping first and last
        sampled_indices = list(range(0, len(rows), snapshot_stride))
        if sampled_indices[-1] != len(rows) - 1:
            sampled_indices.append(len(rows) - 1)

        for idx in sampled_indices:
            row = rows[idx]
            prev_rows = rows[max(0, idx - 24) : idx]  # up to 24h history
            features = extract_snapshot_features(row, prev_rows)

            all_features.append(features)
            all_labels.append(resolved_outcome)
            all_market_ids.append(market_id)
            all_timestamps.append(str(row["ts"]))

    conn.close()

    if not all_features:
        feature_names: list[str] = []
        return FeatureSet(
            X=np.empty((0, 0)),
            y=np.empty(0),
            market_ids=[],
            timestamps=[],
            feature_names=feature_names,
        )

    # Convert to numpy
    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    return FeatureSet(
        X=X,
        y=y,
        market_ids=all_market_ids,
        timestamps=all_timestamps,
        feature_names=feature_names,
    )


@dataclass
class WalkForwardSplit:
    """A single fold of walk-forward validation."""

    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    train_market_ids: list[str]
    val_market_ids: list[str]
    test_market_ids: list[str]


def walk_forward_split(
    dataset: FeatureSet,
    train_cutoff: str = "2025-10-01",
    val_cutoff: str = "2026-01-01",
) -> WalkForwardSplit:
    """Split dataset chronologically by market resolution time.

    Fixed date cutoffs (not percentage-based) for reproducibility:
    - Train: markets resolved before train_cutoff (≤ Q3 2025) — ~63K markets
    - Validation: markets resolved between train_cutoff and val_cutoff (Q4 2025) — ~44K markets
    - Test: markets resolved after val_cutoff (2026+) — ~74K markets
    No data leakage — test markets resolved after all training markets.
    """
    # Group by market_id, find last timestamp per market
    market_last_ts: dict[str, str] = {}
    for mid, ts in zip(dataset.market_ids, dataset.timestamps):
        if mid not in market_last_ts or ts > market_last_ts[mid]:
            market_last_ts[mid] = ts

    # Split by fixed date cutoffs
    train_set: set[str] = set()
    val_set: set[str] = set()
    test_set: set[str] = set()
    for mid, ts in market_last_ts.items():
        if ts < train_cutoff:
            train_set.add(mid)
        elif ts < val_cutoff:
            val_set.add(mid)
        else:
            test_set.add(mid)

    # Build index masks
    train_mask = [mid in train_set for mid in dataset.market_ids]
    val_mask = [mid in val_set for mid in dataset.market_ids]
    test_mask = [mid in test_set for mid in dataset.market_ids]

    return WalkForwardSplit(
        train_X=dataset.X[train_mask],
        train_y=dataset.y[train_mask],
        val_X=dataset.X[val_mask],
        val_y=dataset.y[val_mask],
        test_X=dataset.X[test_mask],
        test_y=dataset.y[test_mask],
        train_market_ids=[mid for mid, m in zip(dataset.market_ids, train_mask) if m],
        val_market_ids=[mid for mid, m in zip(dataset.market_ids, val_mask) if m],
        test_market_ids=[mid for mid, m in zip(dataset.market_ids, test_mask) if m],
    )
