from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

from .db import bulk_add_snapshots, get_market_ids
from .types import ensure_utc

PRICE_FLOOR = 0.001
PRICE_CEILING = 0.999

SNAPSHOT_COLUMNS = (
    "market_id",
    "ts",
    "status",
    "best_bid",
    "best_ask",
    "last_trade",
    "volume_1m",
    "volume_24h",
    "open_interest",
    "tick_size",
)


def build_regular_snapshots(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    interval_minutes: int = 5,
) -> int:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero")

    snapshots = _load_snapshots(conn, market_id)
    if not snapshots:
        return 0

    interval = timedelta(minutes=interval_minutes)
    bucketed: dict[datetime, dict[str, Any]] = {}

    for snapshot in snapshots:
        bucket_ts = _floor_to_interval(snapshot["ts"], interval_minutes)
        bucketed[bucket_ts] = _replace_ts(snapshot, bucket_ts)

    regular_snapshots: list[dict[str, Any]] = []
    last_seen: dict[str, Any] | None = None
    current_ts = min(bucketed)
    end_ts = max(bucketed)

    while current_ts <= end_ts:
        source = bucketed.get(current_ts)
        if source is not None:
            last_seen = source
            regular_snapshots.append({**source, "interpolated": False, "forward_filled": False})
        elif last_seen is not None:
            regular_snapshots.append(
                {**_replace_ts(last_seen, current_ts), "interpolated": False, "forward_filled": True}
            )
        current_ts += interval

    if not regular_snapshots:
        return 0

    return bulk_add_snapshots(conn, regular_snapshots)


def build_all_snapshots(
    conn: sqlite3.Connection,
    *,
    interval_minutes: int = 5,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for market_id in get_market_ids(conn):
        counts[market_id] = build_regular_snapshots(
            conn,
            market_id=market_id,
            interval_minutes=interval_minutes,
        )
    return counts


def derive_bid_ask_from_trades(
    trades: list[dict[str, Any]],
    spread_bps: int = 100,
) -> tuple[float, float]:
    if spread_bps < 0:
        raise ValueError("spread_bps must be non-negative")

    last_price: float | None = None
    for trade in reversed(trades):
        raw_price = trade.get("price")
        if raw_price is None:
            continue
        last_price = float(raw_price)
        break

    if last_price is None:
        raise ValueError("trades must contain at least one trade with a price")

    half_spread = spread_bps / 10000.0 / 2.0
    bid = _clamp(last_price - half_spread, PRICE_FLOOR, PRICE_CEILING)
    ask = _clamp(last_price + half_spread, PRICE_FLOOR, PRICE_CEILING)
    return bid, ask


def fill_snapshot_gaps(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    max_gap_minutes: int = 60,
) -> int:
    if max_gap_minutes <= 0:
        raise ValueError("max_gap_minutes must be greater than zero")

    snapshots = _load_snapshots(conn, market_id)
    if len(snapshots) < 2:
        return 0

    gap_limit = timedelta(minutes=max_gap_minutes)
    filled_snapshots: list[dict[str, Any]] = []

    for current, nxt in pairwise(snapshots):
        current_ts = current["ts"]
        next_ts = nxt["ts"]
        gap = next_ts - current_ts

        if gap <= gap_limit:
            continue

        insert_ts = current_ts + gap_limit
        while insert_ts < next_ts:
            fraction = (insert_ts - current_ts).total_seconds() / gap.total_seconds()
            snap = _interpolate_snapshot(current, nxt, insert_ts, fraction)
            snap["interpolated"] = True
            snap["forward_filled"] = False
            filled_snapshots.append(snap)
            insert_ts += gap_limit

    if not filled_snapshots:
        return 0

    return bulk_add_snapshots(conn, filled_snapshots)


def _load_snapshots(conn: sqlite3.Connection, market_id: str) -> list[dict[str, Any]]:
    cursor = conn.execute(
        f"""
        SELECT {", ".join(SNAPSHOT_COLUMNS)}
        FROM market_snapshots
        WHERE market_id = ?
        ORDER BY ts ASC
        """,
        (market_id,),
    )
    rows = cursor.fetchall()
    if not rows:
        return []

    snapshots: list[dict[str, Any]] = []
    for row in rows:
        snapshot = dict(zip(SNAPSHOT_COLUMNS, row, strict=True))
        snapshot["market_id"] = str(snapshot["market_id"])
        snapshot["ts"] = ensure_utc(datetime.fromisoformat(str(snapshot["ts"])))
        snapshot["status"] = str(snapshot["status"])
        for field_name in SNAPSHOT_COLUMNS:
            if field_name in {"market_id", "ts", "status"}:
                continue
            snapshot[field_name] = float(snapshot[field_name])
        snapshots.append(snapshot)

    return snapshots


def _replace_ts(snapshot: dict[str, Any], ts: datetime) -> dict[str, Any]:
    return {
        **snapshot,
        "ts": ensure_utc(ts),
    }


def _floor_to_interval(ts: datetime, interval_minutes: int) -> datetime:
    normalized = ensure_utc(ts)
    interval_seconds = interval_minutes * 60
    floor_epoch = int(normalized.timestamp() // interval_seconds * interval_seconds)
    return datetime.fromtimestamp(floor_epoch, tz=UTC)


def _interpolate_snapshot(
    start: dict[str, Any],
    end: dict[str, Any],
    ts: datetime,
    fraction: float,
) -> dict[str, Any]:
    return {
        "market_id": start["market_id"],
        "ts": ensure_utc(ts),
        "status": start["status"],
        "best_bid": _interpolate_float(start["best_bid"], end["best_bid"], fraction),
        "best_ask": _interpolate_float(start["best_ask"], end["best_ask"], fraction),
        "last_trade": _interpolate_float(start["last_trade"], end["last_trade"], fraction),
        "volume_1m": _interpolate_float(start["volume_1m"], end["volume_1m"], fraction),
        "volume_24h": _interpolate_float(start["volume_24h"], end["volume_24h"], fraction),
        "open_interest": _interpolate_float(start["open_interest"], end["open_interest"], fraction),
        "tick_size": start["tick_size"],
    }


def _interpolate_float(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
