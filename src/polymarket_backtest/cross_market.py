from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketMetadata:
    market_id: str
    title: str
    event_id: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class RelatedMarket:
    market_id: str
    correlation: float


class CrossMarketTracker:
    """Track event-linked markets and their historical price correlations."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._metadata_by_market: dict[str, MarketMetadata] = {}
        self._markets_by_event: dict[str, list[str]] = {}
        self._correlations_by_event: dict[str, dict[str, list[RelatedMarket]]] = {}
        self._load_metadata()

    def get_market_metadata(self, market_id: str) -> MarketMetadata | None:
        return self._metadata_by_market.get(market_id)

    def get_related_markets(self, market_id: str) -> list[RelatedMarket]:
        metadata = self._metadata_by_market.get(market_id)
        if metadata is None or not metadata.event_id:
            return []
        event_id = metadata.event_id
        if event_id not in self._correlations_by_event:
            self._correlations_by_event[event_id] = self._compute_event_correlations(event_id)
        return list(self._correlations_by_event[event_id].get(market_id, []))

    def _load_metadata(self) -> None:
        rows = self.conn.execute(
            """
            SELECT market_id, title, event_id, tags_json
            FROM markets
            ORDER BY market_id
            """
        ).fetchall()
        for row in rows:
            event_id = str(row["event_id"]).strip() if row["event_id"] else None
            metadata = MarketMetadata(
                market_id=str(row["market_id"]),
                title=str(row["title"]),
                event_id=event_id,
                tags=tuple(_parse_tags_json(row["tags_json"])),
            )
            self._metadata_by_market[metadata.market_id] = metadata
            if event_id:
                self._markets_by_event.setdefault(event_id, []).append(metadata.market_id)

    def _compute_event_correlations(self, event_id: str) -> dict[str, list[RelatedMarket]]:
        market_ids = list(self._markets_by_event.get(event_id, []))
        correlations = {market_id: [] for market_id in market_ids}
        if len(market_ids) < 2:
            return correlations

        placeholders = ", ".join("?" for _ in market_ids)
        rows = self.conn.execute(
            f"""
            SELECT market_id, ts, mid
            FROM market_snapshots
            WHERE market_id IN ({placeholders})
            ORDER BY ts, market_id
            """,
            tuple(market_ids),
        ).fetchall()
        if not rows:
            return correlations

        snapshots_by_market: dict[str, list[tuple[str, float]]] = {market_id: [] for market_id in market_ids}
        union_timestamps: list[str] = []
        seen_timestamps: set[str] = set()

        for row in rows:
            market_id = str(row["market_id"])
            ts = str(row["ts"])
            snapshots_by_market.setdefault(market_id, []).append((ts, float(row["mid"])))
            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                union_timestamps.append(ts)

        aligned_series = {
            market_id: _forward_fill_series(series, union_timestamps)
            for market_id, series in snapshots_by_market.items()
        }

        for index, market_id in enumerate(market_ids):
            left_series = aligned_series.get(market_id, {})
            for related_market_id in market_ids[index + 1 :]:
                right_series = aligned_series.get(related_market_id, {})
                common_timestamps = sorted(left_series.keys() & right_series.keys())
                left_prices = [left_series[ts] for ts in common_timestamps]
                right_prices = [right_series[ts] for ts in common_timestamps]
                correlation = _pearson_correlation(left_prices, right_prices)
                correlations[market_id].append(
                    RelatedMarket(market_id=related_market_id, correlation=correlation)
                )
                correlations[related_market_id].append(
                    RelatedMarket(market_id=market_id, correlation=correlation)
                )

        for market_id in correlations:
            correlations[market_id].sort(key=lambda item: (-abs(item.correlation), item.market_id))
        return correlations


def _forward_fill_series(
    series: list[tuple[str, float]],
    union_timestamps: list[str],
) -> dict[str, float]:
    if not series:
        return {}

    filled: dict[str, float] = {}
    current_value: float | None = None
    series_index = 0
    for ts in union_timestamps:
        while series_index < len(series) and series[series_index][0] <= ts:
            current_value = series[series_index][1]
            series_index += 1
        if current_value is not None:
            filled[ts] = current_value
    return filled


def _pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0

    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum((l - left_mean) * (r - right_mean) for l, r in zip(left, right, strict=True))
    left_variance = sum((l - left_mean) ** 2 for l in left)
    right_variance = sum((r - right_mean) ** 2 for r in right)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, covariance / denominator))


def _parse_tags_json(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        tag = str(item).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags
