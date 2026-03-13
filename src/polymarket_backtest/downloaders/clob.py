from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import polars as pl

from ..types import isoformat

type PricePoint = dict[str, int | float]

CLOB_BASE_URL = "https://clob.polymarket.com"
PRICES_HISTORY_PATH = "/prices-history"
DEFAULT_INTERVAL = "max"
DEFAULT_FIDELITY = 60  # minutes — 60 = hourly candles
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=20.0)
DEFAULT_REQUEST_DELAY_SECONDS = 0.1
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.5
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; polymarket-backtest/0.1)",
}


def fetch_price_history(
    *,
    token_id: str,
    interval: str = DEFAULT_INTERVAL,
    fidelity: int = DEFAULT_FIDELITY,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[PricePoint]:
    """Fetch CLOB price history for a single outcome token.

    For resolved/closed markets, use ``start_ts``/``end_ts`` (unix timestamps)
    instead of ``interval`` — the CLOB API only returns data relative to *now*
    when using ``interval``, so resolved markets return 0 points.

    ``fidelity`` is granularity in **minutes**: 1 = per-minute, 60 = hourly.
    """
    token = _validate_token_id(token_id)
    if fidelity <= 0:
        raise ValueError("fidelity must be positive")

    with _build_client() as client:
        if start_ts is not None and end_ts is not None:
            return _request_price_history_range(
                client, token_id=token, start_ts=start_ts, end_ts=end_ts, fidelity=fidelity
            )
        normalized_interval = _validate_interval(interval)
        return _request_price_history(client, token_id=token, interval=normalized_interval, fidelity=fidelity)


def fetch_market_price_histories(
    *,
    token_ids: list[str],
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, list[PricePoint]]:
    """Fetch price histories for multiple outcome tokens."""

    if not token_ids:
        return {}

    normalized_interval = _validate_interval(interval)
    normalized_tokens = [_validate_token_id(token_id) for token_id in token_ids]

    histories: dict[str, list[PricePoint]] = {}
    with _build_client() as client:
        for index, token_id in enumerate(normalized_tokens):
            histories[token_id] = _request_price_history(
                client,
                token_id=token_id,
                interval=normalized_interval,
                fidelity=DEFAULT_FIDELITY,
            )
            if index < len(normalized_tokens) - 1:
                time.sleep(DEFAULT_REQUEST_DELAY_SECONDS)

    return histories


def _estimate_half_spread(price: float) -> float:
    """Estimate half-spread based on price level.

    More realistic spread estimation: wider spreads in the middle range
    where liquidity is typically thinner, narrower near extremes where
    prices are more certain.
    """
    if price <= 0.10 or price >= 0.90:
        return 0.01  # 200 bps near extremes
    if price <= 0.20 or price >= 0.80:
        return 0.015  # 300 bps
    return 0.02  # 400 bps in the middle range


def price_history_to_snapshots(
    token_id: str,
    history: list[dict[str, Any]],
    *,
    spread_override: float | None = None,
) -> list[dict[str, Any]]:
    """Convert normalized price history into market snapshot rows.

    Args:
        token_id: The CLOB token ID.
        history: Raw price history points from the CLOB API.
        spread_override: If provided, use this fixed half-spread instead of the
            price-dependent estimate.

    Note:
        volume_1m and volume_24h are set to NaN because the CLOB price history
        API does not provide volume data per candle. Callers that need volume
        should enrich snapshots from another source (e.g. Gamma API volume24hr).
    """

    market_id = _validate_token_id(token_id)
    snapshots: list[dict[str, Any]] = []

    for point in history:
        normalized = _normalize_price_point(point)
        if normalized is None:
            continue

        price = float(normalized["price"])
        ts = int(normalized["timestamp"])

        half_spread = spread_override if spread_override is not None else _estimate_half_spread(price)

        snapshots.append(
            {
                "market_id": market_id,
                "ts": isoformat(datetime.fromtimestamp(ts, tz=UTC)),
                "status": "active",
                "best_bid": _clamp_price(price - half_spread),
                "best_ask": _clamp_price(price + half_spread),
                "last_trade": _clamp_price(price),
                "volume_1m": float("nan"),
                "volume_24h": float("nan"),
                "open_interest": 0.0,
                "tick_size": 0.01,
            }
        )

    return snapshots


def price_histories_to_dataframe(histories: dict[str, list[dict[str, Any]]]) -> pl.DataFrame:
    """Convert token price histories into one Polars dataframe."""

    rows: list[dict[str, Any]] = []
    for token_id, history in histories.items():
        market_id = _validate_token_id(token_id)
        for point in history:
            normalized = _normalize_price_point(point)
            if normalized is None:
                continue

            ts = int(normalized["timestamp"])
            rows.append(
                {
                    "market_id": market_id,
                    "timestamp": ts,
                    "ts": isoformat(datetime.fromtimestamp(ts, tz=UTC)),
                    "price": float(normalized["price"]),
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "market_id": pl.String,
                "timestamp": pl.Int64,
                "ts": pl.String,
                "price": pl.Float64,
            }
        )

    return pl.DataFrame(rows).sort(["market_id", "timestamp"]).select("market_id", "timestamp", "ts", "price")


def _build_client() -> httpx.Client:
    return httpx.Client(
        base_url=CLOB_BASE_URL,
        headers=REQUEST_HEADERS,
        follow_redirects=True,
        timeout=DEFAULT_TIMEOUT,
    )


def _request_price_history_range(
    client: httpx.Client,
    *,
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int,
    chunk_days: int = 7,
) -> list[PricePoint]:
    """Fetch price history using startTs/endTs, chunking into windows to avoid gaps."""
    all_points: list[PricePoint] = []
    chunk_seconds = chunk_days * 86400
    cursor = start_ts

    while cursor < end_ts:
        chunk_end = min(cursor + chunk_seconds, end_ts)
        params = {
            "market": token_id,
            "startTs": str(cursor),
            "endTs": str(chunk_end),
            "fidelity": str(fidelity),
        }
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.get(PRICES_HISTORY_PATH, params=params)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                if not _should_retry(exc, attempt=attempt):
                    break
                time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            all_points.extend(_parse_price_history_payload(payload))
            last_error = None
            break

        if last_error is not None:
            break  # stop chunking on persistent failure
        cursor = chunk_end
        if cursor < end_ts:
            time.sleep(DEFAULT_REQUEST_DELAY_SECONDS)

    # Deduplicate by timestamp, keep last
    seen: dict[int, PricePoint] = {}
    for pt in all_points:
        seen[int(pt["timestamp"])] = pt
    return sorted(seen.values(), key=lambda p: int(p["timestamp"]))


def _request_price_history(
    client: httpx.Client,
    *,
    token_id: str,
    interval: str,
    fidelity: int,
) -> list[PricePoint]:
    last_error: Exception | None = None
    params = {
        "market": token_id,
        "interval": interval,
        "fidelity": str(fidelity),
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(PRICES_HISTORY_PATH, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_error = exc
            if not _should_retry(exc, attempt=attempt):
                break
            time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
            continue

        return _parse_price_history_payload(payload)

    raise RuntimeError(
        f"CLOB price history request failed after {MAX_RETRIES} attempts for token {token_id}"
    ) from last_error


def _parse_price_history_payload(payload: Any) -> list[PricePoint]:
    if payload is None:
        return []

    raw_history: Any = payload
    if isinstance(payload, dict):
        if "history" not in payload:
            if not payload:
                return []
            raise RuntimeError("CLOB price history payload did not include a history field")
        raw_history = payload.get("history")

    if raw_history is None:
        return []
    if not isinstance(raw_history, list):
        raise RuntimeError(f"CLOB price history payload returned {type(raw_history).__name__}, expected list")

    points: list[PricePoint] = []
    for item in raw_history:
        normalized = _normalize_price_point(item)
        if normalized is not None:
            points.append(normalized)

    points.sort(key=lambda point: int(point["timestamp"]))
    return points


def _normalize_price_point(value: Any) -> PricePoint | None:
    if not isinstance(value, dict):
        return None

    timestamp = _coerce_int(value.get("t", value.get("timestamp")))
    price = _coerce_float(value.get("p", value.get("price")))
    if timestamp is None or price is None:
        return None

    return {
        "timestamp": timestamp,
        "price": price,
    }


def _should_retry(exc: Exception, *, attempt: int) -> bool:
    if attempt >= MAX_RETRIES - 1:
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code in {403, 408, 429} or status_code >= 500
    return isinstance(exc, httpx.RequestError | json.JSONDecodeError)


def _validate_interval(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("interval must be a non-empty string")
    return value.strip()


def _validate_token_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("token_id must be a non-empty string")
    return value.strip()


def _clamp_price(value: float) -> float:
    return max(0.0, min(1.0, value))


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    return None
