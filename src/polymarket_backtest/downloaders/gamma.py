from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import httpx
import polars as pl

from ..types import isoformat

type MarketDict = dict[str, Any]

BASE_URL = "https://gamma-api.polymarket.com"
MARKETS_PATH = "/markets"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_PAGE_DELAY_SECONDS = 0.05
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.5
RESOLVED_HIGH_WATERMARK = 0.999
RESOLVED_LOW_WATERMARK = 0.001
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; polymarket-backtest/0.1)",
}


def fetch_resolved_markets(*, limit: int = 100, max_markets: int = 500) -> list[MarketDict]:
    """Fetch closed Gamma markets with offset pagination."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_markets <= 0:
        raise ValueError("max_markets must be positive")

    markets: list[MarketDict] = []
    offset = 0

    with _build_client() as client:
        while len(markets) < max_markets:
            page_size = min(limit, max_markets - len(markets))
            page = _request_markets_page(
                client,
                params={
                    "closed": "true",
                    "limit": str(page_size),
                    "offset": str(offset),
                },
            )
            if not page:
                break

            markets.extend(page)
            offset += len(page)

            if len(page) < page_size or len(markets) >= max_markets:
                break

            time.sleep(DEFAULT_PAGE_DELAY_SECONDS)

    return markets[:max_markets]


def parse_resolution(market: MarketDict) -> MarketDict | None:
    """Parse a resolved binary Gamma market into the backtest schema."""
    if not market.get("closed"):
        return None

    market_id = market.get("conditionId")
    title = market.get("question")
    resolution_ts = _parse_timestamp(market.get("closedTime"))

    if not isinstance(market_id, str) or not market_id:
        return None
    if not isinstance(title, str) or not title:
        return None
    if resolution_ts is None:
        return None

    outcomes = _parse_json_list(market.get("outcomes"))
    outcome_prices = _parse_price_list(market.get("outcomePrices"))
    if len(outcomes) != 2 or len(outcome_prices) != 2:
        return None

    normalized_outcomes = [str(outcome).strip().lower() for outcome in outcomes]
    if set(normalized_outcomes) != {"yes", "no"}:
        return None

    yes_index = normalized_outcomes.index("yes")
    no_index = normalized_outcomes.index("no")
    resolved_outcome = _resolve_binary_outcome(
        yes_price=outcome_prices[yes_index],
        no_price=outcome_prices[no_index],
    )
    if resolved_outcome is None:
        return None
    volume = _coerce_float(market.get("volume"))

    return {
        "market_id": market_id,
        "title": title,
        "domain": _derive_domain(market),
        "event_id": _extract_event_id(market),
        "tags": _extract_market_tags(market),
        "resolution_ts": resolution_ts,
        "resolved_outcome": resolved_outcome,
        "status": "resolved",
        "volume": 0.0 if volume is None else volume,
    }


def fetch_market_rules(condition_id: str) -> str:
    """Fetch the description field for a single market."""
    if not condition_id:
        raise ValueError("condition_id must be provided")

    with _build_client() as client:
        markets = _request_markets_page(
            client,
            params={
                "condition_ids": condition_id,
                "limit": "1",
            },
        )

    if not markets:
        return ""

    market = next(
        (candidate for candidate in markets if candidate.get("conditionId") == condition_id),
        markets[0],
    )
    description = market.get("description")
    if isinstance(description, str):
        return description

    events = market.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                event_description = event.get("description")
                if isinstance(event_description, str):
                    return event_description

    return ""


def markets_to_dataframe(markets: list[MarketDict]) -> pl.DataFrame:
    """Convert parsed markets into a Polars dataframe."""
    rows: list[MarketDict] = []
    for market in markets:
        parsed = market if "market_id" in market else parse_resolution(market)
        if parsed is not None:
            rows.append(
                {
                    "market_id": parsed["market_id"],
                    "title": parsed["title"],
                    "domain": parsed["domain"],
                    "event_id": parsed.get("event_id"),
                    "tags": json.dumps(parsed.get("tags", []), ensure_ascii=True, sort_keys=True),
                    "resolution_ts": parsed["resolution_ts"],
                    "resolved_outcome": parsed["resolved_outcome"],
                    "status": parsed["status"],
                    "volume": _coerce_float(parsed.get("volume")) or 0.0,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "market_id": pl.String,
                "title": pl.String,
                "domain": pl.String,
                "event_id": pl.String,
                "tags": pl.String,
                "resolution_ts": pl.String,
                "resolved_outcome": pl.Float64,
                "status": pl.String,
                "volume": pl.Float64,
            }
        )

    return pl.DataFrame(rows).select(
        "market_id",
        "title",
        "domain",
        "event_id",
        "tags",
        "resolution_ts",
        "resolved_outcome",
        "status",
        "volume",
    )


def _build_client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        headers=REQUEST_HEADERS,
        follow_redirects=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


def _request_markets_page(
    client: httpx.Client,
    *,
    params: dict[str, str],
) -> list[MarketDict]:
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(MARKETS_PATH, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_error = exc
            if not _should_retry(exc, attempt=attempt):
                break
            time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
            continue

        if not isinstance(payload, list):
            raise RuntimeError(f"Gamma API returned {type(payload).__name__}, expected list")

        return payload

    raise RuntimeError(f"Gamma API request failed after {MAX_RETRIES} attempts") from last_error


def _should_retry(exc: Exception, *, attempt: int) -> bool:
    if attempt >= MAX_RETRIES - 1:
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code in {403, 408, 429} or status_code >= 500
    return isinstance(exc, httpx.RequestError | json.JSONDecodeError)


def _parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_price_list(value: Any) -> list[float]:
    prices = _parse_json_list(value)
    result: list[float] = []
    for price in prices:
        coerced = _coerce_float(price)
        if coerced is None:
            return []
        result.append(coerced)
    return result


def _resolve_binary_outcome(*, yes_price: float, no_price: float) -> float | None:
    if yes_price >= RESOLVED_HIGH_WATERMARK and no_price <= RESOLVED_LOW_WATERMARK:
        return 1.0
    if no_price >= RESOLVED_HIGH_WATERMARK and yes_price <= RESOLVED_LOW_WATERMARK:
        return 0.0
    return None


def _derive_domain(market: MarketDict) -> str:
    for candidate in _iter_domain_candidates(market.get("tags")):
        return candidate

    category = market.get("category")
    if isinstance(category, str) and category.strip():
        return _normalize_domain(category)

    return "general"


def _extract_event_id(market: MarketDict) -> str | None:
    for key in ("event_id", "eventId"):
        candidate = market.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    events = market.get("events")
    if not isinstance(events, list):
        return None

    for event in events:
        if not isinstance(event, dict):
            continue
        for key in ("id", "event_id", "eventId", "slug"):
            candidate = event.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _extract_market_tags(market: MarketDict) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        if not isinstance(candidate, str):
            return
        tag = candidate.strip()
        if not tag or tag in seen:
            return
        seen.add(tag)
        tags.append(tag)

    raw_tags = market.get("tags")
    parsed_tags = _parse_json_list(raw_tags) if isinstance(raw_tags, str) else raw_tags
    if isinstance(parsed_tags, list):
        for item in parsed_tags:
            if isinstance(item, dict):
                for key in ("name", "label", "slug"):
                    if key in item:
                        _add(item.get(key))
                        break
            else:
                _add(item)

    category = market.get("category")
    _add(category)
    return tags


def _iter_domain_candidates(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = _parse_json_list(value)
        return _iter_domain_candidates(parsed) if parsed else [_normalize_domain(value)]

    if not isinstance(value, list):
        return []

    domains: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            domains.append(_normalize_domain(item))
            continue
        if isinstance(item, dict):
            for key in ("slug", "name", "label"):
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    domains.append(_normalize_domain(candidate))
                    break
    return domains


def _normalize_domain(value: str) -> str:
    return "-".join(value.strip().lower().split()) or "general"


def _parse_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return isoformat(timestamp)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None
