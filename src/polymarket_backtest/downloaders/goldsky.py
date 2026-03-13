from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
import polars as pl

type FillDict = dict[str, Any]

logger = logging.getLogger(__name__)

GRAPHQL_ENDPOINT = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/polymarket-orderbook-resync/prod/gn"
)
DEFAULT_TIMEOUT_SECONDS = 20.0
FIXED_POINT_SCALE = Decimal("1000000")
REQUEST_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; polymarket-backtest/0.1)",
}
ORDER_FILLS_QUERY = """
query OrderFilledEvents($tokenId: BigInt!, $afterTimestamp: BigInt!, $first: Int!) {
  orderFilledEvents(
    first: $first
    orderBy: timestamp
        orderDirection: asc
        where: {
          or: [
        { makerAssetId: $tokenId, timestamp_gte: $afterTimestamp }
        { takerAssetId: $tokenId, timestamp_gte: $afterTimestamp }
      ]
    }
  ) {
    id
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
    fee
    timestamp
    transactionHash
  }
}
"""


def fetch_order_fills(
    *,
    token_id: str,
    after_timestamp: str = "0",
    limit: int = 1000,
    max_fills: int = 10000,
) -> list[FillDict]:
    """Fetch filled orders for a single Polymarket token using timestamp keyset pagination."""
    if not token_id:
        raise ValueError("token_id must be provided")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_fills <= 0:
        raise ValueError("max_fills must be positive")

    fills: list[FillDict] = []
    seen_ids: set[str] = set()
    last_timestamp = after_timestamp

    with _build_client() as client:
        while len(fills) < max_fills:
            page_size = min(limit, max_fills - len(fills))
            page = _request_order_fills_page(
                client,
                token_id=token_id,
                after_timestamp=last_timestamp,
                limit=page_size,
            )
            if not page:
                break

            new_page: list[FillDict] = []
            for fill in page:
                fill_id = _require_str(fill, "id")
                if fill_id in seen_ids:
                    continue
                seen_ids.add(fill_id)
                new_page.append(fill)
            fills.extend(_normalize_fill(fill, token_id=token_id, market_id=token_id) for fill in new_page)

            if len(page) < page_size or len(fills) >= max_fills:
                break

            next_timestamp = _require_str(page[-1], "timestamp")
            if _to_decimal(next_timestamp) < _to_decimal(last_timestamp):
                raise RuntimeError("Goldsky pagination regressed the timestamp cursor")
            if not new_page:
                logger.warning(
                    "Goldsky pagination stalled at timestamp=%s after ID deduplication; stopping early",
                    last_timestamp,
                )
                break
            last_timestamp = next_timestamp

    return fills[:max_fills]


def fetch_fills_for_market(
    *,
    token_ids: list[str],
    after_timestamp: str = "0",
    max_fills: int = 10000,
) -> list[FillDict]:
    """Fetch and merge fills for every token that belongs to a market."""
    unique_token_ids = [token_id for token_id in dict.fromkeys(token_ids) if token_id]
    if not unique_token_ids:
        raise ValueError("token_ids must contain at least one token id")
    if max_fills <= 0:
        raise ValueError("max_fills must be positive")

    market_id = _derive_market_id(unique_token_ids)
    merged: dict[str, FillDict] = {}

    for token_id in unique_token_ids:
        for fill in fetch_order_fills(
            token_id=token_id,
            after_timestamp=after_timestamp,
            max_fills=max_fills,
        ):
            normalized = dict(fill)
            normalized["market_id"] = market_id
            merged.setdefault(_require_str(normalized, "id"), normalized)

    return sorted(merged.values(), key=_fill_sort_key)[:max_fills]


def compute_price_from_fill(fill: FillDict) -> float:
    """Compute the token price implied by a fill, normalized to the [0, 1] range."""
    if "price" in fill:
        price = _to_decimal(fill["price"])
    else:
        token_amount = _extract_token_amount(fill)
        usd_amount = _extract_usd_amount_decimal(fill)
        if token_amount <= 0:
            raise ValueError("fill token amount must be positive")
        price = usd_amount / token_amount

    if not Decimal("0") <= price <= Decimal("1"):
        inverse = Decimal("0")
        if price != 0:
            inverse = Decimal("1") / price
        if Decimal("0") <= inverse <= Decimal("1"):
            price = inverse
        else:
            raise ValueError(f"fill price must be between 0 and 1, got {price}")

    return float(price)


def fills_to_dataframe(fills: list[FillDict]) -> pl.DataFrame:
    """Convert normalized fill dictionaries into a Polars dataframe."""
    if not fills:
        return pl.DataFrame(
            schema={
                "timestamp": pl.String,
                "market_id": pl.String,
                "price": pl.Float64,
                "usd_amount": pl.Float64,
                "direction": pl.String,
            }
        )

    rows = [
        {
            "timestamp": _require_str(fill, "timestamp"),
            "market_id": _market_id_for_fill(fill),
            "price": compute_price_from_fill(fill),
            "usd_amount": float(_extract_usd_amount_decimal(fill)),
            "direction": _direction_for_fill(fill),
        }
        for fill in sorted(fills, key=_fill_sort_key)
    ]
    return pl.DataFrame(rows).select("timestamp", "market_id", "price", "usd_amount", "direction")


def _build_client() -> httpx.Client:
    return httpx.Client(
        headers=REQUEST_HEADERS,
        follow_redirects=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


def _request_order_fills_page(
    client: httpx.Client,
    *,
    token_id: str,
    after_timestamp: str,
    limit: int,
) -> list[FillDict]:
    response = client.post(
        GRAPHQL_ENDPOINT,
        json={
            "query": ORDER_FILLS_QUERY,
            "variables": {
                "tokenId": token_id,
                "afterTimestamp": after_timestamp,
                "first": limit,
            },
        },
    )
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Goldsky returned {type(payload).__name__}, expected dict")

    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"Goldsky GraphQL returned errors: {errors}")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Goldsky GraphQL response is missing a data object")

    order_fills = data.get("orderFilledEvents")
    if not isinstance(order_fills, list):
        raise RuntimeError("Goldsky GraphQL response is missing orderFilledEvents")

    return order_fills


def _normalize_fill(raw_fill: FillDict, *, token_id: str, market_id: str) -> FillDict:
    maker_asset_id = _require_str(raw_fill, "makerAssetId")
    taker_asset_id = _require_str(raw_fill, "takerAssetId")
    direction = _direction_from_assets(token_id=token_id, maker_asset_id=maker_asset_id, taker_asset_id=taker_asset_id)
    token_amount_key = "makerAmountFilled" if direction == "sell" else "takerAmountFilled"
    usd_amount_key = "takerAmountFilled" if direction == "sell" else "makerAmountFilled"

    fill = {
        "id": _require_str(raw_fill, "id"),
        "maker": _require_str(raw_fill, "maker"),
        "taker": _require_str(raw_fill, "taker"),
        "makerAmountFilled": _require_str(raw_fill, "makerAmountFilled"),
        "takerAmountFilled": _require_str(raw_fill, "takerAmountFilled"),
        "fee": _require_str(raw_fill, "fee"),
        "timestamp": _require_str(raw_fill, "timestamp"),
        "transactionHash": _require_str(raw_fill, "transactionHash"),
        "makerAssetId": maker_asset_id,
        "takerAssetId": taker_asset_id,
        "token_id": token_id,
        "market_id": market_id,
        "direction": direction,
        "tokenAmountFilled": _require_str(raw_fill, token_amount_key),
        "usdAmountFilled": _require_str(raw_fill, usd_amount_key),
    }
    fill["price"] = compute_price_from_fill(fill)
    fill["usd_amount"] = float(_extract_usd_amount_decimal(fill))
    return fill


def _extract_token_amount(fill: FillDict) -> Decimal:
    if "tokenAmountFilled" in fill:
        return _to_decimal(fill["tokenAmountFilled"]) / FIXED_POINT_SCALE

    token_id = _token_id_for_fill(fill)
    direction = _direction_from_assets(
        token_id=token_id,
        maker_asset_id=_require_str(fill, "makerAssetId"),
        taker_asset_id=_require_str(fill, "takerAssetId"),
    )
    amount_key = "makerAmountFilled" if direction == "sell" else "takerAmountFilled"
    return _to_decimal(fill[amount_key]) / FIXED_POINT_SCALE


def _extract_usd_amount_decimal(fill: FillDict) -> Decimal:
    if "usdAmountFilled" in fill:
        return _to_decimal(fill["usdAmountFilled"]) / FIXED_POINT_SCALE

    token_id = _token_id_for_fill(fill)
    direction = _direction_from_assets(
        token_id=token_id,
        maker_asset_id=_require_str(fill, "makerAssetId"),
        taker_asset_id=_require_str(fill, "takerAssetId"),
    )
    amount_key = "takerAmountFilled" if direction == "sell" else "makerAmountFilled"
    return _to_decimal(fill[amount_key]) / FIXED_POINT_SCALE


def _token_id_for_fill(fill: FillDict) -> str:
    token_id = fill.get("token_id")
    if isinstance(token_id, str) and token_id:
        return token_id
    raise ValueError("fill is missing token_id metadata")


def _market_id_for_fill(fill: FillDict) -> str:
    market_id = fill.get("market_id")
    if isinstance(market_id, str) and market_id:
        return market_id
    return _token_id_for_fill(fill)


def _direction_for_fill(fill: FillDict) -> str:
    direction = fill.get("direction")
    if direction in {"buy", "sell"}:
        return str(direction)

    return _direction_from_assets(
        token_id=_token_id_for_fill(fill),
        maker_asset_id=_require_str(fill, "makerAssetId"),
        taker_asset_id=_require_str(fill, "takerAssetId"),
    )


def _direction_from_assets(*, token_id: str, maker_asset_id: str, taker_asset_id: str) -> str:
    if maker_asset_id == token_id:
        return "sell"
    if taker_asset_id == token_id:
        return "buy"
    raise ValueError(f"token_id {token_id!r} is not present in fill assets")


def _derive_market_id(token_ids: list[str]) -> str:
    return "|".join(sorted(token_ids))


def _fill_sort_key(fill: FillDict) -> tuple[int, str]:
    return (int(_require_str(fill, "timestamp")), _require_str(fill, "id"))


def _require_str(payload: FillDict, key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str) and value:
        return value
    if value is None:
        raise ValueError(f"missing required field {key!r}")
    return str(value)


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"could not parse decimal value: {value!r}") from exc
