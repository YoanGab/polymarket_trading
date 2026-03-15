from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

CRYPTO_CATEGORY_TAGS: tuple[str, ...] = ("Crypto", "Crypto Prices")
# Tags used for general sports category routing (strategy grid).
SPORTS_CATEGORY_TAGS: tuple[str, ...] = (
    "Sports",
    "NBA",
    "Soccer",
    "NFL",
    "NHL",
    "NCAAB",
    "Serie A",
)
# Only NCAAB and Serie A have fees per Polymarket docs (2025-2026).
# The DB uses "NCAA", "NCAA Basketball", "ncaab" for NCAAB markets.
SPORTS_FEE_BEARING_TAGS: tuple[str, ...] = (
    "NCAAB",
    "NCAA",
    "NCAA Basketball",
    "ncaab",
    "Serie A",
)
FEE_BEARING_CATEGORY_TAGS: tuple[str, ...] = tuple(dict.fromkeys(CRYPTO_CATEGORY_TAGS + SPORTS_FEE_BEARING_TAGS))


@dataclass(frozen=True)
class FeeSettings:
    """Per-market fee configuration matching Polymarket's on-chain parameters."""

    fees_enabled: bool
    fee_rate: float
    fee_exponent: float
    maker_rebate_rate: float


# Polymarket fee parameters per docs.polymarket.com/trading/fees:
#   fee = C * p * feeRate * (p * (1-p))^exponent
#   Crypto:  feeRate=0.25,   exponent=2, maker rebate=20%
#   Sports:  feeRate=0.0175, exponent=1, maker rebate=25%
_CRYPTO_FEE = FeeSettings(fees_enabled=True, fee_rate=0.25, fee_exponent=2.0, maker_rebate_rate=0.20)
_SPORTS_FEE = FeeSettings(fees_enabled=True, fee_rate=0.0175, fee_exponent=1.0, maker_rebate_rate=0.25)
_NO_FEE = FeeSettings(fees_enabled=False, fee_rate=0.0, fee_exponent=0.0, maker_rebate_rate=0.0)


def normalize_market_tags(tags: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags or ():
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def has_any_category(market_tags: Iterable[str], categories: Iterable[str]) -> bool:
    tag_set = set(normalize_market_tags(market_tags))
    return any(category in tag_set for category in categories)


def category_fee_settings(market_tags: Iterable[str]) -> FeeSettings:
    """Return fee parameters for a market based on its tags.

    Polymarket fee formula: fee = C * p * feeRate * (p * (1-p))^exponent
    where C = shares traded, p = share price.

    Returns:
        FeeSettings with the correct fee_rate, exponent, and maker rebate
        for the market's category.
    """
    tags = normalize_market_tags(market_tags)
    if has_any_category(tags, CRYPTO_CATEGORY_TAGS):
        return _CRYPTO_FEE
    if has_any_category(tags, SPORTS_FEE_BEARING_TAGS):
        return _SPORTS_FEE
    return _NO_FEE
