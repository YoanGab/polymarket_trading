from __future__ import annotations

from collections.abc import Iterable

CRYPTO_CATEGORY_TAGS: tuple[str, ...] = ("Crypto", "Crypto Prices")
SPORTS_CATEGORY_TAGS: tuple[str, ...] = (
    "Sports",
    "NBA",
    "Soccer",
    "NFL",
    "NHL",
    "NCAAB",
    "Serie A",
)
FEE_BEARING_CATEGORY_TAGS: tuple[str, ...] = tuple(dict.fromkeys(CRYPTO_CATEGORY_TAGS + SPORTS_CATEGORY_TAGS))


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


def category_fee_settings(market_tags: Iterable[str]) -> tuple[bool, float]:
    tags = normalize_market_tags(market_tags)
    if has_any_category(tags, CRYPTO_CATEGORY_TAGS):
        return True, 0.0025
    if has_any_category(tags, SPORTS_CATEGORY_TAGS):
        return True, 0.000175
    return False, 0.0
