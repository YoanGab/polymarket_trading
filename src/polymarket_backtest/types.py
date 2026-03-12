from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal


UTC = timezone.utc


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def isoformat(value: datetime) -> str:
    return ensure_utc(value).isoformat(timespec="seconds")


@dataclass(frozen=True)
class OrderLevel:
    side: Literal["bid", "ask"]
    price: float
    quantity: float
    level_no: int


@dataclass(frozen=True)
class NewsItem:
    document_id: str
    source: str
    url: str
    title: str
    published_ts: datetime
    first_seen_ts: datetime
    ingested_ts: datetime
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketState:
    market_id: str
    title: str
    domain: str
    market_type: str
    ts: datetime
    status: str
    best_bid: float
    best_ask: float
    mid: float
    last_trade: float
    volume_1m: float
    volume_24h: float
    open_interest: float
    tick_size: float
    rules_text: str
    additional_context: str
    resolution_ts: datetime | None
    fees_enabled: bool
    fee_rate: float
    fee_exponent: float
    maker_rebate_rate: float
    orderbook: list[OrderLevel]

    @property
    def visible_depth_ask(self) -> float:
        return sum(level.quantity for level in self.orderbook if level.side == "ask")

    @property
    def visible_depth_bid(self) -> float:
        return sum(level.quantity for level in self.orderbook if level.side == "bid")

    @property
    def seconds_to_resolution(self) -> float | None:
        if self.resolution_ts is None:
            return None
        return max(0.0, (self.resolution_ts - self.ts).total_seconds())


@dataclass(frozen=True)
class ForecastInput:
    as_of: datetime
    market: MarketState
    recent_news: list[NewsItem]
    related_markets: list[dict[str, Any]]


@dataclass(frozen=True)
class ForecastOutput:
    agent_name: str
    model_id: str
    model_release: str
    as_of: datetime
    market_id: str
    domain: str
    probability_yes: float
    confidence: float
    expected_edge_bps: float
    thesis: str
    reasoning: str
    evidence: list[dict[str, Any]]
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    family: Literal["carry_only", "news_driven", "deep_research", "cross_market_arb"]
    kelly_fraction: float
    edge_threshold_bps: float
    max_position_notional: float
    max_holding_minutes: int | None
    use_time_stop: bool = False
    use_thesis_stop: bool = False
    thesis_stop_delta: float = 0.08
    aggressive_entry: bool = True
    carry_price_min: float = 0.95
    carry_price_max: float = 0.99
    min_confidence: float = 0.55


@dataclass(frozen=True)
class OrderIntent:
    strategy_name: str
    market_id: str
    ts: datetime
    side: Literal["buy", "sell"]
    liquidity_intent: Literal["aggressive", "passive"]
    limit_price: float
    requested_quantity: float
    kelly_fraction: float
    edge_bps: float
    holding_period_minutes: int | None
    thesis: str


@dataclass(frozen=True)
class FillResult:
    order_id: str
    market_id: str
    strategy_name: str
    fill_ts: datetime
    side: Literal["buy", "sell"]
    liquidity_role: Literal["maker", "taker"]
    price: float
    quantity: float
    fee_usdc: float
    rebate_usdc: float
    impact_bps: float
    fill_delay_seconds: float


@dataclass
class PositionState:
    strategy_name: str
    market_id: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    total_opened_quantity: float = 0.0
    opened_ts: datetime | None = None
    closed_ts: datetime | None = None
    entry_probability: float = 0.0
    thesis: str = ""
    realized_pnl: float = 0.0
    realized_pnl_pre_resolution: float = 0.0
    fees_paid: float = 0.0
    rebates_earned: float = 0.0

    def age(self, as_of: datetime) -> timedelta | None:
        if self.opened_ts is None:
            return None
        return ensure_utc(as_of) - ensure_utc(self.opened_ts)


@dataclass(frozen=True)
class ReplayConfig:
    experiment_name: str
    starting_cash: float
    lookback_minutes: int
    markout_horizons_min: tuple[int, ...] = (1, 5, 30, 240)
