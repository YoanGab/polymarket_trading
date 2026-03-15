import dataclasses as _dc
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

UTC = UTC


def dc_replace[T](obj: T, **kwargs: Any) -> T:
    """Typed wrapper around ``dataclasses.replace`` that satisfies ty."""
    return cast(T, _dc.replace(cast(Any, obj), **kwargs))


def dc_asdict(obj: Any) -> dict[str, Any]:
    """Typed wrapper around ``dataclasses.asdict`` that satisfies ty."""
    result: dict[str, Any] = _dc.asdict(obj)
    return result


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
    scheduled_close_ts: datetime | None = None  # from endDateIso (known in advance)
    tags: list[str] = field(default_factory=list)
    outcome_count: int = 2
    outcome_tokens: list[str] = field(default_factory=list)

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
    prev_snapshots: list[dict[str, Any]] = field(default_factory=list)


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
    family: Literal[
        "carry_only",
        "news_driven",
        "edge_based",
        "sell_edge",
        "arbitrage",
        "mean_reversion",
        "contrarian",
        "momentum",
        "volume_breakout",
        "resolution_convergence",
        "market_making",
    ]
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
    # Mean reversion: volume spike ratio to detect overreaction
    volume_spike_ratio: float = 3.0
    # Mean reversion: minimum move in bps to trigger
    reversion_move_bps: float = 200.0
    # Contrarian: extreme price threshold (buy NO when mid > this)
    extreme_high: float = 0.93
    # Contrarian: extreme price threshold (buy YES when mid < this)
    extreme_low: float = 0.07
    # Resolution convergence: max hours to resolution to activate
    resolution_hours_max: float = 72.0
    # Momentum: minimum consecutive direction snapshots (approximated via mid vs last_trade)
    momentum_min_edge_bps: float = 50.0
    # Carry exit: threshold below which mid triggers a sell
    carry_exit_threshold: float = 0.05
    # Generic exit: take profit after realizing this share of the max possible gain
    profit_target_pct: float = 0.0
    # Generic exit: fraction of the open position to sell when an exit triggers
    exit_fraction: float = 1.0
    # Generic exit: close positions older than this many hours
    time_exit_hours: float = 0.0
    # Capital management: max share of current cash to deploy into one position
    max_portfolio_pct: float = 0.5
    # Capital management: max share of starting cash allowed to remain invested
    max_total_invested_pct: float = 0.8
    # Volume sizing: cap position quantity to a fraction of 24h volume
    volume_sizing: bool = False
    # Volume sizing: max fraction of 24h volume for a single position
    volume_sizing_fraction: float = 0.001
    # Pyramiding: allow adding to profitable positions when edge persists
    allow_pyramiding: bool = False
    # Pyramiding: minimum edge improvement ratio to add (1.5 = edge must be 50% larger)
    pyramid_edge_improvement: float = 1.5
    # Category routing: None means all categories are eligible
    allowed_categories: list[str] | None = None
    # Category routing: None means no categories are blocked
    blocked_categories: list[str] | None = None
    # Market making: total spread quoted around mid, in basis points
    mm_spread_bps: float = 200.0
    # Market making: maximum directional inventory in contracts
    mm_max_inventory: float = 100.0

    def __post_init__(self) -> None:
        if not (0.0 < self.kelly_fraction <= 1.0):
            raise ValueError(f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}")
        if self.edge_threshold_bps < 0:
            raise ValueError(f"edge_threshold_bps must be >= 0, got {self.edge_threshold_bps}")
        if self.max_position_notional <= 0:
            raise ValueError(f"max_position_notional must be > 0, got {self.max_position_notional}")
        if self.profit_target_pct < 0:
            raise ValueError(f"profit_target_pct must be >= 0, got {self.profit_target_pct}")
        if not (0.0 < self.exit_fraction <= 1.0):
            raise ValueError(f"exit_fraction must be in (0, 1], got {self.exit_fraction}")
        if self.time_exit_hours < 0:
            raise ValueError(f"time_exit_hours must be >= 0, got {self.time_exit_hours}")
        if not (0.0 < self.max_portfolio_pct <= 1.0):
            raise ValueError(f"max_portfolio_pct must be in (0, 1], got {self.max_portfolio_pct}")
        if not (0.0 < self.max_total_invested_pct <= 1.0):
            raise ValueError(f"max_total_invested_pct must be in (0, 1], got {self.max_total_invested_pct}")
        if self.mm_spread_bps <= 0:
            raise ValueError(f"mm_spread_bps must be > 0, got {self.mm_spread_bps}")
        if self.mm_max_inventory <= 0:
            raise ValueError(f"mm_max_inventory must be > 0, got {self.mm_max_inventory}")
        if self.carry_price_min >= self.carry_price_max:
            raise ValueError(
                f"carry_price_min ({self.carry_price_min}) must be < carry_price_max ({self.carry_price_max})"
            )
        if self.extreme_low >= self.extreme_high:
            raise ValueError(f"extreme_low ({self.extreme_low}) must be < extreme_high ({self.extreme_high})")


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
    is_no_bet: bool = False
    order_type: Literal["default", "fok", "post_only"] = "default"

    def __post_init__(self) -> None:
        if not (0.001 <= self.limit_price <= 0.999):
            raise ValueError(f"limit_price must be in [0.001, 0.999], got {self.limit_price}")
        if self.requested_quantity <= 0:
            raise ValueError(f"requested_quantity must be > 0, got {self.requested_quantity}")


@dataclass
class RestingOrder:
    order_id: str
    strategy_name: str
    market_id: str
    placed_ts: datetime
    side: Literal["buy", "sell"]
    limit_price: float
    remaining_quantity: float
    is_no_bet: bool = False
    gtd_expiry: datetime | None = None

    def __post_init__(self) -> None:
        if not (0.001 <= self.limit_price <= 0.999):
            raise ValueError(f"limit_price must be in [0.001, 0.999], got {self.limit_price}")
        if self.remaining_quantity <= 0:
            raise ValueError(f"remaining_quantity must be > 0, got {self.remaining_quantity}")


@dataclass(frozen=True)
class CancelOrderAction:
    order_id: str

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id must be non-empty")


@dataclass(frozen=True)
class AmendOrderAction:
    order_id: str
    new_price: float
    new_quantity: float

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id must be non-empty")
        if not (0.001 <= self.new_price <= 0.999):
            raise ValueError(f"new_price must be in [0.001, 0.999], got {self.new_price}")
        if self.new_quantity <= 0:
            raise ValueError(f"new_quantity must be > 0, got {self.new_quantity}")


type StrategyAction = OrderIntent | CancelOrderAction | AmendOrderAction


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
    total_opened_notional: float = 0.0
    opened_ts: datetime | None = None
    closed_ts: datetime | None = None
    entry_probability: float = 0.0
    thesis: str = ""
    realized_pnl: float = 0.0
    realized_pnl_pre_resolution: float = 0.0
    fees_paid: float = 0.0
    rebates_earned: float = 0.0
    is_no_bet: bool = False

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
    eval_stride: int = 1
