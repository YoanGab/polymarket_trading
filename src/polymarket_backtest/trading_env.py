from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import db
from .cross_market import CrossMarketTracker
from .features import extract_snapshot_features
from .market_categories import category_fee_settings, normalize_market_tags
from .market_simulator import MarketSimulator
from .ml_transport import MLModelTransport
from .replay_engine import ReplayEngine, StrategyPortfolio
from .splits import HOLDOUT_CUTOFF, TRAIN_CUTOFF, VAL_CUTOFF
from .strategies import no_ask_price, no_bid_price, normalized_contract_price
from .types import (
    FillResult,
    MarketState,
    OrderIntent,
    PositionState,
    ReplayConfig,
    ensure_utc,
    isoformat,
)

DEFAULT_FEATURE_LOOKBACK = 24
DEFAULT_TOP_RELATED = 5
MIN_CONTRACT_PRICE = 0.001


@dataclass(frozen=True)
class _MarketMeta:
    """Static metadata for a market, loaded once and reused across all steps."""

    market_id: str
    title: str
    domain: str
    market_type: str
    tags_json: str | None
    resolution_ts: datetime | None
    fees_enabled: bool
    fee_rate: float
    fee_exponent: float
    maker_rebate_rate: float
    rules_text: str
    additional_context: str
    outcome_count: int
    outcome_tokens: list[str]
    # Pre-computed from tags
    normalized_tags: list[str]
    fee_cfg_fees_enabled: bool
    fee_cfg_fee_rate: float
    fee_cfg_fee_exponent: float
    fee_cfg_maker_rebate_rate: float


@dataclass(frozen=True)
class PositionInfo:
    market_id: str
    direction: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    age_hours: float
    edge_at_entry: float


@dataclass(frozen=True)
class RelatedMarketInfo:
    market_id: str
    title: str
    correlation: float
    best_bid: float
    best_ask: float
    mid: float
    category: str
    event_id: str | None = None


@dataclass
class TradingState:
    market_id: str
    timestamp: datetime
    hours_to_resolution: float
    category: str
    best_bid: float
    best_ask: float
    mid: float
    spread: float
    volume_24h: float
    momentum_3h: float
    momentum_6h: float
    momentum_12h: float
    momentum_24h: float
    volatility_24h: float
    price_range_24h: float
    ml_probability_yes: float | None
    ml_confidence: float | None
    ml_edge_bps: float | None
    cash: float
    cash_pct: float
    positions: list[PositionInfo]
    total_invested: float
    total_unrealized_pnl: float
    n_open_positions: int
    yes_position: PositionInfo | None
    no_position: PositionInfo | None
    related_markets: list[RelatedMarketInfo]
    starting_cash: float = 1_000.0

    def to_array(self) -> np.ndarray:
        def _value(value: float | None, scale: float = 1.0) -> float:
            if value is None or not math.isfinite(value):
                return 0.0
            return float(value) / scale

        top_related = self.related_markets[:3]
        related_corr = [_value(item.correlation) for item in top_related]
        related_mid = [_value(item.mid) for item in top_related]
        while len(related_corr) < 3:
            related_corr.append(0.0)
            related_mid.append(0.0)

        yes_qty = self.yes_position.quantity if self.yes_position is not None else 0.0
        yes_pnl_pct = self.yes_position.unrealized_pnl_pct if self.yes_position is not None else 0.0
        no_qty = self.no_position.quantity if self.no_position is not None else 0.0
        no_pnl_pct = self.no_position.unrealized_pnl_pct if self.no_position is not None else 0.0

        features = [
            _value(min(self.hours_to_resolution, 720.0), 720.0),
            _value(self.best_bid),
            _value(self.best_ask),
            _value(self.mid),
            _value(self.spread, 0.25),
            _value(min(self.volume_24h, 100_000.0), 100_000.0),
            _value(self.momentum_3h),
            _value(self.momentum_6h),
            _value(self.momentum_12h),
            _value(self.momentum_24h),
            _value(self.volatility_24h, 0.25),
            _value(self.price_range_24h),
            _value(self.ml_probability_yes),
            _value(self.ml_confidence),
            _value(self.ml_edge_bps, 1_000.0),
            _value(self.cash, max(self.starting_cash, 1.0)),
            _value(self.cash_pct),
            _value(self.total_invested, max(self.starting_cash, 1.0)),
            _value(self.total_unrealized_pnl, max(self.starting_cash, 1.0)),
            _value(self.n_open_positions, 10.0),
            _value(yes_qty, 100.0),
            _value(yes_pnl_pct),
            _value(no_qty, 100.0),
            _value(no_pnl_pct),
            _value(len(self.related_markets), 10.0),
            *related_corr,
            *related_mid,
        ]
        return np.asarray(features, dtype=np.float32)


@dataclass(frozen=True)
class Action:
    action_type: Literal[
        "hold",
        "buy_yes",
        "buy_no",
        "sell_yes",
        "sell_no",
        "buy_yes_limit",
        "buy_no_limit",
        "sell_yes_limit",
        "sell_no_limit",
        "mint_pair",
        "redeem_pair",
        "cancel_orders",
    ]
    quantity: float = 0.0
    price: float | None = None
    fraction: float = 1.0

    @staticmethod
    def hold() -> Action:
        return Action(action_type="hold")

    @staticmethod
    def buy_yes(quantity: float = 0.0, price: float | None = None) -> Action:
        return Action(action_type="buy_yes", quantity=quantity, price=price)

    @staticmethod
    def buy_no(quantity: float = 0.0, price: float | None = None) -> Action:
        return Action(action_type="buy_no", quantity=quantity, price=price)

    @staticmethod
    def sell_yes(fraction: float = 1.0, quantity: float = 0.0, price: float | None = None) -> Action:
        return Action(action_type="sell_yes", quantity=quantity, price=price, fraction=fraction)

    @staticmethod
    def sell_no(fraction: float = 1.0, quantity: float = 0.0, price: float | None = None) -> Action:
        return Action(action_type="sell_no", quantity=quantity, price=price, fraction=fraction)

    @staticmethod
    def buy_yes_limit(quantity: float = 0.0, price: float | None = None) -> Action:
        return Action(action_type="buy_yes_limit", quantity=quantity, price=price)

    @staticmethod
    def buy_no_limit(quantity: float = 0.0, price: float | None = None) -> Action:
        return Action(action_type="buy_no_limit", quantity=quantity, price=price)

    @staticmethod
    def sell_yes_limit(fraction: float = 1.0, quantity: float = 0.0, price: float | None = None) -> Action:
        return Action(action_type="sell_yes_limit", quantity=quantity, price=price, fraction=fraction)

    @staticmethod
    def sell_no_limit(fraction: float = 1.0, quantity: float = 0.0, price: float | None = None) -> Action:
        return Action(action_type="sell_no_limit", quantity=quantity, price=price, fraction=fraction)

    @staticmethod
    def mint_pair(quantity: float = 0.0) -> Action:
        return Action(action_type="mint_pair", quantity=quantity)

    @staticmethod
    def redeem_pair(quantity: float = 0.0) -> Action:
        return Action(action_type="redeem_pair", quantity=quantity)

    @staticmethod
    def cancel_orders() -> Action:
        return Action(action_type="cancel_orders")


@dataclass
class StepResult:
    reward: float
    new_state: TradingState
    done: bool
    info: dict[str, Any]
    filled_quantity: float
    fill_price: float
    fee_paid: float
    slippage_bps: float


@dataclass
class MultiMarketState:
    timestamp: datetime
    cash: float
    portfolio_value: float
    positions: list[PositionInfo]
    markets: dict[str, TradingState]


@dataclass
class _PendingOrder:
    order_id: str
    market_id: str
    intent: OrderIntent
    remaining_quantity: float


@dataclass
class _MarketEpisode:
    market_id: str
    snapshot_rows: list[Any]
    resolution_ts: datetime | None
    index: int = 0
    done: bool = False

    @property
    def current_row(self) -> Any:
        return self.snapshot_rows[self.index]

    @property
    def current_ts(self) -> datetime:
        return datetime.fromisoformat(str(self.current_row["ts"]))

    def next_row(self) -> Any | None:
        next_index = self.index + 1
        if next_index >= len(self.snapshot_rows):
            return None
        return self.snapshot_rows[next_index]


class _DictRow:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def keys(self) -> list[str]:
        return list(self._data.keys())


class _ExecutionShim:
    _normalize_quotes = ReplayEngine._normalize_quotes
    _ensure_orderbook = ReplayEngine._ensure_orderbook
    _build_degraded_next_market = ReplayEngine._build_degraded_next_market
    _normalized_resolved_outcome = ReplayEngine._normalized_resolved_outcome
    _settlement_fee_rate = ReplayEngine._settlement_fee_rate
    _apply_fill = ReplayEngine._apply_fill
    _persist_and_evict_position = ReplayEngine._persist_and_evict_position
    _redeem_matched_pairs = ReplayEngine._redeem_matched_pairs
    _apply_redemption_to_position = ReplayEngine._apply_redemption_to_position
    _payout_ratio_for_position = ReplayEngine._payout_ratio_for_position
    _position_mark_price = ReplayEngine._position_mark_price

    def __init__(self, *, starting_cash: float, simulator: MarketSimulator) -> None:
        self.simulator = simulator
        self.config = ReplayConfig(
            experiment_name="trading-environment",
            starting_cash=starting_cash,
            lookback_minutes=0,
        )

    def _persist_position(self, position: PositionState) -> None:
        return None


class _TradingCore:
    def __init__(
        self,
        *,
        conn: Any,
        starting_cash: float,
        auto_order_cash_fraction: float,
        top_related_markets: int,
        feature_lookback: int,
        enable_ml_predictions: bool,
        random_seed: int | None,
    ) -> None:
        self.conn = conn
        self.starting_cash = starting_cash
        self.auto_order_cash_fraction = auto_order_cash_fraction
        self.top_related_markets = top_related_markets
        self.feature_lookback = feature_lookback
        self.simulator = MarketSimulator()
        self.executor = _ExecutionShim(starting_cash=starting_cash, simulator=self.simulator)
        self.cross_market = CrossMarketTracker(conn)
        self.portfolio = StrategyPortfolio(cash=starting_cash)
        self.pending_orders: dict[str, _PendingOrder] = {}
        self._rng = random.Random(random_seed)
        self._resolution_cache: dict[str, Any] = {}
        self._ml_transport = self._load_ml_transport() if enable_ml_predictions else None
        # Caches for fast MarketState construction (no DB per step)
        self._market_meta_cache: dict[str, _MarketMeta] = {}
        self._market_state_cache: dict[tuple[str, int], MarketState] = {}

    def reset_portfolio(self) -> None:
        self.portfolio = StrategyPortfolio(cash=self.starting_cash)
        self.pending_orders = {}

    def clear_market_state_cache(self) -> None:
        """Clear the per-step market state cache (call between episodes/resets)."""
        self._market_state_cache.clear()

    def build_state(self, episode: _MarketEpisode) -> TradingState:
        market = self._market_state_for_episode(episode)
        self._update_mark_for_market(market)
        features = self._feature_dict(episode)
        ml_output = self._predict_market(market, episode)
        positions = self._position_infos(active_markets={episode.market_id: market})
        yes_position = next(
            (item for item in positions if item.market_id == episode.market_id and item.direction == "yes"),
            None,
        )
        no_position = next(
            (item for item in positions if item.market_id == episode.market_id and item.direction == "no"),
            None,
        )
        total_invested = sum(item.quantity * item.entry_price for item in positions)
        total_unrealized_pnl = sum(item.unrealized_pnl for item in positions)

        hours_to_resolution = 720.0
        if market.seconds_to_resolution is not None:
            hours_to_resolution = market.seconds_to_resolution / 3600.0

        return TradingState(
            market_id=episode.market_id,
            timestamp=market.ts,
            hours_to_resolution=hours_to_resolution,
            category=self._market_category(market),
            best_bid=market.best_bid,
            best_ask=market.best_ask,
            mid=market.mid,
            spread=market.best_ask - market.best_bid,
            volume_24h=market.volume_24h,
            momentum_3h=float(features.get("momentum_3h", 0.0)),
            momentum_6h=float(features.get("momentum_6h", 0.0)),
            momentum_12h=float(features.get("momentum_12h", 0.0)),
            momentum_24h=float(features.get("momentum_24h", 0.0)),
            volatility_24h=float(features.get("volatility_24h", 0.0)),
            price_range_24h=float(features.get("price_range_24h", 0.0)),
            ml_probability_yes=ml_output["probability_yes"] if ml_output is not None else None,
            ml_confidence=ml_output["confidence"] if ml_output is not None else None,
            ml_edge_bps=ml_output["expected_edge_bps"] if ml_output is not None else None,
            cash=self.portfolio.cash,
            cash_pct=self.portfolio.cash / max(self.starting_cash, 1.0),
            positions=positions,
            total_invested=total_invested,
            total_unrealized_pnl=total_unrealized_pnl,
            n_open_positions=len(positions),
            yes_position=yes_position,
            no_position=no_position,
            related_markets=self._related_markets(market),
            starting_cash=self.starting_cash,
        )

    def available_actions(self, episode: _MarketEpisode) -> list[Action]:
        market = self._market_state_for_episode(episode)
        actions = [Action.hold()]
        min_qty = self.simulator.minimum_fill_quantity
        if self._default_buy_quantity(price=market.best_ask) >= min_qty:
            actions.append(Action.buy_yes())
            actions.append(Action.buy_yes_limit(price=market.best_bid))
        if self._default_buy_quantity(price=no_ask_price(market)) >= min_qty:
            actions.append(Action.buy_no())
            actions.append(Action.buy_no_limit(price=no_bid_price(market)))
        yes_position = self.portfolio.positions.get(episode.market_id)
        no_position = self.portfolio.positions.get(f"{episode.market_id}:NO")
        if yes_position is not None and yes_position.quantity >= min_qty:
            actions.append(Action.sell_yes())
            actions.append(Action.sell_yes_limit(price=market.best_ask))
        if no_position is not None and no_position.quantity >= min_qty:
            actions.append(Action.sell_no())
            actions.append(Action.sell_no_limit(price=no_ask_price(market)))
        if self._default_pair_quantity() >= min_qty:
            actions.append(Action.mint_pair())
        if self._redeemable_quantity(episode.market_id) >= min_qty:
            actions.append(Action.redeem_pair())
        if self.pending_orders:
            actions.append(Action.cancel_orders())
        return actions

    def _execute_single_action(
        self,
        episode: _MarketEpisode,
        action: Action,
        current_market: MarketState,
        next_market: MarketState | None,
    ) -> tuple[list[FillResult], dict[str, Any]]:
        """Execute one action at the current timestamp without advancing time.

        Returns:
            A tuple of (fill_records, extra_info) where extra_info contains
            action-specific metadata (cancelled_orders, redeemed_quantity, etc.).
        """
        extra_info: dict[str, Any] = {}
        fill_records: list[FillResult] = []

        if action.action_type == "cancel_orders":
            extra_info["cancelled_orders"] = len(self.pending_orders)
            self.pending_orders.clear()
        elif action.action_type == "mint_pair":
            fill_records.extend(self._mint_pair(episode, current_market, action))
        elif action.action_type == "redeem_pair":
            extra_info["redeemed_quantity"] = self._redeem_pair(episode.market_id, current_market, action.quantity)
        elif action.action_type != "hold":
            intent = self._action_to_intent(action, current_market)
            if intent is not None:
                if intent.liquidity_intent == "aggressive":
                    fill_records.extend(self._execute_intent(current_market, next_market, intent))
                else:
                    order_id = self._next_order_id()
                    self.pending_orders[order_id] = _PendingOrder(
                        order_id=order_id,
                        market_id=episode.market_id,
                        intent=intent,
                        remaining_quantity=intent.requested_quantity,
                    )

        return fill_records, extra_info

    def _advance_episode(self, episode: _MarketEpisode, current_market: MarketState) -> None:
        """Advance the episode to the next snapshot and handle settlement."""
        next_market = self._next_market_state_for_episode(episode)
        if next_market is not None:
            episode.index += 1
        else:
            episode.done = True

        settlement_ts = self._settlement_timestamp(episode, current_market)
        self._settle_market(episode.market_id, settlement_ts)
        if episode.resolution_ts is not None and ensure_utc(settlement_ts) >= ensure_utc(episode.resolution_ts):
            episode.done = True

    def step_episode_fast(self, episode: _MarketEpisode, action: Action) -> None:
        """Execute action and advance without building state or computing value.

        This is the fast path for the gym env which builds observations
        externally. It skips the TradingState construction and portfolio_value
        computation that step_episode does.
        """
        if episode.done:
            return

        current_market = self._market_state_for_episode(episode)
        next_market = self._next_market_state_for_episode(episode)

        fill_records, _extra_info = self._execute_single_action(
            episode,
            action,
            current_market,
            next_market,
        )
        fill_records.extend(
            self._process_pending_orders(current_market, next_market, episode.market_id),
        )

        self._advance_episode(episode, current_market)

    def step_episode(self, episode: _MarketEpisode, action: Action) -> StepResult:
        if episode.done:
            raise RuntimeError(f"Episode for market {episode.market_id} is already done")

        current_market = self._market_state_for_episode(episode)
        next_market = self._next_market_state_for_episode(episode)
        before_value = self.portfolio_value(active_markets={episode.market_id: current_market})

        fill_records, extra_info = self._execute_single_action(episode, action, current_market, next_market)
        fill_records.extend(self._process_pending_orders(current_market, next_market, episode.market_id))

        self._advance_episode(episode, current_market)

        state = self.build_state(episode)
        after_value = self.portfolio_value(active_markets={episode.market_id: self._market_state_for_episode(episode)})

        info: dict[str, Any] = {
            "action": action,
            "cancelled_orders": extra_info.get("cancelled_orders", 0),
            "pending_orders_before": len(self.pending_orders),
            "fills": [self._fill_to_dict(fill) for fill in fill_records],
            "pending_orders_after": len(self.pending_orders),
        }
        if "redeemed_quantity" in extra_info:
            info["redeemed_quantity"] = extra_info["redeemed_quantity"]

        total_qty = sum(fill.quantity for fill in fill_records)
        total_fee = sum(fill.fee_usdc - fill.rebate_usdc for fill in fill_records)
        fill_price = sum(fill.price * fill.quantity for fill in fill_records) / total_qty if total_qty > 0 else 0.0
        slippage_bps = (
            sum(fill.impact_bps * fill.quantity for fill in fill_records) / total_qty if total_qty > 0 else 0.0
        )
        return StepResult(
            reward=after_value - before_value,
            new_state=state,
            done=episode.done,
            info=info,
            filled_quantity=total_qty,
            fill_price=fill_price,
            fee_paid=total_fee,
            slippage_bps=slippage_bps,
        )

    def step_multi_episode(self, episode: _MarketEpisode, actions: list[Action]) -> StepResult:
        """Execute multiple actions at the same timestamp, then advance.

        This allows placing multiple orders simultaneously (e.g., buy YES +
        set limit sell) before the market moves to the next snapshot.
        """
        if episode.done:
            raise RuntimeError(f"Episode for market {episode.market_id} is already done")

        current_market = self._market_state_for_episode(episode)
        next_market = self._next_market_state_for_episode(episode)
        before_value = self.portfolio_value(active_markets={episode.market_id: current_market})

        all_fill_records: list[FillResult] = []
        merged_info: dict[str, Any] = {
            "actions": list(actions),
            "cancelled_orders": 0,
            "pending_orders_before": len(self.pending_orders),
        }

        for action in actions:
            fill_records, extra_info = self._execute_single_action(episode, action, current_market, next_market)
            all_fill_records.extend(fill_records)
            merged_info["cancelled_orders"] += extra_info.get("cancelled_orders", 0)
            if "redeemed_quantity" in extra_info:
                merged_info.setdefault("redeemed_quantity", 0.0)
                merged_info["redeemed_quantity"] += extra_info["redeemed_quantity"]

        all_fill_records.extend(self._process_pending_orders(current_market, next_market, episode.market_id))

        self._advance_episode(episode, current_market)

        state = self.build_state(episode)
        after_value = self.portfolio_value(active_markets={episode.market_id: self._market_state_for_episode(episode)})

        merged_info["fills"] = [self._fill_to_dict(fill) for fill in all_fill_records]
        merged_info["pending_orders_after"] = len(self.pending_orders)

        total_qty = sum(fill.quantity for fill in all_fill_records)
        total_fee = sum(fill.fee_usdc - fill.rebate_usdc for fill in all_fill_records)
        fill_price = sum(fill.price * fill.quantity for fill in all_fill_records) / total_qty if total_qty > 0 else 0.0
        slippage_bps = (
            sum(fill.impact_bps * fill.quantity for fill in all_fill_records) / total_qty if total_qty > 0 else 0.0
        )
        return StepResult(
            reward=after_value - before_value,
            new_state=state,
            done=episode.done,
            info=merged_info,
            filled_quantity=total_qty,
            fill_price=fill_price,
            fee_paid=total_fee,
            slippage_bps=slippage_bps,
        )

    def portfolio_value(self, *, active_markets: dict[str, MarketState]) -> float:
        for market in active_markets.values():
            self._update_mark_for_market(market)
        inventory_value = sum(
            position.quantity
            * self.portfolio.last_known_mids.get(
                _position_key(position.market_id, position.is_no_bet),
                position.avg_entry_price,
            )
            for position in self.portfolio.positions.values()
            if position.quantity > 0
        )
        return self.portfolio.cash + inventory_value

    def _load_ml_transport(self) -> MLModelTransport | None:
        try:
            return MLModelTransport()
        except FileNotFoundError:
            return None

    def _load_market_meta(self, market_id: str) -> _MarketMeta:
        """Load static market metadata once and cache it."""
        if market_id in self._market_meta_cache:
            return self._market_meta_cache[market_id]

        row = self.conn.execute(
            """
            SELECT market_id, title, domain, market_type, tags_json,
                   resolution_ts, fees_enabled, fee_rate, fee_exponent,
                   maker_rebate_rate
            FROM markets
            WHERE market_id = ?
            """,
            (market_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Market {market_id!r} not found in markets table")

        tags_json = row["tags_json"]
        tags = db._parse_tags_json(tags_json)
        normalized_tags = normalize_market_tags(tags)
        if normalized_tags:
            fee_cfg = category_fee_settings(normalized_tags)
            cfg_fees_enabled = fee_cfg.fees_enabled
            cfg_fee_rate = fee_cfg.fee_rate
            cfg_fee_exponent = fee_cfg.fee_exponent
            cfg_maker_rebate_rate = fee_cfg.maker_rebate_rate
        else:
            cfg_fees_enabled = bool(row["fees_enabled"])
            cfg_fee_rate = float(row["fee_rate"])
            cfg_fee_exponent = float(row["fee_exponent"])
            cfg_maker_rebate_rate = float(row["maker_rebate_rate"])

        resolution_ts = datetime.fromisoformat(str(row["resolution_ts"])) if row["resolution_ts"] is not None else None

        # Load rules (latest revision, or empty)
        rule_row = self.conn.execute(
            """
            SELECT rules_text, additional_context
            FROM market_rule_revisions
            WHERE market_id = ?
            ORDER BY effective_ts DESC
            LIMIT 1
            """,
            (market_id,),
        ).fetchone()

        event_outcome_tokens = db.get_event_outcome_tokens(self.conn, market_id)
        outcome_tokens = event_outcome_tokens if len(event_outcome_tokens) > 1 else []

        meta = _MarketMeta(
            market_id=market_id,
            title=str(row["title"]),
            domain=str(row["domain"]),
            market_type=str(row["market_type"]),
            tags_json=tags_json,
            resolution_ts=resolution_ts,
            fees_enabled=bool(row["fees_enabled"]),
            fee_rate=float(row["fee_rate"]),
            fee_exponent=float(row["fee_exponent"]),
            maker_rebate_rate=float(row["maker_rebate_rate"]),
            rules_text=str(rule_row["rules_text"]) if rule_row else "",
            additional_context=str(rule_row["additional_context"]) if rule_row else "",
            outcome_count=len(outcome_tokens) if outcome_tokens else 2,
            outcome_tokens=outcome_tokens,
            normalized_tags=normalized_tags if normalized_tags else tags,
            fee_cfg_fees_enabled=cfg_fees_enabled,
            fee_cfg_fee_rate=cfg_fee_rate,
            fee_cfg_fee_exponent=cfg_fee_exponent,
            fee_cfg_maker_rebate_rate=cfg_maker_rebate_rate,
        )
        self._market_meta_cache[market_id] = meta
        return meta

    def _market_state_from_row(self, market_id: str, snapshot_row: Any) -> MarketState:
        """Build a MarketState from an in-memory snapshot row + cached metadata.

        This avoids all DB queries and is the fast path used by the gym env.
        """
        meta = self._load_market_meta(market_id)
        market = MarketState(
            market_id=meta.market_id,
            title=meta.title,
            domain=meta.domain,
            market_type=meta.market_type,
            ts=datetime.fromisoformat(str(snapshot_row["ts"])),
            status=str(snapshot_row["status"]),
            best_bid=float(snapshot_row["best_bid"]),
            best_ask=float(snapshot_row["best_ask"]),
            mid=float(snapshot_row["mid"]),
            last_trade=float(snapshot_row["last_trade"]),
            volume_1m=float(snapshot_row["volume_1m"]),
            volume_24h=float(snapshot_row["volume_24h"]),
            open_interest=float(snapshot_row["open_interest"]),
            tick_size=float(snapshot_row["tick_size"]),
            rules_text=meta.rules_text,
            additional_context=meta.additional_context,
            resolution_ts=meta.resolution_ts,
            fees_enabled=meta.fee_cfg_fees_enabled,
            fee_rate=meta.fee_cfg_fee_rate,
            fee_exponent=meta.fee_cfg_fee_exponent,
            maker_rebate_rate=meta.fee_cfg_maker_rebate_rate,
            orderbook=[],  # Will be synthesized by _ensure_orderbook
            tags=meta.normalized_tags if meta.normalized_tags else meta.normalized_tags,
            outcome_count=meta.outcome_count,
            outcome_tokens=meta.outcome_tokens,
        )
        market = self.executor._normalize_quotes(market)
        return self.executor._ensure_orderbook(market, reason="trading_env")

    _MARKET_STATE_CACHE_MAX = 10_000

    def _market_state_for_episode(self, episode: _MarketEpisode) -> MarketState:
        cache_key = (episode.market_id, episode.index)
        cached = self._market_state_cache.get(cache_key)
        if cached is not None:
            return cached

        market = self._market_state_from_row(episode.market_id, episode.current_row)
        if len(self._market_state_cache) >= self._MARKET_STATE_CACHE_MAX:
            self._market_state_cache.clear()
        self._market_state_cache[cache_key] = market
        return market

    def _next_market_state_for_episode(self, episode: _MarketEpisode) -> MarketState | None:
        next_row = episode.next_row()
        if next_row is None:
            return None

        next_index = episode.index + 1
        cache_key = (episode.market_id, next_index)
        cached = self._market_state_cache.get(cache_key)
        if cached is not None:
            return cached

        market = self._market_state_from_row(episode.market_id, next_row)
        if len(self._market_state_cache) >= self._MARKET_STATE_CACHE_MAX:
            self._market_state_cache.clear()
        self._market_state_cache[cache_key] = market
        return market

    def _feature_row(self, snapshot_row: Any, resolution_ts: datetime | None) -> _DictRow:
        keys = snapshot_row.keys()
        payload = {key: snapshot_row[key] for key in keys}
        payload["resolution_ts"] = isoformat(resolution_ts) if resolution_ts is not None else None
        return _DictRow(payload)

    def _feature_dict(self, episode: _MarketEpisode) -> dict[str, float]:
        current_row = self._feature_row(episode.current_row, episode.resolution_ts)
        start = max(0, episode.index - self.feature_lookback)
        prev_rows = [
            self._feature_row(row, episode.resolution_ts) for row in episode.snapshot_rows[start : episode.index]
        ]
        return extract_snapshot_features(current_row, prev_rows)

    def _predict_market(self, market: MarketState, episode: _MarketEpisode) -> dict[str, float] | None:
        if self._ml_transport is None:
            return None
        start = max(0, episode.index - self.feature_lookback)
        prev_snapshots = []
        for row in episode.snapshot_rows[start : episode.index]:
            keys = row.keys()
            payload = {key: row[key] for key in keys}
            payload["resolution_ts"] = isoformat(episode.resolution_ts) if episode.resolution_ts is not None else None
            prev_snapshots.append(payload)
        raw = self._ml_transport.complete(
            model_release="local-ml-model",
            system_prompt="",
            context_bundle={
                "as_of": isoformat(market.ts),
                "market": {
                    "market_id": market.market_id,
                    "best_bid": market.best_bid,
                    "best_ask": market.best_ask,
                    "mid": market.mid,
                    "last_trade": market.last_trade,
                    "volume_1m": market.volume_1m,
                    "volume_24h": market.volume_24h,
                    "open_interest": market.open_interest,
                    "resolution_ts": (isoformat(market.resolution_ts) if market.resolution_ts is not None else None),
                    "tags": list(market.tags),
                },
                "prev_snapshots": prev_snapshots,
            },
        )
        return {
            "probability_yes": float(raw["probability_yes"]),
            "confidence": float(raw["confidence"]),
            "expected_edge_bps": float(raw["expected_edge_bps"]),
        }

    def _position_infos(self, *, active_markets: dict[str, MarketState]) -> list[PositionInfo]:
        position_infos: list[PositionInfo] = []
        for position in self.portfolio.positions.values():
            if position.quantity <= 0:
                continue
            key = _position_key(position.market_id, position.is_no_bet)
            market = active_markets.get(position.market_id)
            if market is not None:
                current_price = self.executor._position_mark_price(market, position)
                self.portfolio.last_known_mids[key] = current_price
            else:
                current_price = self.portfolio.last_known_mids.get(key, position.avg_entry_price)
            unrealized = (current_price - position.avg_entry_price) * position.quantity
            unrealized_pct = unrealized / max(position.avg_entry_price * position.quantity, 1e-9)
            age_hours = 0.0
            if position.opened_ts is not None:
                age_hours = (
                    ensure_utc(next(iter(active_markets.values())).ts) - ensure_utc(position.opened_ts)
                ).total_seconds() / 3600.0
            position_infos.append(
                PositionInfo(
                    market_id=position.market_id,
                    direction="no" if position.is_no_bet else "yes",
                    quantity=position.quantity,
                    entry_price=position.avg_entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=unrealized_pct,
                    age_hours=age_hours,
                    edge_at_entry=(position.entry_probability - position.avg_entry_price) * 10_000.0,
                )
            )
        position_infos.sort(key=lambda item: (item.market_id, item.direction))
        return position_infos

    def _related_markets(self, market: MarketState) -> list[RelatedMarketInfo]:
        related_infos: list[RelatedMarketInfo] = []
        for related in self.cross_market.get_related_markets(market.market_id)[: self.top_related_markets]:
            related_market = db.get_market_state_as_of(self.conn, related.market_id, market.ts)
            if related_market is None:
                continue
            metadata = self.cross_market.get_market_metadata(related.market_id)
            related_infos.append(
                RelatedMarketInfo(
                    market_id=related.market_id,
                    title=related_market.title,
                    correlation=related.correlation,
                    best_bid=related_market.best_bid,
                    best_ask=related_market.best_ask,
                    mid=related_market.mid,
                    category=self._market_category(related_market),
                    event_id=metadata.event_id if metadata is not None else None,
                )
            )
        return related_infos

    def _market_category(self, market: MarketState) -> str:
        tags = normalize_market_tags(market.tags)
        if tags:
            return tags[0].lower()
        return str(market.domain or market.market_type).lower()

    def _default_buy_quantity(self, *, price: float) -> float:
        if price <= 0:
            return 0.0
        notional = self.portfolio.cash * self.auto_order_cash_fraction
        return max(0.0, notional / price)

    def _default_pair_quantity(self) -> float:
        notional = self.portfolio.cash * self.auto_order_cash_fraction
        return max(0.0, notional)

    def _current_position(self, market_id: str, *, is_no_bet: bool) -> PositionState | None:
        return self.portfolio.positions.get(_position_key(market_id, is_no_bet))

    def _sell_quantity(self, market_id: str, *, is_no_bet: bool, action: Action) -> float:
        position = self._current_position(market_id, is_no_bet=is_no_bet)
        if position is None or position.quantity <= 0:
            return 0.0
        if action.quantity > 0:
            return min(position.quantity, action.quantity)
        return position.quantity * max(0.0, min(1.0, action.fraction))

    def _action_to_intent(self, action: Action, market: MarketState) -> OrderIntent | None:
        is_limit = action.action_type.endswith("_limit")
        is_buy = action.action_type.startswith("buy_")
        is_no_bet = "_no" in action.action_type

        if is_buy:
            default_price = self._default_buy_price(market, is_no_bet=is_no_bet, passive=is_limit)
            quantity = action.quantity if action.quantity > 0 else self._default_buy_quantity(price=default_price)
        else:
            default_price = self._default_sell_price(market, is_no_bet=is_no_bet, passive=is_limit)
            quantity = self._sell_quantity(market.market_id, is_no_bet=is_no_bet, action=action)

        if quantity < self.simulator.minimum_fill_quantity:
            return None

        limit_price = normalized_contract_price(action.price if action.price is not None else default_price)
        return OrderIntent(
            strategy_name="trading_env",
            market_id=market.market_id,
            ts=market.ts,
            side="buy" if is_buy else "sell",
            liquidity_intent="passive" if is_limit else "aggressive",
            limit_price=limit_price,
            requested_quantity=quantity,
            kelly_fraction=0.0,
            edge_bps=0.0,
            holding_period_minutes=None,
            thesis=f"Action {action.action_type}",
            is_no_bet=is_no_bet,
            order_type="post_only" if is_limit else "default",
        )

    def _default_buy_price(self, market: MarketState, *, is_no_bet: bool, passive: bool) -> float:
        if is_no_bet:
            return no_bid_price(market) if passive else no_ask_price(market)
        return market.best_bid if passive else market.best_ask

    def _default_sell_price(self, market: MarketState, *, is_no_bet: bool, passive: bool) -> float:
        if is_no_bet:
            return no_ask_price(market) if passive else no_bid_price(market)
        return market.best_ask if passive else market.best_bid

    def _execute_intent(
        self,
        market: MarketState,
        next_market: MarketState | None,
        intent: OrderIntent,
    ) -> list[FillResult]:
        position = self._current_position(intent.market_id, is_no_bet=intent.is_no_bet)
        if intent.side == "sell" and (position is None or position.quantity < self.simulator.minimum_fill_quantity):
            return []
        if intent.side == "sell" and position is not None and intent.requested_quantity > position.quantity:
            intent = OrderIntent(
                strategy_name=intent.strategy_name,
                market_id=intent.market_id,
                ts=intent.ts,
                side=intent.side,
                liquidity_intent=intent.liquidity_intent,
                limit_price=intent.limit_price,
                requested_quantity=position.quantity,
                kelly_fraction=intent.kelly_fraction,
                edge_bps=intent.edge_bps,
                holding_period_minutes=intent.holding_period_minutes,
                thesis=intent.thesis,
                is_no_bet=intent.is_no_bet,
                order_type=intent.order_type,
            )

        effective_next_market = (
            next_market if next_market is not None else self.executor._build_degraded_next_market(market, intent)
        )
        fills = self.simulator.simulate(
            order_id=self._next_order_id(),
            market=market,
            next_market=effective_next_market,
            intent=intent,
        )
        applied: list[FillResult] = []
        entry_probability = market.mid if not intent.is_no_bet else 1.0 - market.mid
        for fill in fills:
            adjusted_fill = self._clamp_fill_to_cash(fill) if fill.side == "buy" else fill
            if adjusted_fill.quantity < self.simulator.minimum_fill_quantity:
                continue
            self.executor._apply_fill(
                self.portfolio,
                adjusted_fill,
                intent.thesis,
                entry_probability,
                is_no_bet=intent.is_no_bet,
            )
            applied.append(adjusted_fill)
        return applied

    def _clamp_fill_to_cash(self, fill: FillResult) -> FillResult:
        fill_cost = fill.price * fill.quantity + fill.fee_usdc - fill.rebate_usdc
        if fill_cost <= self.portfolio.cash + 1e-9:
            return fill
        max_affordable_qty = self.portfolio.cash / max(fill.price, 1e-12)
        if max_affordable_qty < self.simulator.minimum_fill_quantity:
            return FillResult(
                order_id=fill.order_id,
                market_id=fill.market_id,
                strategy_name=fill.strategy_name,
                fill_ts=fill.fill_ts,
                side=fill.side,
                liquidity_role=fill.liquidity_role,
                price=fill.price,
                quantity=0.0,
                fee_usdc=0.0,
                rebate_usdc=0.0,
                impact_bps=fill.impact_bps,
                fill_delay_seconds=fill.fill_delay_seconds,
            )
        ratio = max_affordable_qty / max(fill.quantity, 1e-12)
        return FillResult(
            order_id=fill.order_id,
            market_id=fill.market_id,
            strategy_name=fill.strategy_name,
            fill_ts=fill.fill_ts,
            side=fill.side,
            liquidity_role=fill.liquidity_role,
            price=fill.price,
            quantity=round(max_affordable_qty, 4),
            fee_usdc=round(fill.fee_usdc * ratio, 4),
            rebate_usdc=round(fill.rebate_usdc * ratio, 4),
            impact_bps=fill.impact_bps,
            fill_delay_seconds=fill.fill_delay_seconds,
        )

    def _process_pending_orders(
        self,
        market: MarketState,
        next_market: MarketState | None,
        market_id: str,
    ) -> list[FillResult]:
        fills: list[FillResult] = []
        for order_id, pending in list(self.pending_orders.items()):
            if pending.market_id != market_id:
                continue
            intent = OrderIntent(
                strategy_name=pending.intent.strategy_name,
                market_id=pending.intent.market_id,
                ts=market.ts,
                side=pending.intent.side,
                liquidity_intent=pending.intent.liquidity_intent,
                limit_price=pending.intent.limit_price,
                requested_quantity=pending.remaining_quantity,
                kelly_fraction=pending.intent.kelly_fraction,
                edge_bps=pending.intent.edge_bps,
                holding_period_minutes=pending.intent.holding_period_minutes,
                thesis=pending.intent.thesis,
                is_no_bet=pending.intent.is_no_bet,
                order_type=pending.intent.order_type,
            )
            applied = self._execute_intent(market, next_market, intent)
            filled_qty = sum(fill.quantity for fill in applied)
            pending.remaining_quantity = max(0.0, pending.remaining_quantity - filled_qty)
            fills.extend(applied)
            if pending.remaining_quantity < self.simulator.minimum_fill_quantity or market.status not in {
                "active",
                "open",
            }:
                del self.pending_orders[order_id]
        return fills

    def _mint_pair(self, episode: _MarketEpisode, market: MarketState, action: Action) -> list[FillResult]:
        quantity = action.quantity if action.quantity > 0 else self._default_pair_quantity()
        quantity = min(quantity, self.portfolio.cash)
        if quantity < self.simulator.minimum_fill_quantity:
            return []

        yes_fill = FillResult(
            order_id=self._next_order_id(),
            market_id=episode.market_id,
            strategy_name="trading_env",
            fill_ts=market.ts,
            side="buy",
            liquidity_role="maker",
            price=normalized_contract_price(market.mid),
            quantity=quantity,
            fee_usdc=0.0,
            rebate_usdc=0.0,
            impact_bps=0.0,
            fill_delay_seconds=0.0,
        )
        no_fill = FillResult(
            order_id=self._next_order_id(),
            market_id=episode.market_id,
            strategy_name="trading_env",
            fill_ts=market.ts,
            side="buy",
            liquidity_role="maker",
            price=normalized_contract_price(1.0 - market.mid),
            quantity=quantity,
            fee_usdc=0.0,
            rebate_usdc=0.0,
            impact_bps=0.0,
            fill_delay_seconds=0.0,
        )
        self.executor._apply_fill(self.portfolio, yes_fill, "Mint YES leg", market.mid, is_no_bet=False)
        self.executor._apply_fill(self.portfolio, no_fill, "Mint NO leg", 1.0 - market.mid, is_no_bet=True)
        return [yes_fill, no_fill]

    def _redeemable_quantity(self, market_id: str) -> float:
        yes_position = self._current_position(market_id, is_no_bet=False)
        no_position = self._current_position(market_id, is_no_bet=True)
        if yes_position is None or no_position is None:
            return 0.0
        return min(yes_position.quantity, no_position.quantity)

    def _redeem_pair(self, market_id: str, market: MarketState, quantity: float) -> float:
        redeemable = self._redeemable_quantity(market_id)
        if redeemable < self.simulator.minimum_fill_quantity:
            return 0.0
        if quantity <= 0 or quantity >= redeemable - 1e-12:
            return self.executor._redeem_matched_pairs(
                portfolio=self.portfolio,
                strategy_name="trading_env",
                market=market,
            )

        yes_key = market_id
        no_key = f"{market_id}:NO"
        yes_position = self.portfolio.positions.get(yes_key)
        no_position = self.portfolio.positions.get(no_key)
        if yes_position is None or no_position is None:
            return 0.0

        yes_cost_basis = yes_position.avg_entry_price * quantity
        no_cost_basis = no_position.avg_entry_price * quantity
        total_cost_basis = yes_cost_basis + no_cost_basis
        redemption_value = quantity
        redemption_pnl = redemption_value - total_cost_basis
        yes_realized = (
            redemption_pnl * (yes_cost_basis / total_cost_basis) if total_cost_basis > 0 else redemption_pnl / 2.0
        )
        no_realized = redemption_pnl - yes_realized

        self.portfolio.cash += redemption_value
        self.portfolio.realized_pnl += redemption_pnl

        self._apply_partial_redemption(yes_key, yes_position, quantity, yes_realized, market.ts)
        self._apply_partial_redemption(no_key, no_position, quantity, no_realized, market.ts)
        return quantity

    def _apply_partial_redemption(
        self,
        position_key: str,
        position: PositionState,
        quantity: float,
        realized_pnl: float,
        redeem_ts: datetime,
    ) -> None:
        position.realized_pnl += realized_pnl
        position.realized_pnl_pre_resolution += realized_pnl
        position.quantity = max(0.0, position.quantity - quantity)
        if position.quantity > 0:
            return
        position.quantity = 0.0
        position.closed_ts = redeem_ts
        self.executor._persist_and_evict_position(self.portfolio, position_key)

    def _settlement_timestamp(self, episode: _MarketEpisode, market: MarketState) -> datetime:
        if episode.done and episode.resolution_ts is not None:
            return max(ensure_utc(market.ts), ensure_utc(episode.resolution_ts))
        return self._market_state_for_episode(episode).ts

    def _settle_market(self, market_id: str, as_of: datetime) -> None:
        resolution = self._get_resolution(market_id)
        if resolution is None or resolution["resolution_ts"] is None:
            return
        resolution_ts = datetime.fromisoformat(str(resolution["resolution_ts"]))
        if ensure_utc(as_of) < ensure_utc(resolution_ts):
            return

        resolved_outcome = self.executor._normalized_resolved_outcome(
            resolution,
            market_id,
            log_ambiguity=False,
        )
        if resolved_outcome is None:
            return

        for position_key in (_position_key(market_id, False), _position_key(market_id, True)):
            position = self.portfolio.positions.get(position_key)
            if position is None or position.quantity <= 0:
                continue
            payout_ratio = self.executor._payout_ratio_for_position(position, resolved_outcome)
            gross_payout = payout_ratio * position.quantity
            profit = gross_payout - position.total_opened_notional
            settlement_fee = max(0.0, profit) * self.executor._settlement_fee_rate(market_id)
            net_payout = gross_payout - settlement_fee
            settlement_increment = net_payout - position.avg_entry_price * position.quantity

            self.portfolio.cash += net_payout
            self.portfolio.realized_pnl += settlement_increment
            position.realized_pnl += settlement_increment
            position.fees_paid += settlement_fee
            position.quantity = 0.0
            position.closed_ts = resolution_ts
            self.executor._persist_and_evict_position(self.portfolio, position_key)

    def _update_mark_for_market(self, market: MarketState) -> None:
        if (yes_position := self.portfolio.positions.get(_position_key(market.market_id, False))) is not None:
            self.portfolio.last_known_mids[_position_key(market.market_id, False)] = self.executor._position_mark_price(
                market,
                yes_position,
            )
        else:
            self.portfolio.last_known_mids[_position_key(market.market_id, False)] = market.mid
        if (no_position := self.portfolio.positions.get(_position_key(market.market_id, True))) is not None:
            self.portfolio.last_known_mids[_position_key(market.market_id, True)] = self.executor._position_mark_price(
                market,
                no_position,
            )

    def _fill_to_dict(self, fill: FillResult) -> dict[str, Any]:
        return {
            "order_id": fill.order_id,
            "market_id": fill.market_id,
            "side": fill.side,
            "price": fill.price,
            "quantity": fill.quantity,
            "fee_usdc": fill.fee_usdc,
            "rebate_usdc": fill.rebate_usdc,
            "impact_bps": fill.impact_bps,
            "fill_delay_seconds": fill.fill_delay_seconds,
        }

    def _get_resolution(self, market_id: str) -> Any:
        if market_id not in self._resolution_cache:
            self._resolution_cache[market_id] = db.get_resolution(self.conn, market_id)
        return self._resolution_cache[market_id]

    def _next_order_id(self) -> str:
        return str(uuid.uuid4())


class TradingEnvironment:
    """Universal trading environment for Polymarket."""

    def __init__(
        self,
        db_path: str | Path,
        starting_cash: float = 1_000.0,
        market_ids: list[str] | None = None,
        split: str = "val",
        *,
        auto_order_cash_fraction: float = 0.25,
        top_related_markets: int = DEFAULT_TOP_RELATED,
        feature_lookback: int = DEFAULT_FEATURE_LOOKBACK,
        enable_ml_predictions: bool = True,
        random_seed: int | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.conn = db.connect(db_path)
        db.init_db(self.conn)
        self._core = _TradingCore(
            conn=self.conn,
            starting_cash=starting_cash,
            auto_order_cash_fraction=auto_order_cash_fraction,
            top_related_markets=top_related_markets,
            feature_lookback=feature_lookback,
            enable_ml_predictions=enable_ml_predictions,
            random_seed=random_seed,
        )
        self.split = split
        self.market_ids = market_ids or self._market_ids_for_split(split)
        if not self.market_ids:
            raise ValueError(f"No markets available for split {split!r}")
        self._episode: _MarketEpisode | None = None

    def reset(self, market_id: str | None = None) -> TradingState:
        self._core.reset_portfolio()
        selected_market = market_id or self._core._rng.choice(self.market_ids)
        if selected_market not in self.market_ids:
            raise ValueError(f"market_id={selected_market!r} is not part of this environment")

        rows = self.conn.execute(
            """
            SELECT ts, status, best_bid, best_ask, mid, last_trade, volume_1m,
                   volume_24h, open_interest, tick_size
            FROM market_snapshots
            WHERE market_id = ?
            ORDER BY ts ASC
            """,
            (selected_market,),
        ).fetchall()
        if not rows:
            raise ValueError(f"Market {selected_market!r} has no snapshots")
        resolution = db.get_resolution(self.conn, selected_market)
        resolution_ts = (
            datetime.fromisoformat(str(resolution["resolution_ts"]))
            if resolution is not None and resolution["resolution_ts"] is not None
            else None
        )
        self._episode = _MarketEpisode(
            market_id=selected_market,
            snapshot_rows=list(rows),
            resolution_ts=resolution_ts,
            index=0,
            done=False,
        )
        return self.get_state()

    def get_state(self) -> TradingState:
        if self._episode is None:
            return self.reset()
        return self._core.build_state(self._episode)

    def get_available_actions(self) -> list[Action]:
        if self._episode is None:
            self.reset()
        assert self._episode is not None
        return self._core.available_actions(self._episode)

    def step(self, action: Action) -> StepResult:
        if self._episode is None:
            self.reset()
        assert self._episode is not None
        return self._core.step_episode(self._episode, action)

    def step_multi(self, actions: list[Action]) -> StepResult:
        """Execute multiple actions at the same timestamp, then advance.

        On real Polymarket a trader can place multiple orders simultaneously
        (e.g., buy YES + set a limit sell).  This method accepts a list of
        actions, executes them all at the current snapshot, and then advances
        to the next snapshot -- unlike ``step()`` which handles only one action
        per time step.
        """
        if self._episode is None:
            self.reset()
        assert self._episode is not None
        return self._core.step_multi_episode(self._episode, actions)

    @property
    def done(self) -> bool:
        return self._episode.done if self._episode is not None else False

    @property
    def portfolio_value(self) -> float:
        if self._episode is None:
            return self._core.portfolio.cash
        market = self._core._market_state_for_episode(self._episode)
        return self._core.portfolio_value(active_markets={self._episode.market_id: market})

    def _market_ids_for_split(self, split: str, *, allow_holdout: bool = False) -> list[str]:
        if split == "holdout" and not allow_holdout:
            raise ValueError("Holdout set is locked. Use --final-eval to unlock.")

        if split == "all":
            condition = ""
            params: tuple[str, ...] = ()
        elif split == "train":
            condition = "WHERE split_ts < ?"
            params = (TRAIN_CUTOFF,)
        elif split == "val":
            condition = "WHERE split_ts >= ? AND split_ts < ?"
            params = (TRAIN_CUTOFF, VAL_CUTOFF)
        elif split == "test":
            condition = "WHERE split_ts >= ? AND split_ts < ?"
            params = (VAL_CUTOFF, HOLDOUT_CUTOFF)
        elif split == "holdout":
            condition = "WHERE split_ts >= ?"
            params = (HOLDOUT_CUTOFF,)
        else:
            raise ValueError(f"Unknown split {split!r}")

        rows = self.conn.execute(
            f"""
            WITH market_bounds AS (
                SELECT m.market_id, COALESCE(m.resolution_ts, MAX(s.ts)) AS split_ts
                FROM markets m
                JOIN market_snapshots s ON s.market_id = m.market_id
                GROUP BY m.market_id
            )
            SELECT market_id
            FROM market_bounds
            {condition}
            ORDER BY market_id
            """,
            params,
        ).fetchall()
        return [str(row["market_id"]) for row in rows]


class MultiMarketEnvironment:
    """Shared-cash environment for stepping multiple markets in one portfolio."""

    def __init__(
        self,
        db_path: str | Path,
        starting_cash: float = 1_000.0,
        market_ids: list[str] | None = None,
        split: str = "val",
        *,
        auto_order_cash_fraction: float = 0.25,
        top_related_markets: int = DEFAULT_TOP_RELATED,
        feature_lookback: int = DEFAULT_FEATURE_LOOKBACK,
        enable_ml_predictions: bool = True,
        random_seed: int | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.conn = db.connect(db_path)
        db.init_db(self.conn)
        self._core = _TradingCore(
            conn=self.conn,
            starting_cash=starting_cash,
            auto_order_cash_fraction=auto_order_cash_fraction,
            top_related_markets=top_related_markets,
            feature_lookback=feature_lookback,
            enable_ml_predictions=enable_ml_predictions,
            random_seed=random_seed,
        )
        self._single_env = TradingEnvironment(
            db_path=db_path,
            starting_cash=starting_cash,
            market_ids=market_ids,
            split=split,
            auto_order_cash_fraction=auto_order_cash_fraction,
            top_related_markets=top_related_markets,
            feature_lookback=feature_lookback,
            enable_ml_predictions=enable_ml_predictions,
            random_seed=random_seed,
        )
        self.market_ids = market_ids or self._single_env.market_ids
        self._episodes: dict[str, _MarketEpisode] = {}

    def reset(self, market_ids: list[str] | None = None) -> MultiMarketState:
        self._core.reset_portfolio()
        selected_market_ids = market_ids or self.market_ids
        episodes: dict[str, _MarketEpisode] = {}
        for market_id in selected_market_ids:
            rows = self.conn.execute(
                """
                SELECT ts, status, best_bid, best_ask, mid, last_trade, volume_1m,
                       volume_24h, open_interest, tick_size
                FROM market_snapshots
                WHERE market_id = ?
                ORDER BY ts ASC
                """,
                (market_id,),
            ).fetchall()
            if not rows:
                continue
            resolution = db.get_resolution(self.conn, market_id)
            resolution_ts = (
                datetime.fromisoformat(str(resolution["resolution_ts"]))
                if resolution is not None and resolution["resolution_ts"] is not None
                else None
            )
            episodes[market_id] = _MarketEpisode(
                market_id=market_id,
                snapshot_rows=list(rows),
                resolution_ts=resolution_ts,
            )
        if not episodes:
            raise ValueError("No active markets found for reset()")
        self._episodes = episodes
        return self.get_state()

    def get_state(self) -> MultiMarketState:
        if not self._episodes:
            self.reset()
        markets: dict[str, TradingState] = {}
        active_market_states: dict[str, MarketState] = {}
        latest_ts: datetime | None = None
        for market_id, episode in self._episodes.items():
            markets[market_id] = self._core.build_state(episode)
            active_market_states[market_id] = self._core._market_state_for_episode(episode)
            latest_ts = (
                active_market_states[market_id].ts
                if latest_ts is None
                else max(latest_ts, active_market_states[market_id].ts)
            )
        positions = self._core._position_infos(active_markets=active_market_states)
        return MultiMarketState(
            timestamp=latest_ts or datetime.now(),
            cash=self._core.portfolio.cash,
            portfolio_value=self._core.portfolio_value(active_markets=active_market_states),
            positions=positions,
            markets=markets,
        )

    def step(self, actions: dict[str, Action]) -> dict[str, StepResult]:
        if not self._episodes:
            self.reset()
        ordered_market_ids = sorted(
            self._episodes,
            key=lambda market_id: (self._episodes[market_id].current_ts, market_id),
        )
        results: dict[str, StepResult] = {}
        for market_id in ordered_market_ids:
            episode = self._episodes[market_id]
            if episode.done:
                continue
            action = actions.get(market_id, Action.hold())
            results[market_id] = self._core.step_episode(episode, action)
        return results


def _position_key(market_id: str, is_no_bet: bool) -> str:
    return f"{market_id}:NO" if is_no_bet else market_id
