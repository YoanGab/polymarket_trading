import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .market_categories import has_any_category
from .types import (
    ForecastOutput,
    MarketState,
    OrderIntent,
    PositionState,
    StrategyConfig,
    ensure_utc,
)

logger = logging.getLogger(__name__)

MIN_ORDER_QUANTITY = 1.0
MIN_ARBITRAGE_PAIR_PRICE = 0.998


def _apply_volume_cap(quantity: float, config: StrategyConfig, market: MarketState) -> float:
    """Cap requested quantity based on market volume when volume_sizing is enabled."""
    if not config.volume_sizing or market.volume_24h <= 0:
        return quantity
    max_quantity = market.volume_24h * config.volume_sizing_fraction
    return min(quantity, max_quantity)


def _pyramid_quantity(
    *,
    config: StrategyConfig,
    market: MarketState,
    position: PositionState,
    net_edge_bps: float,
    ask_price: float,
    available_cash: float,
    forecast_probability: float,
) -> float:
    """Compute add-on quantity for pyramiding into a profitable position.

    Returns 0.0 if pyramiding conditions are not met.
    """
    if not config.allow_pyramiding:
        return 0.0
    # Position must be profitable
    mark = position_mark_price(market, position)
    if mark <= position.avg_entry_price:
        return 0.0
    # Edge must still be positive and above threshold
    if net_edge_bps < config.edge_threshold_bps:
        return 0.0
    # Remaining room under max_position_notional
    current_notional = position.quantity * position.avg_entry_price
    remaining_notional = config.max_position_notional - current_notional
    if remaining_notional <= 0:
        return 0.0
    # Scale the add-on by remaining edge (smaller adds as position grows)
    edge_ratio = net_edge_bps / max(config.edge_threshold_bps, 1.0)
    scale = min(1.0, edge_ratio * 0.5)  # at 2x threshold, add full remaining
    kelly = kelly_fraction_for_probability(ask_price, forecast_probability)
    add_notional = min(
        remaining_notional * scale,
        available_cash * config.kelly_fraction * kelly,
    )
    if add_notional <= 0:
        return 0.0
    qty = add_notional / ask_price
    qty = _apply_volume_cap(qty, config, market)
    if qty < MIN_ORDER_QUANTITY:
        return 0.0
    return qty


def normalized_contract_price(price: float) -> float:
    if price <= 0 or price >= 1:
        logger.warning("Clamping out-of-range price %.6f to [0.001, 0.999]", price)
    return min(0.999, max(0.001, price))


def estimated_fee_bps(price: float, fee_rate: float = 0.02) -> float:
    """Estimate round-trip fee in basis points for a trade at the given price."""
    per_side = min(price, 1.0 - price) * fee_rate
    return per_side * 10_000.0


def kelly_fraction_for_probability(price: float, probability: float) -> float:
    if price >= 1.0:
        return 0.0
    return max(0.0, (probability - price) / max(1.0 - price, 1e-9))


def kelly_fraction_for_yes(price: float, probability_yes: float) -> float:
    return kelly_fraction_for_probability(price, probability_yes)


def normalized_ask_price(price: float) -> float:
    return normalized_contract_price(price)


def no_ask_price(market: MarketState) -> float:
    return normalized_contract_price(1.0 - market.best_bid)


def no_bid_price(market: MarketState) -> float:
    return normalized_contract_price(1.0 - market.best_ask)


def held_contract_probability(forecast: ForecastOutput, *, is_no_bet: bool) -> float:
    return 1.0 - forecast.probability_yes if is_no_bet else forecast.probability_yes


def position_mark_price(market: MarketState, position: PositionState) -> float:
    if position.is_no_bet:
        return normalized_contract_price(1.0 - market.mid)
    return normalized_contract_price(market.mid)


@dataclass
class StrategyEngine:
    def _market_category_allowed(self, config: StrategyConfig, market: MarketState) -> bool:
        if config.allowed_categories and not has_any_category(market.tags, config.allowed_categories):
            return False
        return not (config.blocked_categories and has_any_category(market.tags, config.blocked_categories))

    def decide(
        self,
        *,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
        available_cash: float,
        no_position: PositionState | None = None,
        portfolio_cash: float | None = None,
        starting_cash: float | None = None,
        total_invested: float | None = None,
        on_missed_trade: Callable[[float, str], None] | None = None,
        related_market_prices: dict[str, dict[str, Any]] | None = None,
    ) -> list[OrderIntent]:
        if market.status not in {"active", "open"}:
            return []
        if not self._market_category_allowed(config, market):
            return []
        yes_position = position if position is None or not position.is_no_bet else None
        held_no_position = no_position if no_position is None or no_position.is_no_bet else None
        orders: list[OrderIntent] = []
        if config.family == "carry_only":
            orders.extend(self._carry_only(config, market, forecast, yes_position, available_cash))
        elif config.family == "arbitrage":
            orders.extend(
                self._arbitrage(
                    config,
                    market,
                    available_cash,
                    portfolio_cash=portfolio_cash,
                    starting_cash=starting_cash,
                    total_invested=total_invested,
                    on_missed_trade=on_missed_trade,
                )
            )
        elif config.family in {"news_driven", "edge_based"}:
            orders.extend(
                self._edge_based(
                    config,
                    market,
                    forecast,
                    yes_position,
                    available_cash,
                    portfolio_cash=portfolio_cash,
                    starting_cash=starting_cash,
                    total_invested=total_invested,
                    on_missed_trade=on_missed_trade,
                )
            )
        elif config.family == "mean_reversion":
            orders.extend(self._mean_reversion(config, market, forecast, yes_position, available_cash))
        elif config.family == "contrarian":
            orders.extend(self._contrarian(config, market, forecast, yes_position, available_cash))
        elif config.family == "momentum":
            orders.extend(self._momentum(config, market, forecast, yes_position, available_cash))
        elif config.family == "volume_breakout":
            orders.extend(self._volume_breakout(config, market, forecast, yes_position, available_cash))
        elif config.family == "market_making":
            orders.extend(
                self._market_making(
                    config,
                    market,
                    yes_position,
                    held_no_position,
                    available_cash,
                )
            )
        elif config.family == "resolution_convergence":
            orders.extend(
                self._resolution_convergence(
                    config,
                    market,
                    forecast,
                    yes_position,
                    held_no_position,
                    available_cash,
                    portfolio_cash=portfolio_cash,
                    starting_cash=starting_cash,
                    total_invested=total_invested,
                    on_missed_trade=on_missed_trade,
                )
            )
        return orders

    def _market_making(
        self,
        config: StrategyConfig,
        market: MarketState,
        yes_position: PositionState | None,
        no_position: PositionState | None,
        available_cash: float,
    ) -> list[OrderIntent]:
        yes_inventory = yes_position.quantity if yes_position is not None and yes_position.quantity > 0 else 0.0
        no_inventory = no_position.quantity if no_position is not None and no_position.quantity > 0 else 0.0
        net_yes_inventory = yes_inventory - no_inventory

        half_spread = max(market.tick_size, (config.mm_spread_bps / 10_000.0) / 2.0)
        inventory_ratio = max(-1.0, min(1.0, net_yes_inventory / max(config.mm_max_inventory, 1.0)))
        inventory_skew = half_spread * inventory_ratio
        quote_tick = max(market.tick_size, 0.001)

        bid_price = normalized_contract_price(market.mid - half_spread - inventory_skew)
        ask_price = normalized_contract_price(market.mid + half_spread + inventory_skew)
        bid_price = min(bid_price, normalized_contract_price(max(0.001, market.best_ask - quote_tick)))
        ask_price = max(ask_price, normalized_contract_price(min(0.999, market.best_bid + quote_tick)))
        if bid_price >= ask_price:
            bid_price = normalized_contract_price(max(0.001, market.mid - quote_tick))
            ask_price = normalized_contract_price(min(0.999, market.mid + quote_tick))

        quote_quantity = max(
            MIN_ORDER_QUANTITY,
            min(
                config.mm_max_inventory * 0.1,
                config.max_position_notional / max(market.mid, 0.05),
            ),
        )
        buy_capacity = max(0.0, config.mm_max_inventory - max(0.0, net_yes_inventory))
        buy_quantity = min(
            quote_quantity,
            buy_capacity,
            available_cash / max(bid_price, 1e-9),
        )
        sell_quantity = quote_quantity

        orders: list[OrderIntent] = []
        maker_edge_bps = max(config.mm_spread_bps / 2.0, 0.0)

        if buy_quantity >= MIN_ORDER_QUANTITY:
            orders.append(
                OrderIntent(
                    strategy_name=config.name,
                    market_id=market.market_id,
                    ts=market.ts,
                    side="buy",
                    liquidity_intent="passive",
                    limit_price=bid_price,
                    requested_quantity=buy_quantity,
                    kelly_fraction=config.kelly_fraction,
                    edge_bps=maker_edge_bps,
                    holding_period_minutes=config.max_holding_minutes,
                    thesis=(
                        "Market making bid: quote below mid to earn spread and maker rebate "
                        f"(inventory={net_yes_inventory:.2f})"
                    ),
                )
            )

        if sell_quantity >= MIN_ORDER_QUANTITY:
            orders.append(
                OrderIntent(
                    strategy_name=config.name,
                    market_id=market.market_id,
                    ts=market.ts,
                    side="sell",
                    liquidity_intent="passive",
                    limit_price=ask_price,
                    requested_quantity=sell_quantity,
                    kelly_fraction=config.kelly_fraction,
                    edge_bps=maker_edge_bps,
                    holding_period_minutes=config.max_holding_minutes,
                    thesis=(
                        "Market making ask: quote above mid to earn spread and maker rebate "
                        f"(inventory={net_yes_inventory:.2f})"
                    ),
                )
            )

        return orders

    def _entry_cash_cap(
        self,
        *,
        config: StrategyConfig,
        available_cash: float,
        portfolio_cash: float | None,
        starting_cash: float | None,
        total_invested: float | None,
        edge_bps: float,
        on_missed_trade: Callable[[float, str], None] | None,
    ) -> float:
        effective_available_cash = max(0.0, available_cash)
        effective_portfolio_cash = max(0.0, portfolio_cash if portfolio_cash is not None else effective_available_cash)
        if starting_cash is not None:
            invested_cap = starting_cash * config.max_total_invested_pct
            effective_total_invested = max(
                0.0,
                total_invested if total_invested is not None else starting_cash - effective_portfolio_cash,
            )
            if effective_total_invested > invested_cap:
                if on_missed_trade is not None:
                    on_missed_trade(edge_bps, "max_total_invested_pct")
                return 0.0

        per_position_cash = min(effective_available_cash, effective_portfolio_cash * config.max_portfolio_pct)
        if per_position_cash <= 0.0 and on_missed_trade is not None:
            on_missed_trade(edge_bps, "max_portfolio_pct")
        return per_position_cash

    def should_exit(
        self,
        *,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
    ) -> bool:
        if position is None or position.quantity <= 0:
            return False

        current_price = position_mark_price(market, position)

        if config.profit_target_pct > 0 and position.avg_entry_price < 1.0:
            max_gain = 1.0 - position.avg_entry_price
            current_gain = current_price - position.avg_entry_price
            if current_gain >= max_gain * config.profit_target_pct:
                return True

        opened_ts = position.opened_ts
        if config.time_exit_hours > 0 and opened_ts is not None:
            age_hours = (ensure_utc(market.ts) - ensure_utc(opened_ts)).total_seconds() / 3600.0
            if age_hours >= config.time_exit_hours:
                return True

        max_hold = config.max_holding_minutes
        if config.use_time_stop and max_hold is not None and opened_ts is not None:
            age_minutes = (ensure_utc(market.ts) - ensure_utc(opened_ts)).total_seconds() / 60.0
            if age_minutes >= max_hold:
                return True
        held_probability = held_contract_probability(forecast, is_no_bet=position.is_no_bet)
        return config.use_thesis_stop and held_probability < position.entry_probability - config.thesis_stop_delta

    def exit_order(
        self,
        *,
        config: StrategyConfig,
        market: MarketState,
        position: PositionState,
    ) -> OrderIntent:
        exit_price = no_bid_price(market) if position.is_no_bet else market.best_bid
        sell_quantity = position.quantity * config.exit_fraction
        return OrderIntent(
            strategy_name=config.name,
            market_id=market.market_id,
            ts=market.ts,
            side="sell",
            liquidity_intent="aggressive",
            limit_price=exit_price,
            requested_quantity=sell_quantity,
            kelly_fraction=config.kelly_fraction,
            edge_bps=0.0,
            holding_period_minutes=0,
            thesis=f"Exit due to {config.name} stop policy",
            is_no_bet=position.is_no_bet,
        )

    def _carry_only(
        self,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
        available_cash: float,
    ) -> list[OrderIntent]:
        ask_price = normalized_ask_price(market.best_ask)
        if position is None or position.quantity <= 0:
            if not (config.carry_price_min <= ask_price <= config.carry_price_max):
                return []
            edge_bps = (forecast.probability_yes - ask_price) * 10_000.0
            fee_bps = estimated_fee_bps(ask_price, market.fee_rate)
            net_edge_bps = edge_bps - fee_bps
            if net_edge_bps < config.edge_threshold_bps:
                return []
            notional = min(
                config.max_position_notional,
                available_cash
                * config.kelly_fraction
                * max(0.0, kelly_fraction_for_yes(ask_price, forecast.probability_yes)),
            )
            if notional <= 0:
                return []
            quantity = _apply_volume_cap(notional / ask_price, config, market)
            if quantity < MIN_ORDER_QUANTITY:
                return []
            return [
                OrderIntent(
                    strategy_name=config.name,
                    market_id=market.market_id,
                    ts=market.ts,
                    side="buy",
                    liquidity_intent="passive" if not config.aggressive_entry else "aggressive",
                    limit_price=ask_price if config.aggressive_entry else market.best_bid,
                    requested_quantity=quantity,
                    kelly_fraction=config.kelly_fraction,
                    edge_bps=net_edge_bps,
                    holding_period_minutes=config.max_holding_minutes,
                    thesis=forecast.thesis,
                )
            ]
        if market.mid < config.carry_exit_threshold:
            return [
                OrderIntent(
                    strategy_name=config.name,
                    market_id=market.market_id,
                    ts=market.ts,
                    side="sell",
                    liquidity_intent="aggressive",
                    limit_price=market.best_bid,
                    requested_quantity=position.quantity,
                    kelly_fraction=config.kelly_fraction,
                    edge_bps=(position.avg_entry_price - market.best_bid) * 10_000.0,
                    holding_period_minutes=0,
                    thesis="Carry exit: market mid moved near zero",
                )
            ]
        return []

    def _arbitrage(
        self,
        config: StrategyConfig,
        market: MarketState,
        available_cash: float,
        *,
        portfolio_cash: float | None = None,
        starting_cash: float | None = None,
        total_invested: float | None = None,
        on_missed_trade: Callable[[float, str], None] | None = None,
    ) -> list[OrderIntent]:
        yes_ask = normalized_ask_price(market.best_ask)
        no_ask = no_ask_price(market)
        pair_price = yes_ask + no_ask
        arbitrage_edge_bps = max(0.0, (1.0 - pair_price) * 10_000.0)
        required_pair_price = 1.0 - max(config.edge_threshold_bps / 10_000.0, 1.0 - MIN_ARBITRAGE_PAIR_PRICE)
        if pair_price >= required_pair_price:
            return []

        pair_cash_cap = self._entry_cash_cap(
            config=config,
            available_cash=available_cash,
            portfolio_cash=portfolio_cash,
            starting_cash=starting_cash,
            total_invested=total_invested,
            edge_bps=arbitrage_edge_bps,
            on_missed_trade=on_missed_trade,
        )
        if pair_cash_cap <= 0:
            return []

        pair_notional = min(config.max_position_notional, pair_cash_cap)
        if pair_notional <= 0 or pair_price <= 0:
            return []

        quantity = _apply_volume_cap(pair_notional / pair_price, config, market)
        if quantity < MIN_ORDER_QUANTITY:
            return []

        thesis = f"Mint/redeem arbitrage: YES ask {yes_ask:.3f} + NO ask {no_ask:.3f} = {pair_price:.3f}"
        return [
            OrderIntent(
                strategy_name=config.name,
                market_id=market.market_id,
                ts=market.ts,
                side="buy",
                liquidity_intent="aggressive",
                limit_price=yes_ask,
                requested_quantity=quantity,
                kelly_fraction=config.kelly_fraction,
                edge_bps=arbitrage_edge_bps,
                holding_period_minutes=0,
                thesis=thesis,
            ),
            OrderIntent(
                strategy_name=config.name,
                market_id=market.market_id,
                ts=market.ts,
                side="buy",
                liquidity_intent="aggressive",
                limit_price=no_ask,
                requested_quantity=quantity,
                kelly_fraction=config.kelly_fraction,
                edge_bps=arbitrage_edge_bps,
                holding_period_minutes=0,
                thesis=thesis,
                is_no_bet=True,
            ),
        ]

    def _edge_based(
        self,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
        available_cash: float,
        *,
        portfolio_cash: float | None = None,
        starting_cash: float | None = None,
        total_invested: float | None = None,
        on_missed_trade: Callable[[float, str], None] | None = None,
    ) -> list[OrderIntent]:
        ask_price = normalized_ask_price(market.best_ask)
        if forecast.confidence < config.min_confidence:
            return []
        if position is None or position.quantity <= 0:
            # Buy YES when forecast says YES is underpriced
            edge_bps = (forecast.probability_yes - ask_price) * 10_000.0
            fee_bps = estimated_fee_bps(ask_price, market.fee_rate)
            net_edge_bps = edge_bps - fee_bps
            if net_edge_bps >= config.edge_threshold_bps:
                per_position_cash = self._entry_cash_cap(
                    config=config,
                    available_cash=available_cash,
                    portfolio_cash=portfolio_cash,
                    starting_cash=starting_cash,
                    total_invested=total_invested,
                    edge_bps=net_edge_bps,
                    on_missed_trade=on_missed_trade,
                )
                if per_position_cash <= 0:
                    return []
                kelly = kelly_fraction_for_yes(ask_price, forecast.probability_yes)
                notional = min(config.max_position_notional, per_position_cash * config.kelly_fraction * kelly)
                if notional <= 0:
                    return []
                quantity = _apply_volume_cap(notional / ask_price, config, market)
                if quantity < MIN_ORDER_QUANTITY:
                    return []
                return [
                    OrderIntent(
                        strategy_name=config.name,
                        market_id=market.market_id,
                        ts=market.ts,
                        side="buy",
                        liquidity_intent="aggressive" if config.aggressive_entry else "passive",
                        limit_price=ask_price if config.aggressive_entry else market.best_bid,
                        requested_quantity=quantity,
                        kelly_fraction=config.kelly_fraction,
                        edge_bps=net_edge_bps,
                        holding_period_minutes=config.max_holding_minutes,
                        thesis=forecast.thesis,
                    )
                ]

            return []
        sell_edge_bps = (market.best_bid - forecast.probability_yes) * 10_000.0
        fee_bps_sell = estimated_fee_bps(market.best_bid, market.fee_rate)
        net_sell_edge_bps = sell_edge_bps - fee_bps_sell
        if forecast.probability_yes < market.best_bid and net_sell_edge_bps >= config.edge_threshold_bps:
            return [
                OrderIntent(
                    strategy_name=config.name,
                    market_id=market.market_id,
                    ts=market.ts,
                    side="sell",
                    liquidity_intent="aggressive",
                    limit_price=market.best_bid,
                    requested_quantity=position.quantity,
                    kelly_fraction=config.kelly_fraction,
                    edge_bps=net_sell_edge_bps,
                    holding_period_minutes=0,
                    thesis=forecast.thesis,
                )
            ]
        return []

    def _mean_reversion(
        self,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
        available_cash: float,
    ) -> list[OrderIntent]:
        """Buy when volume spikes AND price moved sharply away from forecast.

        Idea: sudden volume + large price move often = overreaction.
        Bet on reversion toward the model's fair value.
        """
        if position is not None and position.quantity > 0:
            # Already in a position -- check if mid reverted toward entry
            revert_edge = (market.best_bid - position.avg_entry_price) * 10_000.0
            if revert_edge >= config.edge_threshold_bps:
                return [
                    OrderIntent(
                        strategy_name=config.name,
                        market_id=market.market_id,
                        ts=market.ts,
                        side="sell",
                        liquidity_intent="aggressive",
                        limit_price=market.best_bid,
                        requested_quantity=position.quantity,
                        kelly_fraction=config.kelly_fraction,
                        edge_bps=revert_edge,
                        holding_period_minutes=0,
                        thesis="Mean reversion: price reverted to target",
                    ),
                ]
            return []

        # Entry: need a volume spike + price dislocation from forecast
        avg_volume = market.volume_24h / 24.0 if market.volume_24h > 0 else 1.0
        volume_ratio = (market.volume_1m * 60.0) / max(avg_volume, 1e-9)
        if volume_ratio < config.volume_spike_ratio:
            return []

        ask_price = normalized_ask_price(market.best_ask)
        move_bps = abs(forecast.probability_yes - ask_price) * 10_000.0
        if move_bps < config.reversion_move_bps:
            return []

        # Only buy if forecast says YES is underpriced
        edge_bps = (forecast.probability_yes - ask_price) * 10_000.0
        fee_bps = estimated_fee_bps(ask_price, market.fee_rate)
        net_edge_bps = edge_bps - fee_bps
        if net_edge_bps < config.edge_threshold_bps:
            return []

        kelly = kelly_fraction_for_yes(ask_price, forecast.probability_yes)
        notional = min(config.max_position_notional, available_cash * config.kelly_fraction * kelly)
        if notional <= 0:
            return []
        quantity = _apply_volume_cap(notional / ask_price, config, market)
        if quantity < MIN_ORDER_QUANTITY:
            return []
        return [
            OrderIntent(
                strategy_name=config.name,
                market_id=market.market_id,
                ts=market.ts,
                side="buy",
                liquidity_intent="aggressive",
                limit_price=ask_price,
                requested_quantity=quantity,
                kelly_fraction=config.kelly_fraction,
                edge_bps=net_edge_bps,
                holding_period_minutes=config.max_holding_minutes,
                thesis=f"Mean reversion: volume spike {volume_ratio:.1f}x, price dislocated {move_bps:.0f}bps",
            ),
        ]

    def _contrarian(
        self,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
        available_cash: float,
    ) -> list[OrderIntent]:
        """Fade extreme prices: buy YES when mid < extreme_low, buy NO when mid > extreme_high.

        Idea: markets at 0.95+ or 0.05- often overshoot. If the forecast
        disagrees with the extreme, bet on the correction.
        """
        if position is not None and position.quantity > 0:
            # Exit if mid moved back toward 0.50 (contrarian thesis played out)
            # For nothesis variants, use thesis_stop_delta as exit buffer above extreme_low
            exit_low = config.extreme_low
            if not config.use_thesis_stop:
                exit_low = config.extreme_low + config.thesis_stop_delta
            if exit_low < market.mid < config.extreme_high:
                return [
                    OrderIntent(
                        strategy_name=config.name,
                        market_id=market.market_id,
                        ts=market.ts,
                        side="sell",
                        liquidity_intent="aggressive",
                        limit_price=market.best_bid,
                        requested_quantity=position.quantity,
                        kelly_fraction=config.kelly_fraction,
                        edge_bps=(market.best_bid - position.avg_entry_price) * 10_000.0,
                        holding_period_minutes=0,
                        thesis="Contrarian exit: price normalized from extreme",
                    ),
                ]
            return []

        ask_price = normalized_ask_price(market.best_ask)

        # Buy YES when market is extremely low and forecast is higher
        if market.mid <= config.extreme_low and forecast.probability_yes > ask_price:
            edge_bps = (forecast.probability_yes - ask_price) * 10_000.0
            fee_bps = estimated_fee_bps(ask_price, market.fee_rate)
            net_edge_bps = edge_bps - fee_bps
            if net_edge_bps < config.edge_threshold_bps:
                return []
            kelly = kelly_fraction_for_yes(ask_price, forecast.probability_yes)
            notional = min(config.max_position_notional, available_cash * config.kelly_fraction * kelly)
            if notional <= 0:
                return []
            quantity = _apply_volume_cap(notional / ask_price, config, market)
            if quantity < MIN_ORDER_QUANTITY:
                return []
            return [
                OrderIntent(
                    strategy_name=config.name,
                    market_id=market.market_id,
                    ts=market.ts,
                    side="buy",
                    liquidity_intent="aggressive",
                    limit_price=ask_price,
                    requested_quantity=quantity,
                    kelly_fraction=config.kelly_fraction,
                    edge_bps=net_edge_bps,
                    holding_period_minutes=config.max_holding_minutes,
                    thesis=f"Contrarian: market at {market.mid:.2f}, forecast {forecast.probability_yes:.2f}",
                ),
            ]

        return []

    def _momentum(
        self,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
        available_cash: float,
    ) -> list[OrderIntent]:
        """Follow price direction when momentum aligns with forecast.

        Idea: if mid > last_trade (price rising) AND forecast agrees YES is underpriced,
        ride the momentum. Volume confirms the move isn't noise.
        """
        if position is not None and position.quantity > 0:
            # Exit if momentum reversed (mid dropped below entry)
            if market.mid < position.avg_entry_price:
                return [
                    OrderIntent(
                        strategy_name=config.name,
                        market_id=market.market_id,
                        ts=market.ts,
                        side="sell",
                        liquidity_intent="aggressive",
                        limit_price=market.best_bid,
                        requested_quantity=position.quantity,
                        kelly_fraction=config.kelly_fraction,
                        edge_bps=(market.best_bid - position.avg_entry_price) * 10_000.0,
                        holding_period_minutes=0,
                        thesis="Momentum exit: price reversed below entry",
                    ),
                ]
            return []

        # Need upward momentum: mid > last_trade
        momentum_bps = (market.mid - market.last_trade) * 10_000.0
        if momentum_bps < config.momentum_min_edge_bps:
            return []

        ask_price = normalized_ask_price(market.best_ask)
        edge_bps = (forecast.probability_yes - ask_price) * 10_000.0
        fee_bps = estimated_fee_bps(ask_price, market.fee_rate)
        net_edge_bps = edge_bps - fee_bps
        if net_edge_bps < config.edge_threshold_bps:
            return []

        # Volume confirmation: current volume should be above average
        avg_volume = market.volume_24h / 24.0 if market.volume_24h > 0 else 1.0
        hourly_volume = market.volume_1m * 60.0
        if hourly_volume < avg_volume * 0.8:
            return []

        kelly = kelly_fraction_for_yes(ask_price, forecast.probability_yes)
        notional = min(config.max_position_notional, available_cash * config.kelly_fraction * kelly)
        if notional <= 0:
            return []
        quantity = _apply_volume_cap(notional / ask_price, config, market)
        if quantity < MIN_ORDER_QUANTITY:
            return []
        return [
            OrderIntent(
                strategy_name=config.name,
                market_id=market.market_id,
                ts=market.ts,
                side="buy",
                liquidity_intent="aggressive",
                limit_price=ask_price,
                requested_quantity=quantity,
                kelly_fraction=config.kelly_fraction,
                edge_bps=net_edge_bps,
                holding_period_minutes=config.max_holding_minutes,
                thesis=f"Momentum: price rising {momentum_bps:.0f}bps, forecast confirms edge",
            ),
        ]

    def _volume_breakout(
        self,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
        available_cash: float,
    ) -> list[OrderIntent]:
        """Enter when volume spikes AND price is moving in the direction of forecast.

        Unlike mean_reversion (which fades the move), this follows the move
        when volume confirms a breakout.
        """
        if position is not None and position.quantity > 0:
            # Exit if volume dried up (move exhausted)
            avg_volume = market.volume_24h / 24.0 if market.volume_24h > 0 else 1.0
            hourly_volume = market.volume_1m * 60.0
            if hourly_volume < avg_volume * 0.5:
                return [
                    OrderIntent(
                        strategy_name=config.name,
                        market_id=market.market_id,
                        ts=market.ts,
                        side="sell",
                        liquidity_intent="aggressive",
                        limit_price=market.best_bid,
                        requested_quantity=position.quantity,
                        kelly_fraction=config.kelly_fraction,
                        edge_bps=(market.best_bid - position.avg_entry_price) * 10_000.0,
                        holding_period_minutes=0,
                        thesis="Volume breakout exit: volume dried up",
                    ),
                ]
            return []

        # Need volume spike
        avg_volume = market.volume_24h / 24.0 if market.volume_24h > 0 else 1.0
        volume_ratio = (market.volume_1m * 60.0) / max(avg_volume, 1e-9)
        if volume_ratio < config.volume_spike_ratio:
            return []

        # Price must be moving in the same direction as forecast
        ask_price = normalized_ask_price(market.best_ask)
        price_rising = market.mid > market.last_trade
        forecast_bullish = forecast.probability_yes > ask_price

        if not (price_rising and forecast_bullish):
            return []

        edge_bps = (forecast.probability_yes - ask_price) * 10_000.0
        fee_bps = estimated_fee_bps(ask_price, market.fee_rate)
        net_edge_bps = edge_bps - fee_bps
        if net_edge_bps < config.edge_threshold_bps:
            return []

        kelly = kelly_fraction_for_yes(ask_price, forecast.probability_yes)
        notional = min(config.max_position_notional, available_cash * config.kelly_fraction * kelly)
        if notional <= 0:
            return []
        quantity = _apply_volume_cap(notional / ask_price, config, market)
        if quantity < MIN_ORDER_QUANTITY:
            return []
        return [
            OrderIntent(
                strategy_name=config.name,
                market_id=market.market_id,
                ts=market.ts,
                side="buy",
                liquidity_intent="aggressive",
                limit_price=ask_price,
                requested_quantity=quantity,
                kelly_fraction=config.kelly_fraction,
                edge_bps=net_edge_bps,
                holding_period_minutes=config.max_holding_minutes,
                thesis=f"Volume breakout: {volume_ratio:.1f}x volume, price + forecast aligned",
            ),
        ]

    def _resolution_convergence(
        self,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        yes_position: PositionState | None,
        no_position: PositionState | None,
        available_cash: float,
        *,
        portfolio_cash: float | None = None,
        starting_cash: float | None = None,
        total_invested: float | None = None,
        on_missed_trade: Callable[[float, str], None] | None = None,
    ) -> list[OrderIntent]:
        """Exploit uncertainty in mid-range markets near resolution.

        Idea: markets at 0.30-0.70 close to resolution have the most
        uncertainty premium. If our forecast has high confidence, the
        edge is largest here because the market hasn't converged yet.
        """
        # Must be near resolution
        seconds_to_res = market.seconds_to_resolution
        if seconds_to_res is None:
            return []
        hours_to_res = seconds_to_res / 3600.0
        if hours_to_res > config.resolution_hours_max or hours_to_res <= 0:
            return []

        has_yes_position = yes_position is not None and yes_position.quantity > 0
        has_no_position = no_position is not None and no_position.quantity > 0

        # Target mid-range markets (configurable via extreme_low/extreme_high)
        mid_low = config.extreme_low
        mid_high = config.extreme_high
        if not (mid_low <= market.mid <= mid_high):
            return []

        if forecast.confidence < config.min_confidence:
            return []

        # Scale down for edge-of-range markets (further from 0.50 = less confident)
        half_range = max(0.01, (mid_high - mid_low) / 2.0)
        mid_distance = abs(market.mid - 0.50) / half_range
        if mid_distance < 0.5:
            mid_factor = 1.0
        else:
            mid_factor = max(0.05, 2.0 * (1.0 - mid_distance))
        confidence_factor = min(3.0, max(0.5, (forecast.confidence - 0.55) * 10.0))

        ask_price = normalized_ask_price(market.best_ask)
        edge_bps = (forecast.probability_yes - ask_price) * 10_000.0
        fee_bps = estimated_fee_bps(ask_price, market.fee_rate)
        net_edge_bps = edge_bps - fee_bps
        if net_edge_bps >= config.edge_threshold_bps:
            per_position_cash = self._entry_cash_cap(
                config=config,
                available_cash=available_cash,
                portfolio_cash=portfolio_cash,
                starting_cash=starting_cash,
                total_invested=total_invested,
                edge_bps=net_edge_bps,
                on_missed_trade=on_missed_trade,
            )
            if per_position_cash <= 0:
                return []
            if has_yes_position:
                if not config.allow_pyramiding:
                    quantity = 0.0
                else:
                    # Pyramiding: add to existing profitable position
                    quantity = _pyramid_quantity(
                        config=config,
                        market=market,
                        position=yes_position,
                        net_edge_bps=net_edge_bps,
                        ask_price=ask_price,
                        available_cash=per_position_cash,
                        forecast_probability=forecast.probability_yes,
                    )
            else:
                kelly = kelly_fraction_for_yes(ask_price, forecast.probability_yes)
                notional = min(
                    config.max_position_notional,
                    per_position_cash * config.kelly_fraction * kelly * mid_factor * confidence_factor,
                )
                quantity = notional / ask_price if notional > 0 else 0.0
            quantity = _apply_volume_cap(quantity, config, market)
            if quantity >= MIN_ORDER_QUANTITY:
                label = "Pyramid add" if has_yes_position else "Resolution convergence"
                return [
                    OrderIntent(
                        strategy_name=config.name,
                        market_id=market.market_id,
                        ts=market.ts,
                        side="buy",
                        liquidity_intent="aggressive",
                        limit_price=ask_price,
                        requested_quantity=quantity,
                        kelly_fraction=config.kelly_fraction,
                        edge_bps=net_edge_bps,
                        holding_period_minutes=int(hours_to_res * 60),
                        thesis=(f"{label}: {hours_to_res:.0f}h to resolution, mid={market.mid:.2f}"),
                    ),
                ]

        no_price = no_ask_price(market)
        no_edge_bps = (market.best_bid - forecast.probability_yes) * 10_000.0
        fee_bps_no = estimated_fee_bps(no_price, market.fee_rate)
        net_no_edge = no_edge_bps - fee_bps_no
        if net_no_edge < config.edge_threshold_bps:
            return []

        per_position_cash = self._entry_cash_cap(
            config=config,
            available_cash=available_cash,
            portfolio_cash=portfolio_cash,
            starting_cash=starting_cash,
            total_invested=total_invested,
            edge_bps=net_no_edge,
            on_missed_trade=on_missed_trade,
        )
        if per_position_cash <= 0:
            return []
        probability_no = 1.0 - forecast.probability_yes
        if has_no_position:
            if not config.allow_pyramiding:
                quantity_no = 0.0
            else:
                quantity_no = _pyramid_quantity(
                    config=config,
                    market=market,
                    position=no_position,
                    net_edge_bps=net_no_edge,
                    ask_price=no_price,
                    available_cash=per_position_cash,
                    forecast_probability=probability_no,
                )
        else:
            kelly_no = kelly_fraction_for_probability(no_price, probability_no)
            notional_no = min(
                config.max_position_notional,
                per_position_cash * config.kelly_fraction * kelly_no * mid_factor * confidence_factor,
            )
            quantity_no = notional_no / no_price if notional_no > 0 else 0.0
        quantity_no = _apply_volume_cap(quantity_no, config, market)
        if quantity_no < MIN_ORDER_QUANTITY:
            return []
        label_no = "Pyramid add NO" if has_no_position else "Resolution convergence NO"
        return [
            OrderIntent(
                strategy_name=config.name,
                market_id=market.market_id,
                ts=market.ts,
                side="buy",
                liquidity_intent="aggressive",
                limit_price=no_price,
                requested_quantity=quantity_no,
                kelly_fraction=config.kelly_fraction,
                edge_bps=net_no_edge,
                holding_period_minutes=int(hours_to_res * 60),
                thesis=(f"{label_no}: {hours_to_res:.0f}h to resolution, mid={market.mid:.2f}"),
                is_no_bet=True,
            ),
        ]


def default_strategy_grid() -> list[StrategyConfig]:
    return [
        StrategyConfig(
            name="carry_k05_threshold25",
            family="carry_only",
            kelly_fraction=0.05,
            edge_threshold_bps=25.0,
            max_position_notional=250.0,
            max_holding_minutes=None,
            aggressive_entry=True,
        ),
        StrategyConfig(
            name="news_k15_threshold50",
            family="news_driven",
            kelly_fraction=0.15,
            edge_threshold_bps=50.0,
            max_position_notional=400.0,
            max_holding_minutes=240,
            use_time_stop=True,
            use_thesis_stop=True,
            aggressive_entry=True,
            min_confidence=0.60,
        ),
    ]
