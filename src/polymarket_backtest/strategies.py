import logging
from dataclasses import dataclass

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


def estimated_fee_bps(price: float, fee_rate: float = 0.02) -> float:
    """Estimate round-trip fee in basis points for a trade at the given price."""
    per_side = min(price, 1.0 - price) * fee_rate
    return per_side * 10_000.0


def kelly_fraction_for_yes(price: float, probability_yes: float) -> float:
    if price >= 1.0:
        return 0.0
    return max(0.0, (probability_yes - price) / max(1.0 - price, 1e-9))


def normalized_ask_price(price: float) -> float:
    if price <= 0 or price >= 1:
        logger.warning("Clamping out-of-range price %.6f to [0.001, 0.999]", price)
    return min(0.999, max(0.001, price))


@dataclass
class StrategyEngine:
    def decide(
        self,
        *,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
        available_cash: float,
    ) -> list[OrderIntent]:
        if market.status not in {"active", "open"}:
            return []
        orders: list[OrderIntent] = []
        if config.family == "carry_only":
            orders.extend(self._carry_only(config, market, forecast, position, available_cash))
        elif config.family in {"news_driven", "edge_based"}:
            orders.extend(self._edge_based(config, market, forecast, position, available_cash))
        elif config.family == "mean_reversion":
            orders.extend(self._mean_reversion(config, market, forecast, position, available_cash))
        elif config.family == "contrarian":
            orders.extend(self._contrarian(config, market, forecast, position, available_cash))
        elif config.family == "momentum":
            orders.extend(self._momentum(config, market, forecast, position, available_cash))
        elif config.family == "volume_breakout":
            orders.extend(self._volume_breakout(config, market, forecast, position, available_cash))
        elif config.family == "resolution_convergence":
            orders.extend(self._resolution_convergence(config, market, forecast, position, available_cash))
        return orders

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
        opened_ts = position.opened_ts
        if opened_ts is None:
            return False
        max_hold = config.max_holding_minutes
        if config.use_time_stop and max_hold is not None:
            age_minutes = (ensure_utc(market.ts) - ensure_utc(opened_ts)).total_seconds() / 60.0
            if age_minutes >= max_hold:
                return True
        return (
            config.use_thesis_stop and forecast.probability_yes < position.entry_probability - config.thesis_stop_delta
        )

    def exit_order(
        self,
        *,
        config: StrategyConfig,
        market: MarketState,
        position: PositionState,
    ) -> OrderIntent:
        return OrderIntent(
            strategy_name=config.name,
            market_id=market.market_id,
            ts=market.ts,
            side="sell",
            liquidity_intent="aggressive",
            limit_price=market.best_bid,
            requested_quantity=position.quantity,
            kelly_fraction=config.kelly_fraction,
            edge_bps=0.0,
            holding_period_minutes=0,
            thesis=f"Exit due to {config.name} stop policy",
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
            quantity = notional / ask_price
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

    def _edge_based(
        self,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
        available_cash: float,
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
                kelly = kelly_fraction_for_yes(ask_price, forecast.probability_yes)
                notional = min(config.max_position_notional, available_cash * config.kelly_fraction * kelly)
                if notional <= 0:
                    return []
                quantity = notional / ask_price
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
        quantity = notional / ask_price
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
            quantity = notional / ask_price
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
        quantity = notional / ask_price
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
        quantity = notional / ask_price
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
        position: PositionState | None,
        available_cash: float,
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

        if position is not None and position.quantity > 0:
            # Pure hold-to-resolution: no early exits
            return []

        # Target mid-range markets (configurable via extreme_low/extreme_high)
        mid_low = config.extreme_low
        mid_high = config.extreme_high
        if not (mid_low <= market.mid <= mid_high):
            return []

        if forecast.confidence < config.min_confidence:
            return []

        ask_price = normalized_ask_price(market.best_ask)
        edge_bps = (forecast.probability_yes - ask_price) * 10_000.0
        fee_bps = estimated_fee_bps(ask_price, market.fee_rate)
        net_edge_bps = edge_bps - fee_bps
        if net_edge_bps < config.edge_threshold_bps:
            return []

        # Scale down for edge-of-range markets (further from 0.50 = less confident)
        half_range = max(0.01, (mid_high - mid_low) / 2.0)
        mid_distance = abs(market.mid - 0.50) / half_range
        mid_factor = max(0.25, 1.0 - mid_distance * 0.70)
        kelly = kelly_fraction_for_yes(ask_price, forecast.probability_yes)
        notional = min(
            config.max_position_notional,
            available_cash * config.kelly_fraction * kelly * mid_factor,
        )
        if notional <= 0:
            return []
        quantity = notional / ask_price
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
                holding_period_minutes=int(hours_to_res * 60),
                thesis=f"Resolution convergence: {hours_to_res:.0f}h to resolution, mid={market.mid:.2f}",
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
