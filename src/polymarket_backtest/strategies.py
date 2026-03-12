from __future__ import annotations

from dataclasses import dataclass

from .types import ForecastOutput, MarketState, OrderIntent, PositionState, StrategyConfig, ensure_utc


def kelly_fraction_for_yes(price: float, probability_yes: float) -> float:
    if price >= 1.0:
        return 0.0
    return max(0.0, (probability_yes - price) / max(1.0 - price, 1e-9))


@dataclass
class StrategyEngine:
    bankroll: float

    def decide(
        self,
        *,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
    ) -> list[OrderIntent]:
        if market.status not in {"active", "open"}:
            return []
        orders: list[OrderIntent] = []
        if config.family == "carry_only":
            orders.extend(self._carry_only(config, market, forecast, position))
        elif config.family in {"news_driven", "deep_research", "cross_market_arb"}:
            orders.extend(self._edge_based(config, market, forecast, position))
        return orders

    def should_exit(
        self,
        *,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
    ) -> bool:
        if position is None or position.quantity <= 0 or position.opened_ts is None:
            return False
        if config.use_time_stop and config.max_holding_minutes is not None:
            age_minutes = (ensure_utc(market.ts) - ensure_utc(position.opened_ts)).total_seconds() / 60.0
            if age_minutes >= config.max_holding_minutes:
                return True
        if config.use_thesis_stop and forecast.probability_yes < position.entry_probability - config.thesis_stop_delta:
            return True
        return False

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
    ) -> list[OrderIntent]:
        if position is not None and position.quantity > 0:
            return []
        if not (config.carry_price_min <= market.best_ask <= config.carry_price_max):
            return []
        edge_bps = (forecast.probability_yes - market.best_ask) * 10_000.0
        if edge_bps < config.edge_threshold_bps:
            return []
        notional = min(
            config.max_position_notional,
            self.bankroll * config.kelly_fraction * max(0.0, kelly_fraction_for_yes(market.best_ask, forecast.probability_yes)),
        )
        if notional <= 0:
            return []
        quantity = notional / market.best_ask
        return [
            OrderIntent(
                strategy_name=config.name,
                market_id=market.market_id,
                ts=market.ts,
                side="buy",
                liquidity_intent="passive" if not config.aggressive_entry else "aggressive",
                limit_price=market.best_ask if config.aggressive_entry else market.best_bid,
                requested_quantity=quantity,
                kelly_fraction=config.kelly_fraction,
                edge_bps=edge_bps,
                holding_period_minutes=config.max_holding_minutes,
                thesis=forecast.thesis,
            )
        ]

    def _edge_based(
        self,
        config: StrategyConfig,
        market: MarketState,
        forecast: ForecastOutput,
        position: PositionState | None,
    ) -> list[OrderIntent]:
        if position is not None and position.quantity > 0:
            return []
        if forecast.confidence < config.min_confidence:
            return []
        edge_bps = (forecast.probability_yes - market.best_ask) * 10_000.0
        if edge_bps < config.edge_threshold_bps:
            return []
        kelly = kelly_fraction_for_yes(market.best_ask, forecast.probability_yes)
        notional = min(config.max_position_notional, self.bankroll * config.kelly_fraction * kelly)
        if notional <= 0:
            return []
        quantity = notional / market.best_ask
        return [
            OrderIntent(
                strategy_name=config.name,
                market_id=market.market_id,
                ts=market.ts,
                side="buy",
                liquidity_intent="aggressive" if config.aggressive_entry else "passive",
                limit_price=market.best_ask if config.aggressive_entry else market.best_bid,
                requested_quantity=quantity,
                kelly_fraction=config.kelly_fraction,
                edge_bps=edge_bps,
                holding_period_minutes=config.max_holding_minutes,
                thesis=forecast.thesis,
            )
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
