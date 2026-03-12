from __future__ import annotations

import json
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import db
from .grok_replay import ReplayGrokClient
from .market_simulator import MarketSimulator, new_order_id
from .strategies import StrategyEngine
from .types import (
    FillResult,
    PositionState,
    ReplayConfig,
    StrategyConfig,
    ensure_utc,
    isoformat,
)


@dataclass
class StrategyPortfolio:
    cash: float
    positions: dict[str, PositionState] = field(default_factory=dict)
    realized_pnl: float = 0.0


@dataclass
class ReplayEngine:
    conn: sqlite3.Connection
    config: ReplayConfig
    grok: ReplayGrokClient
    strategies: list[StrategyConfig]
    simulator: MarketSimulator = field(default_factory=MarketSimulator)

    def __post_init__(self) -> None:
        self.portfolios = {
            strategy.name: StrategyPortfolio(cash=self.config.starting_cash)
            for strategy in self.strategies
        }
        self.strategy_engine = StrategyEngine(bankroll=self.config.starting_cash)
        self.experiment_id = self.grok.experiment_id
        if self.experiment_id is None:
            raise ValueError("ReplayEngine requires grok.experiment_id")

    def run(self) -> int:
        timestamps = db.get_all_timestamps(self.conn)
        market_ids = db.get_market_ids(self.conn)
        for timestamp in timestamps:
            for market_id in market_ids:
                market = db.get_market_state_as_of(self.conn, market_id, timestamp)
                if market is None:
                    continue
                forecast, prompt_hash, context_hash = self.grok.forecast(market_id, timestamp)
                self._persist_model_output(forecast, prompt_hash, context_hash)
                next_market = db.get_next_market_state(self.conn, market_id, timestamp)
                for strategy in self.strategies:
                    portfolio = self.portfolios[strategy.name]
                    position = portfolio.positions.get(market_id)
                    if self.strategy_engine.should_exit(
                        config=strategy,
                        market=market,
                        forecast=forecast,
                        position=position,
                    ):
                        exit_order = self.strategy_engine.exit_order(
                            config=strategy,
                            market=market,
                            position=position,
                        )
                        self._execute_order(
                            portfolio,
                            market,
                            next_market,
                            exit_order,
                            entry_probability=position.entry_probability if position is not None else market.mid,
                        )
                    orders = self.strategy_engine.decide(
                        config=strategy,
                        market=market,
                        forecast=forecast,
                        position=portfolio.positions.get(market_id),
                    )
                    for order in orders:
                        self._execute_order(
                            portfolio,
                            market,
                            next_market,
                            order,
                            entry_probability=market.best_ask + (order.edge_bps / 10_000.0),
                        )
                    self._mark_portfolio(
                        portfolio=portfolio,
                        strategy_name=strategy.name,
                        market_id=market_id,
                        mark_ts=timestamp,
                    )
            self._settle_resolved_positions(timestamp)
        self.conn.commit()
        self._persist_positions()
        return self.experiment_id

    def _persist_model_output(self, forecast: Any, prompt_hash: str, context_hash: str) -> None:
        self.conn.execute(
            """
            INSERT INTO model_outputs (
                experiment_id, market_id, ts, agent_name, domain, model_id,
                model_release, prompt_hash, context_hash, probability_yes,
                confidence, expected_edge_bps, thesis, reasoning, evidence_json,
                raw_response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.experiment_id,
                forecast.market_id,
                isoformat(forecast.as_of),
                forecast.agent_name,
                forecast.domain,
                forecast.model_id,
                forecast.model_release,
                prompt_hash,
                context_hash,
                forecast.probability_yes,
                forecast.confidence,
                forecast.expected_edge_bps,
                forecast.thesis,
                forecast.reasoning,
                json.dumps(forecast.evidence, sort_keys=True),
                json.dumps(forecast.raw_response, sort_keys=True),
            ),
        )

    def _execute_order(
        self,
        portfolio: StrategyPortfolio,
        market: Any,
        next_market: Any,
        order: Any,
        entry_probability: float,
    ) -> None:
        order_id = new_order_id()
        self.conn.execute(
            """
            INSERT INTO orders (
                order_id, experiment_id, strategy_name, market_id, ts, side,
                liquidity_intent, limit_price, requested_quantity, edge_bps,
                kelly_fraction, holding_period_minutes, thesis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                self.experiment_id,
                order.strategy_name,
                order.market_id,
                isoformat(order.ts),
                order.side,
                order.liquidity_intent,
                order.limit_price,
                order.requested_quantity,
                order.edge_bps,
                order.kelly_fraction,
                order.holding_period_minutes,
                order.thesis,
            ),
        )
        fills = self.simulator.simulate(
            order_id=order_id,
            market=market,
            next_market=next_market,
            intent=order,
        )
        total_filled = 0.0
        for fill in fills:
            total_filled += fill.quantity
            self._persist_fill(fill)
            self._apply_fill(portfolio, fill, order.thesis, entry_probability)
        self.conn.execute(
            "UPDATE orders SET filled_quantity = ? WHERE order_id = ?",
            (total_filled, order_id),
        )

    def _persist_fill(self, fill: FillResult) -> None:
        self.conn.execute(
            """
            INSERT INTO fills (
                fill_id, order_id, experiment_id, market_id, strategy_name,
                fill_ts, side, liquidity_role, price, quantity, notional_usdc,
                fee_usdc, rebate_usdc, impact_bps, fill_delay_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                fill.order_id,
                self.experiment_id,
                fill.market_id,
                fill.strategy_name,
                isoformat(fill.fill_ts),
                fill.side,
                fill.liquidity_role,
                fill.price,
                fill.quantity,
                round(fill.price * fill.quantity, 4),
                fill.fee_usdc,
                fill.rebate_usdc,
                fill.impact_bps,
                fill.fill_delay_seconds,
            ),
        )

    def _apply_fill(
        self,
        portfolio: StrategyPortfolio,
        fill: FillResult,
        thesis: str,
        entry_probability: float,
    ) -> None:
        position = portfolio.positions.setdefault(
            fill.market_id,
            PositionState(strategy_name=fill.strategy_name, market_id=fill.market_id),
        )
        notional = fill.price * fill.quantity
        if fill.side == "buy":
            portfolio.cash -= notional + fill.fee_usdc - fill.rebate_usdc
            new_quantity = position.quantity + fill.quantity
            if new_quantity > 0:
                position.avg_entry_price = (
                    (position.avg_entry_price * position.quantity) + (fill.price * fill.quantity)
                ) / new_quantity
            position.quantity = new_quantity
            position.total_opened_quantity += fill.quantity
            if position.opened_ts is None:
                position.opened_ts = fill.fill_ts
                position.entry_probability = entry_probability
                position.thesis = thesis
        else:
            sold_qty = min(position.quantity, fill.quantity)
            portfolio.cash += fill.price * sold_qty - fill.fee_usdc + fill.rebate_usdc
            realized = (fill.price - position.avg_entry_price) * sold_qty - fill.fee_usdc + fill.rebate_usdc
            portfolio.realized_pnl += realized
            position.realized_pnl += realized
            position.realized_pnl_pre_resolution += realized
            position.quantity -= sold_qty
            if position.quantity <= 0:
                position.closed_ts = fill.fill_ts
        position.fees_paid += fill.fee_usdc
        position.rebates_earned += fill.rebate_usdc

    def _mark_portfolio(
        self,
        *,
        portfolio: StrategyPortfolio,
        strategy_name: str,
        market_id: str,
        mark_ts: datetime,
    ) -> None:
        market = db.get_market_state_as_of(self.conn, market_id, mark_ts)
        if market is None:
            return
        position = portfolio.positions.get(market_id)
        position_qty = position.quantity if position is not None else 0.0
        mark_price = market.mid
        inventory_value = position_qty * mark_price
        unrealized = 0.0
        if position is not None and position_qty > 0:
            unrealized = (mark_price - position.avg_entry_price) * position_qty
        equity = portfolio.cash + inventory_value
        self.conn.execute(
            """
            INSERT INTO pnl_marks (
                experiment_id, strategy_name, market_id, ts, cash, position_qty,
                mark_price, inventory_value, equity, realized_pnl, unrealized_pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.experiment_id,
                strategy_name,
                market_id,
                isoformat(mark_ts),
                round(portfolio.cash, 4),
                round(position_qty, 4),
                round(mark_price, 4),
                round(inventory_value, 4),
                round(equity, 4),
                round(portfolio.realized_pnl, 4),
                round(unrealized, 4),
            ),
        )

    def _settle_resolved_positions(self, replay_ts: datetime) -> None:
        for strategy in self.strategies:
            portfolio = self.portfolios[strategy.name]
            for market_id, position in list(portfolio.positions.items()):
                if position.quantity <= 0:
                    continue
                resolution = db.get_resolution(self.conn, market_id)
                if resolution is None:
                    continue
                resolution_ts = datetime.fromisoformat(str(resolution["resolution_ts"]))
                if ensure_utc(replay_ts) < ensure_utc(resolution_ts):
                    continue
                payout = float(resolution["resolved_outcome"]) * position.quantity
                settlement_increment = payout - position.avg_entry_price * position.quantity
                portfolio.cash += payout
                position.realized_pnl += settlement_increment
                portfolio.realized_pnl += settlement_increment
                position.quantity = 0.0
                position.closed_ts = resolution_ts

    def _persist_positions(self) -> None:
        for strategy in self.strategies:
            portfolio = self.portfolios[strategy.name]
            for market_id, position in portfolio.positions.items():
                resolution = db.get_resolution(self.conn, market_id)
                resolved_outcome = None
                resolution_ts = None
                hold_to_resolution_pnl = 0.0
                if resolution is not None:
                    resolved_outcome = float(resolution["resolved_outcome"])
                    resolution_ts = str(resolution["resolution_ts"])
                    if position.total_opened_quantity > 0:
                        hold_to_resolution_pnl = (
                            resolved_outcome - position.avg_entry_price
                        ) * position.total_opened_quantity - position.fees_paid + position.rebates_earned
                status = "closed" if position.quantity == 0 else "open"
                self.conn.execute(
                    """
                    INSERT INTO positions (
                        position_id, experiment_id, strategy_name, market_id,
                        opened_ts, closed_ts, quantity, avg_entry_price,
                        avg_exit_price, status, realized_pnl_pre_resolution,
                        hold_to_resolution_pnl, resolved_outcome, resolution_ts,
                        thesis
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        self.experiment_id,
                        strategy.name,
                        market_id,
                        isoformat(position.opened_ts) if position.opened_ts else None,
                        isoformat(position.closed_ts) if position.closed_ts else resolution_ts,
                        position.quantity,
                        position.avg_entry_price,
                        None,
                        status,
                        round(position.realized_pnl_pre_resolution, 4),
                        round(hold_to_resolution_pnl, 4),
                        resolved_outcome,
                        resolution_ts,
                        position.thesis,
                    ),
                )
        self.conn.commit()
