from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from .types import FillResult, MarketState, OrderIntent


@dataclass
class PolymarketFeeModel:
    @staticmethod
    def taker_fee_usdc(
        *,
        price: float,
        quantity: float,
        fees_enabled: bool,
        fee_rate: float,
        exponent: float,
    ) -> float:
        if not fees_enabled or quantity <= 0:
            return 0.0
        fee = quantity * price * fee_rate * ((price * (1.0 - price)) ** exponent)
        return round(max(0.0, fee), 4)

    @staticmethod
    def maker_rebate_usdc(
        *,
        taker_fee_usdc: float,
        maker_rebate_rate: float,
        eligible: bool = False,
    ) -> float:
        if not eligible:
            return 0.0
        return round(max(0.0, taker_fee_usdc * maker_rebate_rate), 4)


@dataclass
class FillEstimate:
    quantity: float
    vwap_price: float
    impact_bps: float
    delay_seconds: float
    liquidity_role: str


@dataclass
class MarketSimulator:
    latent_impact_coefficient: float = 0.02
    passive_horizon_seconds: float = 300.0

    def simulate(
        self,
        *,
        order_id: str,
        market: MarketState,
        next_market: MarketState | None,
        intent: OrderIntent,
    ) -> list[FillResult]:
        estimate = (
            self._simulate_aggressive(market, intent)
            if intent.liquidity_intent == "aggressive"
            else self._simulate_passive(market, next_market, intent)
        )
        if estimate.quantity <= 0:
            return []

        fee_usdc = (
            PolymarketFeeModel.taker_fee_usdc(
                price=estimate.vwap_price,
                quantity=estimate.quantity,
                fees_enabled=market.fees_enabled and estimate.liquidity_role == "taker",
                fee_rate=market.fee_rate,
                exponent=market.fee_exponent,
            )
            if estimate.liquidity_role == "taker"
            else 0.0
        )
        rebate_usdc = (
            PolymarketFeeModel.maker_rebate_usdc(
                taker_fee_usdc=fee_usdc,
                maker_rebate_rate=market.maker_rebate_rate,
                eligible=False,
            )
            if estimate.liquidity_role == "maker"
            else 0.0
        )
        return [
            FillResult(
                order_id=order_id,
                market_id=market.market_id,
                strategy_name=intent.strategy_name,
                fill_ts=market.ts,
                side=intent.side,
                liquidity_role="maker" if estimate.liquidity_role == "maker" else "taker",
                price=round(estimate.vwap_price, 4),
                quantity=round(estimate.quantity, 4),
                fee_usdc=fee_usdc,
                rebate_usdc=rebate_usdc,
                impact_bps=round(estimate.impact_bps, 2),
                fill_delay_seconds=round(estimate.delay_seconds, 2),
            )
        ]

    def _simulate_aggressive(self, market: MarketState, intent: OrderIntent) -> FillEstimate:
        levels = [level for level in market.orderbook if level.side == ("ask" if intent.side == "buy" else "bid")]
        levels.sort(key=lambda level: level.level_no)
        remaining = intent.requested_quantity
        fills: list[tuple[float, float]] = []
        visible_depth = 0.0
        worst_visible_price = intent.limit_price

        for level in levels:
            visible_depth += level.quantity
            worst_visible_price = level.price
            if remaining <= 0:
                break
            tradable = min(remaining, level.quantity)
            if tradable > 0:
                fills.append((level.price, tradable))
                remaining -= tradable

        impact_bps = 0.0
        if remaining > 0:
            impact_bps = self.latent_impact_coefficient * math.sqrt(
                remaining / max(visible_depth, 1.0)
            ) * 10_000.0
            tick_move = round(impact_bps / 10_000.0, 4)
            residual_price = worst_visible_price + tick_move if intent.side == "buy" else worst_visible_price - tick_move
            residual_price = min(0.999, max(0.001, residual_price))
            fills.append((residual_price, remaining))

        total_qty = sum(qty for _, qty in fills)
        vwap = sum(price * qty for price, qty in fills) / max(total_qty, 1e-9)
        return FillEstimate(
            quantity=total_qty,
            vwap_price=vwap,
            impact_bps=impact_bps,
            delay_seconds=0.0,
            liquidity_role="taker",
        )

    def _simulate_passive(
        self,
        market: MarketState,
        next_market: MarketState | None,
        intent: OrderIntent,
    ) -> FillEstimate:
        if next_market is None:
            return FillEstimate(0.0, intent.limit_price, 0.0, self.passive_horizon_seconds, "maker")
        book_side = "bid" if intent.side == "buy" else "ask"
        queue_ahead = sum(
            level.quantity
            for level in market.orderbook
            if level.side == book_side and abs(level.price - intent.limit_price) < market.tick_size / 2.0 + 1e-12
        )
        future_flow = max(next_market.volume_1m * 0.35, 0.0)
        fillable = max(0.0, future_flow - queue_ahead)
        quantity = min(intent.requested_quantity, fillable)
        delay = self.passive_horizon_seconds if quantity > 0 else self.passive_horizon_seconds * 2.0
        return FillEstimate(
            quantity=quantity,
            vwap_price=intent.limit_price,
            impact_bps=0.0,
            delay_seconds=delay,
            liquidity_role="maker",
        )


def new_order_id() -> str:
    return str(uuid.uuid4())
