from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from .types import FillResult, MarketState, OrderIntent, OrderLevel

logger = logging.getLogger(__name__)

MIN_CONTRACT_PRICE = 0.001
MAX_CONTRACT_PRICE = 0.999


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
        # Polymarket official formula (docs.polymarket.com/trading/fees):
        #   fee = C * p * feeRate * (p * (1-p))^exponent
        # Crypto:  feeRate=0.25,   exponent=2 (peak 1.56% at p=0.50)
        # Sports:  feeRate=0.0175, exponent=1 (peak 0.44% at p=0.50)
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
    minimum_fill_quantity: float = 0.01
    passive_queue_ahead_ratio: float = 0.5
    passive_flow_capture_rate: float = 0.35
    empty_book_impact_multiplier: float = 2.5
    maker_rebate_eligible: bool = True

    def simulate(
        self,
        *,
        order_id: str,
        market: MarketState,
        next_market: MarketState | None,
        intent: OrderIntent,
    ) -> list[FillResult]:
        effective_market = self._market_for_intent(market, intent)
        effective_next_market = self._market_for_intent(next_market, intent) if next_market is not None else None
        if intent.order_type == "post_only" and self._would_cross_spread(effective_market, intent):
            return []
        estimate = (
            self._simulate_aggressive(effective_market, intent)
            if intent.liquidity_intent == "aggressive"
            else self._simulate_passive(effective_market, effective_next_market, intent)
        )
        if intent.order_type == "fok" and estimate.quantity + 1e-12 < intent.requested_quantity:
            return []
        if estimate.quantity < self.minimum_fill_quantity:
            return []

        # Polymarket fee structure (2025-2026, docs.polymarket.com/trading/fees):
        # fee = C * p * feeRate * (p * (1-p))^exponent
        # - Political/event markets: NO trading fees (fees_enabled=False)
        # - Crypto markets: feeRate=0.25, exponent=2 (peak ~1.56% at p=0.50)
        # - Sports (NCAAB, Serie A only): feeRate=0.0175, exponent=1 (peak ~0.44%)
        # - Settlement: NO fees on resolution winnings
        taker_fee_equivalent_usdc = PolymarketFeeModel.taker_fee_usdc(
            price=estimate.vwap_price,
            quantity=estimate.quantity,
            fees_enabled=effective_market.fees_enabled,
            fee_rate=effective_market.fee_rate,
            exponent=effective_market.fee_exponent,
        )
        fee_usdc = (
            PolymarketFeeModel.taker_fee_usdc(
                price=estimate.vwap_price,
                quantity=estimate.quantity,
                fees_enabled=effective_market.fees_enabled and estimate.liquidity_role == "taker",
                fee_rate=effective_market.fee_rate,
                exponent=effective_market.fee_exponent,
            )
            if estimate.liquidity_role == "taker"
            else 0.0
        )
        rebate_usdc = (
            PolymarketFeeModel.maker_rebate_usdc(
                taker_fee_usdc=taker_fee_equivalent_usdc,
                maker_rebate_rate=effective_market.maker_rebate_rate,
                eligible=self.maker_rebate_eligible,
            )
            if estimate.liquidity_role == "maker"
            else 0.0
        )
        return [
            FillResult(
                order_id=order_id,
                market_id=market.market_id,
                strategy_name=intent.strategy_name,
                fill_ts=market.ts + timedelta(seconds=estimate.delay_seconds),
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

    def _market_for_intent(self, market: MarketState | None, intent: OrderIntent) -> MarketState | None:
        if market is None or not intent.is_no_bet:
            return market
        return self._complement_market(market)

    def _complement_market(self, market: MarketState) -> MarketState:
        return MarketState(
            market_id=market.market_id,
            title=market.title,
            domain=market.domain,
            market_type=market.market_type,
            ts=market.ts,
            status=market.status,
            best_bid=self._clamp_price(1.0 - market.best_ask),
            best_ask=self._clamp_price(1.0 - market.best_bid),
            mid=self._clamp_price(1.0 - market.mid),
            last_trade=self._clamp_price(1.0 - market.last_trade),
            volume_1m=market.volume_1m,
            volume_24h=market.volume_24h,
            open_interest=market.open_interest,
            tick_size=market.tick_size,
            rules_text=market.rules_text,
            additional_context=market.additional_context,
            resolution_ts=market.resolution_ts,
            fees_enabled=market.fees_enabled,
            fee_rate=market.fee_rate,
            fee_exponent=market.fee_exponent,
            maker_rebate_rate=market.maker_rebate_rate,
            orderbook=[
                OrderLevel(
                    side="ask" if level.side == "bid" else "bid",
                    price=self._clamp_price(1.0 - level.price),
                    quantity=level.quantity,
                    level_no=level.level_no,
                )
                for level in market.orderbook
                if level.quantity > 0
            ],
            tags=list(market.tags),
        )

    def simulate_market_order(
        self,
        market: MarketState,
        side: Literal["buy", "sell"],
        quantity: float,
    ) -> FillEstimate:
        return self._simulate_taker_book_walk(
            market=market,
            side=side,
            quantity=quantity,
            limit_price=None,
        )

    def _simulate_aggressive(self, market: MarketState, intent: OrderIntent) -> FillEstimate:
        return self._simulate_taker_book_walk(
            market=market,
            side=intent.side,
            quantity=intent.requested_quantity,
            limit_price=intent.limit_price,
        )

    def _simulate_passive(
        self,
        market: MarketState,
        next_market: MarketState | None,
        intent: OrderIntent,
    ) -> FillEstimate:
        book_side = "bid" if intent.side == "buy" else "ask"
        visible_depth_at_price = sum(
            level.quantity
            for level in market.orderbook
            if level.side == book_side and self._price_matches(level.price, intent.limit_price, market.tick_size)
        )
        if intent.requested_quantity < self.minimum_fill_quantity:
            return FillEstimate(0.0, intent.limit_price, 0.0, self.passive_horizon_seconds, "maker")

        queue_ahead = visible_depth_at_price * self.passive_queue_ahead_ratio
        if self._improves_same_side_quote(market=market, side=intent.side, price=intent.limit_price):
            queue_ahead = 0.0

        future_flow, fill_probability = self._estimate_passive_flow(market=market, next_market=next_market)
        if self._improves_same_side_quote(market=market, side=intent.side, price=intent.limit_price):
            fill_probability = min(1.0, fill_probability * 1.15)

        effective_flow = future_flow * fill_probability
        direct_fill = max(0.0, effective_flow - queue_ahead)
        if direct_fill >= intent.requested_quantity:
            quantity = intent.requested_quantity
        elif queue_ahead > 0:
            queue_relief = min(
                0.5,
                (effective_flow * fill_probability)
                / max(queue_ahead + intent.requested_quantity, self.minimum_fill_quantity),
            )
            quantity = max(direct_fill, intent.requested_quantity * queue_relief)
        else:
            quantity = direct_fill

        if quantity < self.minimum_fill_quantity:
            delay = self.passive_horizon_seconds * (1.5 if next_market is None else 2.0)
            return FillEstimate(0.0, intent.limit_price, 0.0, delay, "maker")

        delay = self.passive_horizon_seconds
        if quantity < intent.requested_quantity:
            delay *= 1.5 if next_market is None else 1.25
        return FillEstimate(
            quantity=min(intent.requested_quantity, quantity),
            vwap_price=intent.limit_price,
            impact_bps=0.0,
            delay_seconds=delay,
            liquidity_role="maker",
        )

    def _simulate_taker_book_walk(
        self,
        *,
        market: MarketState,
        side: Literal["buy", "sell"],
        quantity: float,
        limit_price: float | None,
    ) -> FillEstimate:
        reference_price = self._reference_price(market, side)
        if quantity < self.minimum_fill_quantity:
            return FillEstimate(0.0, reference_price, 0.0, 0.0, "taker")

        book_side = "ask" if side == "buy" else "bid"
        levels = self._sorted_levels(market, book_side)
        best_visible_price = levels[0].price if levels else self._best_opposing_price(market, side)
        spread_cross_impact_bps = self._spread_cross_impact_bps(
            side=side,
            limit_price=limit_price,
            best_visible_price=best_visible_price,
        )

        remaining = quantity
        fills: list[tuple[float, float]] = []
        visible_depth = 0.0
        worst_visible_price: float | None = None

        for level in levels:
            if limit_price is not None and not self._is_marketable(
                side=side,
                execution_price=level.price,
                limit_price=limit_price,
                tick_size=market.tick_size,
            ):
                break
            visible_depth += level.quantity
            worst_visible_price = level.price
            if remaining < self.minimum_fill_quantity:
                break
            tradable = min(remaining, level.quantity)
            if tradable < self.minimum_fill_quantity:
                continue
            fills.append((level.price, tradable))
            remaining -= tradable

        if not levels and remaining >= self.minimum_fill_quantity:
            latent_fill = self._latent_fill(
                market=market,
                side=side,
                quantity=remaining,
                reference_price=reference_price,
                visible_depth=0.0,
                carried_impact_bps=spread_cross_impact_bps,
                limit_price=limit_price,
                empty_book=True,
            )
            if latent_fill is None:
                return FillEstimate(0.0, reference_price, 0.0, 0.0, "taker")
            fills.append((latent_fill[0], latent_fill[1]))
            impact_bps = latent_fill[2]
        elif remaining >= self.minimum_fill_quantity:
            if (
                not fills
                and limit_price is not None
                and best_visible_price is not None
                and not self._is_marketable(
                    side=side,
                    execution_price=best_visible_price,
                    limit_price=limit_price,
                    tick_size=market.tick_size,
                )
            ):
                return FillEstimate(0.0, reference_price, 0.0, 0.0, "taker")
            latent_fill = self._latent_fill(
                market=market,
                side=side,
                quantity=remaining,
                reference_price=worst_visible_price if worst_visible_price is not None else reference_price,
                visible_depth=visible_depth,
                carried_impact_bps=spread_cross_impact_bps,
                limit_price=limit_price,
                empty_book=visible_depth < self.minimum_fill_quantity,
            )
            impact_bps = spread_cross_impact_bps
            if latent_fill is not None:
                fills.append((latent_fill[0], latent_fill[1]))
                impact_bps = max(impact_bps, latent_fill[2])
        else:
            impact_bps = spread_cross_impact_bps

        total_qty = sum(fill_quantity for _, fill_quantity in fills)
        if total_qty < self.minimum_fill_quantity:
            return FillEstimate(0.0, reference_price, 0.0, 0.0, "taker")

        vwap = sum(fill_price * fill_quantity for fill_price, fill_quantity in fills) / total_qty
        return FillEstimate(
            quantity=total_qty,
            vwap_price=vwap,
            impact_bps=impact_bps,
            delay_seconds=0.0,
            liquidity_role="taker",
        )

    def _estimate_passive_flow(
        self,
        *,
        market: MarketState,
        next_market: MarketState | None,
    ) -> tuple[float, float]:
        baseline_volume = market.volume_24h / 1_440.0 if market.volume_24h > 0 else market.volume_1m
        baseline_volume = max(baseline_volume, self.minimum_fill_quantity)
        recent_volume = max(market.volume_1m, 0.0)

        # Use only CURRENT market data for volume projection to avoid look-ahead bias.
        # next_market is still accepted for backward compatibility but its volume is
        # NOT used — fill probability is derived solely from the current snapshot.
        projected_volume = max(recent_volume, baseline_volume)
        trend_ratio = recent_volume / baseline_volume
        fill_probability = min(0.85, max(0.15, 0.25 + 0.3 * min(trend_ratio, 2.0)))

        return projected_volume * self.passive_flow_capture_rate, fill_probability

    def _latent_fill(
        self,
        *,
        market: MarketState,
        side: Literal["buy", "sell"],
        quantity: float,
        reference_price: float,
        visible_depth: float,
        carried_impact_bps: float,
        limit_price: float | None,
        empty_book: bool,
    ) -> tuple[float, float, float] | None:
        if quantity < self.minimum_fill_quantity:
            return None

        depth_anchor = max(visible_depth, quantity * 0.25, 1.0)
        impact_bps = self.latent_impact_coefficient * math.sqrt(quantity / depth_anchor) * 10_000.0
        if empty_book:
            impact_bps = max(impact_bps * self.empty_book_impact_multiplier, market.tick_size * 20_000.0)
        else:
            impact_bps = max(impact_bps, market.tick_size * 10_000.0)
        total_impact_bps = carried_impact_bps + impact_bps

        candidate_price = reference_price + (
            total_impact_bps / 10_000.0 if side == "buy" else -total_impact_bps / 10_000.0
        )
        candidate_price = self._clamp_price(candidate_price)
        if limit_price is not None and not self._is_marketable(
            side=side,
            execution_price=candidate_price,
            limit_price=limit_price,
            tick_size=market.tick_size,
        ):
            return None
        return candidate_price, quantity, total_impact_bps

    def _sorted_levels(self, market: MarketState, book_side: Literal["bid", "ask"]) -> list:
        levels = [level for level in market.orderbook if level.side == book_side and level.quantity > 0]
        if book_side == "ask":
            levels.sort(key=lambda level: (level.price, level.level_no))
        else:
            levels.sort(key=lambda level: (-level.price, level.level_no))
        return levels

    def _reference_price(self, market: MarketState, side: Literal["buy", "sell"]) -> float:
        candidates = (
            [market.mid, market.best_ask, market.last_trade]
            if side == "buy"
            else [market.mid, market.best_bid, market.last_trade]
        )
        for candidate in candidates:
            if MIN_CONTRACT_PRICE <= candidate <= MAX_CONTRACT_PRICE:
                return candidate
        return 0.5

    def _best_opposing_price(self, market: MarketState, side: Literal["buy", "sell"]) -> float:
        best_price = market.best_ask if side == "buy" else market.best_bid
        if MIN_CONTRACT_PRICE <= best_price <= MAX_CONTRACT_PRICE:
            return best_price
        return self._reference_price(market, side)

    def _spread_cross_impact_bps(
        self,
        *,
        side: Literal["buy", "sell"],
        limit_price: float | None,
        best_visible_price: float,
    ) -> float:
        if limit_price is None:
            return 0.0
        if side == "buy" and limit_price > best_visible_price:
            return (limit_price - best_visible_price) * 10_000.0
        if side == "sell" and limit_price < best_visible_price:
            return (best_visible_price - limit_price) * 10_000.0
        return 0.0

    def _improves_same_side_quote(
        self,
        *,
        market: MarketState,
        side: Literal["buy", "sell"],
        price: float,
    ) -> bool:
        tolerance = market.tick_size / 2.0 + 1e-12
        if side == "buy":
            return market.best_bid <= 0 or price > market.best_bid + tolerance
        return market.best_ask <= 0 or price < market.best_ask - tolerance

    def _is_marketable(
        self,
        *,
        side: Literal["buy", "sell"],
        execution_price: float,
        limit_price: float,
        tick_size: float,
    ) -> bool:
        tolerance = tick_size / 2.0 + 1e-12
        if side == "buy":
            return execution_price <= limit_price + tolerance
        return execution_price >= limit_price - tolerance

    def _would_cross_spread(self, market: MarketState | None, intent: OrderIntent) -> bool:
        if market is None:
            return False
        tolerance = market.tick_size / 2.0 + 1e-12
        if intent.side == "buy":
            return intent.limit_price >= market.best_ask - tolerance
        return intent.limit_price <= market.best_bid + tolerance

    def _price_matches(self, lhs: float, rhs: float, tick_size: float) -> bool:
        return abs(lhs - rhs) <= tick_size / 2.0 + 1e-12

    def _clamp_price(self, price: float) -> float:
        return min(MAX_CONTRACT_PRICE, max(MIN_CONTRACT_PRICE, price))


def new_order_id() -> str:
    return str(uuid.uuid4())
