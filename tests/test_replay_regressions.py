from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from statistics import mean, stdev
from types import SimpleNamespace

import pytest

from polymarket_backtest import db
from polymarket_backtest.metrics import (
    compute_calibration_curve,
    compute_periodic_performance,
    compute_sharpe_like,
)
from polymarket_backtest.replay_engine import ReplayEngine, StrategyPortfolio
from polymarket_backtest.strategies import StrategyEngine
from polymarket_backtest.types import (
    FillResult,
    ForecastOutput,
    MarketState,
    OrderIntent,
    OrderLevel,
    PositionState,
    ReplayConfig,
    RestingOrder,
    StrategyConfig,
    dc_replace,
)


def _make_market_state(
    *,
    best_bid: float,
    best_ask: float,
    market_id: str = "test_market",
    tags: list[str] | None = None,
) -> MarketState:
    ts = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    return MarketState(
        market_id=market_id,
        title="Test market",
        domain="test",
        market_type="binary",
        ts=ts,
        status="active",
        best_bid=best_bid,
        best_ask=best_ask,
        mid=round((best_bid + best_ask) / 2.0, 4),
        last_trade=round((best_bid + best_ask) / 2.0, 4),
        volume_1m=100.0,
        volume_24h=1_000.0,
        open_interest=500.0,
        tick_size=0.001,
        rules_text="Test rules",
        additional_context="",
        resolution_ts=ts + timedelta(hours=24),
        fees_enabled=True,
        fee_rate=0.02,
        fee_exponent=1.0,
        maker_rebate_rate=0.0,
        orderbook=[
            OrderLevel(side="bid", price=max(0.001, best_bid), quantity=100.0, level_no=1),
            OrderLevel(side="ask", price=max(0.001, best_ask), quantity=100.0, level_no=1),
        ],
        tags=list(tags or []),
    )


def _make_forecast(*, probability_yes: float = 0.75, confidence: float = 0.8) -> ForecastOutput:
    ts = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    return ForecastOutput(
        agent_name="grok",
        model_id="grok",
        model_release="test",
        as_of=ts,
        market_id="test_market",
        domain="test",
        probability_yes=probability_yes,
        confidence=confidence,
        expected_edge_bps=100.0,
        thesis="Test thesis",
        reasoning="Test reasoning",
        evidence=[],
        raw_response={},
    )


def _make_db_backed_engine() -> tuple[ReplayEngine, StrategyConfig, MarketState]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)

    market = _make_market_state(best_bid=0.49, best_ask=0.51)
    db.add_market(
        conn,
        market_id=market.market_id,
        title=market.title,
        domain=market.domain,
        market_type=market.market_type,
        open_ts=market.ts - timedelta(hours=1),
        close_ts=market.resolution_ts - timedelta(minutes=5) if market.resolution_ts is not None else None,
        resolution_ts=market.resolution_ts,
        status=market.status,
        tags=market.tags,
        fees_enabled=market.fees_enabled,
        fee_rate=market.fee_rate,
        fee_exponent=market.fee_exponent,
        maker_rebate_rate=market.maker_rebate_rate,
    )
    experiment_id = db.create_experiment(
        conn,
        name="resting-orders-test",
        model_id="grok",
        model_release="test",
        system_prompt_hash="test-hash",
        config={"starting_cash": 1_000.0},
    )
    strategy = StrategyConfig(
        name="resting_strategy",
        family="edge_based",
        kelly_fraction=0.1,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=60,
    )
    engine = ReplayEngine(
        conn=conn,
        config=ReplayConfig(experiment_name="resting-orders-test", starting_cash=1_000.0, lookback_minutes=60),
        grok=SimpleNamespace(experiment_id=experiment_id),
        strategies=[strategy],
    )
    return engine, strategy, market


def test_apply_fill_sell_excludes_fees_from_realized_pnl_and_resets_flat_state() -> None:
    engine = object.__new__(ReplayEngine)
    engine._persist_position = lambda position: None
    engine._persist_and_evict_position = ReplayEngine._persist_and_evict_position.__get__(engine, ReplayEngine)
    ts = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    portfolio = StrategyPortfolio(
        cash=100.0,
        positions={
            "test_market": PositionState(
                strategy_name="test_strategy",
                market_id="test_market",
                quantity=10.0,
                avg_entry_price=0.6,
                total_opened_quantity=10.0,
                total_opened_notional=6.0,
                opened_ts=ts - timedelta(hours=1),
                entry_probability=0.7,
                thesis="Original thesis",
            )
        },
    )
    fill = FillResult(
        order_id="order-1",
        market_id="test_market",
        strategy_name="test_strategy",
        fill_ts=ts,
        side="sell",
        liquidity_role="taker",
        price=0.75,
        quantity=10.0,
        fee_usdc=0.02,
        rebate_usdc=0.0,
        impact_bps=0.0,
        fill_delay_seconds=0.0,
    )

    position = portfolio.positions["test_market"]
    ReplayEngine._apply_fill(engine, portfolio, fill, "ignored", 0.0)

    assert portfolio.cash == pytest.approx(107.48)
    assert portfolio.realized_pnl == pytest.approx(1.5)
    assert position.realized_pnl_pre_resolution == pytest.approx(1.5)
    assert position.fees_paid == pytest.approx(0.02)
    assert position.quantity == 0.0
    assert position.opened_ts is None
    assert position.entry_probability == 0.0
    assert position.thesis == ""
    assert position.avg_entry_price == 0.0
    assert "test_market" not in portfolio.positions


def test_apply_fill_tracks_yes_and_no_positions_separately() -> None:
    engine = object.__new__(ReplayEngine)
    engine._persist_position = lambda position: None
    engine._persist_and_evict_position = ReplayEngine._persist_and_evict_position.__get__(engine, ReplayEngine)
    ts = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    portfolio = StrategyPortfolio(cash=100.0)
    yes_fill = FillResult(
        order_id="order-yes",
        market_id="test_market",
        strategy_name="test_strategy",
        fill_ts=ts,
        side="buy",
        liquidity_role="taker",
        price=0.40,
        quantity=10.0,
        fee_usdc=0.0,
        rebate_usdc=0.0,
        impact_bps=0.0,
        fill_delay_seconds=0.0,
    )
    no_fill = FillResult(
        order_id="order-no",
        market_id="test_market",
        strategy_name="test_strategy",
        fill_ts=ts,
        side="buy",
        liquidity_role="taker",
        price=0.30,
        quantity=5.0,
        fee_usdc=0.0,
        rebate_usdc=0.0,
        impact_bps=0.0,
        fill_delay_seconds=0.0,
    )

    ReplayEngine._apply_fill(engine, portfolio, yes_fill, "YES thesis", 0.65)
    ReplayEngine._apply_fill(engine, portfolio, no_fill, "NO thesis", 0.35, is_no_bet=True)

    assert portfolio.cash == pytest.approx(94.5)
    assert set(portfolio.positions) == {"test_market", "test_market:NO"}
    assert portfolio.positions["test_market"].quantity == pytest.approx(10.0)
    assert portfolio.positions["test_market"].is_no_bet is False
    assert portfolio.positions["test_market:NO"].quantity == pytest.approx(5.0)
    assert portfolio.positions["test_market:NO"].is_no_bet is True


def test_normalize_quotes_clamps_zero_quotes_and_preserves_spread() -> None:
    engine = object.__new__(ReplayEngine)
    market = _make_market_state(best_bid=0.0, best_ask=0.0)

    normalized = ReplayEngine._normalize_quotes(engine, market)

    assert normalized.best_bid == pytest.approx(0.001)
    assert normalized.best_ask > normalized.best_bid
    assert normalized.mid == pytest.approx((normalized.best_bid + normalized.best_ask) / 2.0)


def test_execute_order_places_passive_order_as_resting_order() -> None:
    engine, strategy, market = _make_db_backed_engine()
    portfolio = engine.portfolios[strategy.name]
    order = OrderIntent(
        strategy_name=strategy.name,
        market_id=market.market_id,
        ts=market.ts,
        side="buy",
        liquidity_intent="passive",
        limit_price=0.50,
        requested_quantity=10.0,
        kelly_fraction=strategy.kelly_fraction,
        edge_bps=50.0,
        holding_period_minutes=30,
        thesis="Rest bid",
    )

    engine._execute_order(
        portfolio=portfolio,
        market=market,
        next_market=market,
        order=order,
        entry_probability=0.6,
    )

    assert portfolio.positions == {}
    assert len(portfolio.resting_orders) == 1
    resting_order = portfolio.resting_orders[0]
    assert resting_order.limit_price == pytest.approx(0.50)
    assert resting_order.remaining_quantity == pytest.approx(10.0)
    assert resting_order.gtd_expiry == market.ts + timedelta(minutes=30)

    order_row = engine.conn.execute(
        "SELECT liquidity_intent, filled_quantity FROM orders WHERE order_id = ?",
        (resting_order.order_id,),
    ).fetchone()
    assert order_row is not None
    assert order_row["liquidity_intent"] == "passive"
    assert order_row["filled_quantity"] == pytest.approx(0.0)


def test_process_resting_orders_fills_marketable_orders_and_expires_gtd() -> None:
    engine, strategy, market = _make_db_backed_engine()
    portfolio = StrategyPortfolio(cash=1_000.0)
    fill_order = OrderIntent(
        strategy_name=strategy.name,
        market_id=market.market_id,
        ts=market.ts - timedelta(minutes=5),
        side="buy",
        liquidity_intent="passive",
        limit_price=0.52,
        requested_quantity=10.0,
        kelly_fraction=strategy.kelly_fraction,
        edge_bps=60.0,
        holding_period_minutes=30,
        thesis="Fill me later",
    )
    expire_order = OrderIntent(
        strategy_name=strategy.name,
        market_id=market.market_id,
        ts=market.ts - timedelta(minutes=10),
        side="buy",
        liquidity_intent="passive",
        limit_price=0.40,
        requested_quantity=5.0,
        kelly_fraction=strategy.kelly_fraction,
        edge_bps=10.0,
        holding_period_minutes=5,
        thesis="Expire me",
    )
    engine._persist_order("fill-order", fill_order)
    engine._persist_order("expire-order", expire_order)
    portfolio.resting_orders = [
        RestingOrder(
            order_id="fill-order",
            strategy_name=strategy.name,
            market_id=market.market_id,
            placed_ts=fill_order.ts,
            side="buy",
            limit_price=fill_order.limit_price,
            remaining_quantity=fill_order.requested_quantity,
            gtd_expiry=fill_order.ts + timedelta(minutes=30),
        ),
        RestingOrder(
            order_id="expire-order",
            strategy_name=strategy.name,
            market_id=market.market_id,
            placed_ts=expire_order.ts,
            side="buy",
            limit_price=expire_order.limit_price,
            remaining_quantity=expire_order.requested_quantity,
            gtd_expiry=market.ts,
        ),
    ]

    engine._process_resting_orders(
        portfolio=portfolio,
        market=market,
        forecast=_make_forecast(probability_yes=0.62),
    )

    assert portfolio.resting_orders == []
    assert portfolio.cash == pytest.approx(994.8)
    assert portfolio.positions["test_market"].quantity == pytest.approx(10.0)
    assert portfolio.positions["test_market"].entry_probability == pytest.approx(0.62)
    assert portfolio.positions["test_market"].thesis == "Fill me later"

    fill_row = engine.conn.execute("SELECT quantity FROM fills WHERE order_id = 'fill-order'").fetchone()
    expired_fill_row = engine.conn.execute("SELECT quantity FROM fills WHERE order_id = 'expire-order'").fetchone()
    filled_qty_row = engine.conn.execute("SELECT filled_quantity FROM orders WHERE order_id = 'fill-order'").fetchone()
    expired_qty_row = engine.conn.execute(
        "SELECT filled_quantity FROM orders WHERE order_id = 'expire-order'"
    ).fetchone()
    assert fill_row is not None
    assert fill_row["quantity"] == pytest.approx(10.0)
    assert expired_fill_row is None
    assert filled_qty_row is not None and filled_qty_row["filled_quantity"] == pytest.approx(10.0)
    assert expired_qty_row is not None and expired_qty_row["filled_quantity"] == pytest.approx(0.0)


def test_amend_and_cancel_resting_order_update_portfolio_and_order_row() -> None:
    engine, strategy, market = _make_db_backed_engine()
    portfolio = StrategyPortfolio(
        cash=1_000.0,
        resting_orders=[
            RestingOrder(
                order_id="resting-1",
                strategy_name=strategy.name,
                market_id=market.market_id,
                placed_ts=market.ts,
                side="buy",
                limit_price=0.49,
                remaining_quantity=5.0,
            )
        ],
    )
    engine._persist_order(
        "resting-1",
        OrderIntent(
            strategy_name=strategy.name,
            market_id=market.market_id,
            ts=market.ts,
            side="buy",
            liquidity_intent="passive",
            limit_price=0.49,
            requested_quantity=5.0,
            kelly_fraction=strategy.kelly_fraction,
            edge_bps=25.0,
            holding_period_minutes=60,
            thesis="Amend me",
        ),
    )

    amended = engine._amend_resting_order(
        portfolio=portfolio,
        order_id="resting-1",
        new_price=0.48,
        new_quantity=8.0,
    )
    assert amended is True
    assert portfolio.resting_orders[0].limit_price == pytest.approx(0.48)
    assert portfolio.resting_orders[0].remaining_quantity == pytest.approx(8.0)

    order_row = engine.conn.execute(
        "SELECT limit_price, requested_quantity FROM orders WHERE order_id = 'resting-1'"
    ).fetchone()
    assert order_row is not None
    assert order_row["limit_price"] == pytest.approx(0.48)
    assert order_row["requested_quantity"] == pytest.approx(8.0)

    cancelled = engine._cancel_resting_order(portfolio=portfolio, order_id="resting-1")
    assert cancelled is True
    assert portfolio.resting_orders == []


def test_strategy_engine_handles_zero_best_ask_without_division_error() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="carry_test",
        family="carry_only",
        kelly_fraction=0.05,
        edge_threshold_bps=10.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
        carry_price_min=0.001,
        carry_price_max=0.99,
    )

    orders = engine.decide(
        config=config,
        market=_make_market_state(best_bid=0.0, best_ask=0.0),
        forecast=_make_forecast(probability_yes=0.6),
        position=None,
        available_cash=100.0,
    )

    assert orders
    assert orders[0].requested_quantity > 0.0
    assert orders[0].limit_price >= 0.001


def test_strategy_engine_exposes_cancel_and_amend_actions() -> None:
    engine = StrategyEngine()

    cancel_action = engine.cancel_order("order-123")
    amend_action = engine.amend_order("order-123", 0.48, 8.0)

    assert cancel_action.order_id == "order-123"
    assert amend_action.order_id == "order-123"
    assert amend_action.new_price == pytest.approx(0.48)
    assert amend_action.new_quantity == pytest.approx(8.0)


def test_strategy_engine_respects_category_routing() -> None:
    engine = StrategyEngine()
    market = _make_market_state(best_bid=0.49, best_ask=0.51, tags=["Sports"])
    forecast = _make_forecast(probability_yes=0.70, confidence=0.9)

    allowed_config = StrategyConfig(
        name="crypto_only",
        family="edge_based",
        kelly_fraction=0.1,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=60,
        min_confidence=0.6,
        allowed_categories=["Crypto"],
    )
    blocked_config = StrategyConfig(
        name="not_sports",
        family="edge_based",
        kelly_fraction=0.1,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=60,
        min_confidence=0.6,
        blocked_categories=["Sports"],
    )

    assert (
        engine.decide(
            config=allowed_config,
            market=market,
            forecast=forecast,
            position=None,
            available_cash=1_000.0,
        )
        == []
    )
    assert (
        engine.decide(
            config=blocked_config,
            market=market,
            forecast=forecast,
            position=None,
            available_cash=1_000.0,
        )
        == []
    )


def test_arbitrage_strategy_emits_paired_yes_and_no_buys() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="arb_test",
        family="arbitrage",
        kelly_fraction=0.1,
        edge_threshold_bps=0.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
    )
    market = _make_market_state(best_bid=0.505, best_ask=0.492)

    orders = engine.decide(
        config=config,
        market=market,
        forecast=_make_forecast(probability_yes=0.5, confidence=0.5),
        position=None,
        no_position=None,
        available_cash=100.0,
        portfolio_cash=100.0,
        starting_cash=100.0,
        total_invested=0.0,
    )

    assert len(orders) == 2
    assert {order.is_no_bet for order in orders} == {False, True}
    assert all(order.side == "buy" for order in orders)
    assert orders[0].requested_quantity == pytest.approx(orders[1].requested_quantity)
    assert sorted(order.limit_price for order in orders) == pytest.approx([0.492, 0.495])


def test_redeem_matched_pairs_converts_inventory_back_to_cash() -> None:
    engine = object.__new__(ReplayEngine)
    engine.simulator = SimpleNamespace(minimum_fill_quantity=0.01)
    engine._persist_position = lambda position: None
    market = _make_market_state(best_bid=0.49, best_ask=0.51)
    opened_ts = market.ts - timedelta(hours=2)
    portfolio = StrategyPortfolio(
        cash=10.0,
        positions={
            "test_market": PositionState(
                strategy_name="arb_test",
                market_id=market.market_id,
                quantity=5.0,
                avg_entry_price=0.48,
                total_opened_quantity=5.0,
                total_opened_notional=2.4,
                opened_ts=opened_ts,
                entry_probability=0.52,
                thesis="YES leg",
            ),
            "test_market:NO": PositionState(
                strategy_name="arb_test",
                market_id=market.market_id,
                quantity=5.0,
                avg_entry_price=0.50,
                total_opened_quantity=5.0,
                total_opened_notional=2.5,
                opened_ts=opened_ts,
                entry_probability=0.48,
                thesis="NO leg",
                is_no_bet=True,
            ),
        },
    )

    redeemed = ReplayEngine._redeem_matched_pairs(
        engine,
        portfolio=portfolio,
        strategy_name="arb_test",
        market=market,
    )

    assert redeemed == pytest.approx(5.0)
    assert portfolio.cash == pytest.approx(15.0)
    assert portfolio.realized_pnl == pytest.approx(0.1)
    assert portfolio.positions == {}


def test_apply_market_category_metadata_overrides_fees() -> None:
    engine = object.__new__(ReplayEngine)
    engine.market_categories = {
        "crypto_market": ["Crypto"],
        "ncaab_market": ["Sports", "NCAA Basketball"],
        "nba_market": ["Sports", "NBA"],
        "political_market": ["Politics"],
    }

    crypto = ReplayEngine._apply_market_category_metadata(
        engine,
        _make_market_state(best_bid=0.49, best_ask=0.51, market_id="crypto_market"),
    )
    ncaab = ReplayEngine._apply_market_category_metadata(
        engine,
        _make_market_state(best_bid=0.49, best_ask=0.51, market_id="ncaab_market"),
    )
    nba = ReplayEngine._apply_market_category_metadata(
        engine,
        _make_market_state(best_bid=0.49, best_ask=0.51, market_id="nba_market"),
    )
    political = ReplayEngine._apply_market_category_metadata(
        engine,
        _make_market_state(best_bid=0.49, best_ask=0.51, market_id="political_market"),
    )

    # Crypto: feeRate=0.25, exponent=2, maker rebate=20%
    assert crypto.tags == ["Crypto"]
    assert crypto.fees_enabled is True
    assert crypto.fee_rate == pytest.approx(0.25)
    assert crypto.fee_exponent == pytest.approx(2.0)
    assert crypto.maker_rebate_rate == pytest.approx(0.20)
    # NCAAB (fee-bearing sports): feeRate=0.0175, exponent=1, maker rebate=25%
    assert ncaab.fees_enabled is True
    assert ncaab.fee_rate == pytest.approx(0.0175)
    assert ncaab.fee_exponent == pytest.approx(1.0)
    assert ncaab.maker_rebate_rate == pytest.approx(0.25)
    # NBA (non-fee-bearing sports): no fees
    assert nba.fees_enabled is False
    assert nba.fee_rate == pytest.approx(0.0)
    # Political: no fees
    assert political.tags == ["Politics"]
    assert political.fees_enabled is False
    assert political.fee_rate == pytest.approx(0.0)


def test_resolution_convergence_can_buy_no_when_yes_is_overpriced() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="resolution_no",
        family="resolution_convergence",
        kelly_fraction=0.2,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
        min_confidence=0.7,
        resolution_hours_max=72.0,
        extreme_low=0.2,
        extreme_high=0.8,
    )
    market = _make_market_state(best_bid=0.68, best_ask=0.70)
    forecast = _make_forecast(probability_yes=0.45, confidence=0.9)

    orders = engine.decide(
        config=config,
        market=market,
        forecast=forecast,
        position=None,
        available_cash=1_000.0,
    )

    assert len(orders) == 1
    assert orders[0].side == "buy"
    assert orders[0].is_no_bet is True
    assert orders[0].limit_price == pytest.approx(0.32)
    assert "Resolution convergence NO" in orders[0].thesis


def test_resolution_convergence_can_open_no_while_holding_yes() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="resolution_two_sided",
        family="resolution_convergence",
        kelly_fraction=0.2,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
        min_confidence=0.7,
        resolution_hours_max=72.0,
        extreme_low=0.2,
        extreme_high=0.8,
        allow_pyramiding=False,
    )
    market = _make_market_state(best_bid=0.68, best_ask=0.70)
    forecast = _make_forecast(probability_yes=0.45, confidence=0.9)
    yes_position = PositionState(
        strategy_name=config.name,
        market_id=market.market_id,
        quantity=10.0,
        avg_entry_price=0.35,
        total_opened_quantity=10.0,
        total_opened_notional=3.5,
        opened_ts=market.ts - timedelta(hours=2),
        entry_probability=0.60,
    )

    orders = engine.decide(
        config=config,
        market=market,
        forecast=forecast,
        position=yes_position,
        no_position=None,
        available_cash=1_000.0,
    )

    assert len(orders) == 1
    assert orders[0].side == "buy"
    assert orders[0].is_no_bet is True
    assert orders[0].limit_price == pytest.approx(0.32)


def test_resolution_convergence_can_buy_multi_outcome_leg() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="resolution_multi",
        family="resolution_convergence",
        kelly_fraction=0.2,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
        min_confidence=0.7,
        resolution_hours_max=72.0,
    )
    market = dc_replace(
        _make_market_state(best_bid=0.46, best_ask=0.48, market_id="alice"),
        title="Alice wins",
        outcome_count=3,
        outcome_tokens=["alice", "bob", "carol"],
    )
    forecast = _make_forecast(probability_yes=0.60, confidence=0.9)

    orders = engine.decide(
        config=config,
        market=market,
        forecast=forecast,
        position=None,
        available_cash=1_000.0,
        related_market_prices={
            "bob": {"title": "Bob wins", "best_ask": 0.30, "mid": 0.29, "probability_yes": 0.25},
            "carol": {"title": "Carol wins", "best_ask": 0.25, "mid": 0.24, "probability_yes": 0.15},
        },
    )

    assert len(orders) == 1
    assert orders[0].market_id == "alice"
    assert orders[0].side == "buy"
    assert orders[0].is_no_bet is False
    assert "Multi-outcome resolution convergence" in orders[0].thesis


def test_resolution_convergence_buys_all_multi_outcomes_when_asks_sum_below_one() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="resolution_multi_arb",
        family="resolution_convergence",
        kelly_fraction=0.2,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
        min_confidence=0.9,
        resolution_hours_max=72.0,
    )
    market = dc_replace(
        _make_market_state(best_bid=0.28, best_ask=0.30, market_id="alice"),
        title="Alice wins",
        outcome_count=3,
        outcome_tokens=["alice", "bob", "carol"],
    )
    forecast = _make_forecast(probability_yes=0.34, confidence=0.2)

    orders = engine.decide(
        config=config,
        market=market,
        forecast=forecast,
        position=None,
        available_cash=1_000.0,
        portfolio_cash=1_000.0,
        starting_cash=1_000.0,
        total_invested=0.0,
        related_market_prices={
            "bob": {"title": "Bob wins", "best_ask": 0.24, "mid": 0.23, "probability_yes": 0.33},
            "carol": {"title": "Carol wins", "best_ask": 0.20, "mid": 0.19, "probability_yes": 0.33},
        },
    )

    assert [order.market_id for order in orders] == ["alice", "bob", "carol"]
    assert all(order.side == "buy" for order in orders)
    assert all(order.requested_quantity == pytest.approx(orders[0].requested_quantity) for order in orders)
    assert all("Multi-outcome arbitrage" in order.thesis for order in orders)


def test_edge_based_caps_single_position_to_max_portfolio_pct() -> None:
    engine = StrategyEngine()
    capped_config = StrategyConfig(
        name="edge_capped",
        family="edge_based",
        kelly_fraction=1.0,
        edge_threshold_bps=25.0,
        max_position_notional=5_000.0,
        max_holding_minutes=60,
        min_confidence=0.6,
        max_portfolio_pct=0.25,
    )
    uncapped_config = StrategyConfig(
        name="edge_uncapped",
        family="edge_based",
        kelly_fraction=1.0,
        edge_threshold_bps=25.0,
        max_position_notional=5_000.0,
        max_holding_minutes=60,
        min_confidence=0.6,
        max_portfolio_pct=1.0,
    )
    market = _make_market_state(best_bid=0.19, best_ask=0.20)
    forecast = _make_forecast(probability_yes=0.99, confidence=0.9)

    capped_orders = engine.decide(
        config=capped_config,
        market=market,
        forecast=forecast,
        position=None,
        available_cash=1_000.0,
        portfolio_cash=1_000.0,
        starting_cash=1_000.0,
        total_invested=0.0,
    )
    uncapped_orders = engine.decide(
        config=uncapped_config,
        market=market,
        forecast=forecast,
        position=None,
        available_cash=1_000.0,
        portfolio_cash=1_000.0,
        starting_cash=1_000.0,
        total_invested=0.0,
    )

    assert len(capped_orders) == 1
    assert len(uncapped_orders) == 1
    capped_notional = capped_orders[0].requested_quantity * capped_orders[0].limit_price
    uncapped_notional = uncapped_orders[0].requested_quantity * uncapped_orders[0].limit_price
    assert capped_notional <= 250.0 + 1e-6
    assert uncapped_notional > capped_notional * 3.5


def test_resolution_convergence_records_missed_trade_when_capital_is_reserved() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="resolution_reserved",
        family="resolution_convergence",
        kelly_fraction=0.5,
        edge_threshold_bps=25.0,
        max_position_notional=1_000.0,
        max_holding_minutes=None,
        min_confidence=0.7,
        resolution_hours_max=72.0,
        extreme_low=0.2,
        extreme_high=0.8,
    )
    market = _make_market_state(best_bid=0.39, best_ask=0.41)
    forecast = _make_forecast(probability_yes=0.80, confidence=0.9)
    missed: list[tuple[float, str]] = []

    orders = engine.decide(
        config=config,
        market=market,
        forecast=forecast,
        position=None,
        available_cash=0.0,
        portfolio_cash=200.0,
        starting_cash=1_000.0,
        total_invested=800.0,
        on_missed_trade=lambda edge_bps, reason: missed.append((edge_bps, reason)),
    )

    assert orders == []
    assert len(missed) == 1
    assert missed[0][0] > config.edge_threshold_bps
    assert missed[0][1] == "max_portfolio_pct"


def test_should_exit_uses_profit_target_for_no_positions() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="profit_target",
        family="resolution_convergence",
        kelly_fraction=0.1,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
        profit_target_pct=0.5,
    )
    market = _make_market_state(best_bid=0.19, best_ask=0.21)
    forecast = _make_forecast(probability_yes=0.25, confidence=0.8)
    position = PositionState(
        strategy_name="profit_target",
        market_id=market.market_id,
        quantity=10.0,
        avg_entry_price=0.30,
        total_opened_quantity=10.0,
        total_opened_notional=3.0,
        opened_ts=market.ts - timedelta(hours=4),
        entry_probability=0.70,
        is_no_bet=True,
    )

    should_exit = engine.should_exit(
        config=config,
        market=market,
        forecast=forecast,
        position=position,
    )

    assert should_exit is True


def test_exit_order_uses_partial_exit_fraction_after_profit_target() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="partial_profit_target",
        family="edge_based",
        kelly_fraction=0.1,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
        profit_target_pct=0.5,
        exit_fraction=0.5,
    )
    market = _make_market_state(best_bid=0.74, best_ask=0.76)
    forecast = _make_forecast(probability_yes=0.80, confidence=0.8)
    position = PositionState(
        strategy_name=config.name,
        market_id=market.market_id,
        quantity=10.0,
        avg_entry_price=0.50,
        total_opened_quantity=10.0,
        total_opened_notional=5.0,
        opened_ts=market.ts - timedelta(hours=1),
        entry_probability=0.70,
        thesis="Take partial profits",
    )

    assert engine.should_exit(
        config=config,
        market=market,
        forecast=forecast,
        position=position,
    )

    exit_order = engine.exit_order(
        config=config,
        market=market,
        position=position,
    )

    assert exit_order.requested_quantity == pytest.approx(5.0)
    assert exit_order.limit_price == pytest.approx(market.best_bid)


def test_available_cash_for_entries_respects_remaining_investable_cap() -> None:
    engine = object.__new__(ReplayEngine)
    engine.config = ReplayConfig(experiment_name="test", starting_cash=1_000.0, lookback_minutes=60)
    strategy = StrategyConfig(
        name="edge_cap",
        family="edge_based",
        kelly_fraction=0.1,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=60,
        max_total_invested_pct=0.8,
    )
    portfolio = StrategyPortfolio(cash=250.0)

    available_cash = ReplayEngine._available_cash_for_entries(engine, portfolio, strategy)

    assert available_cash == pytest.approx(50.0)


def test_execute_order_tracks_missed_trade_under_cash_pressure() -> None:
    engine = object.__new__(ReplayEngine)
    engine.config = ReplayConfig(experiment_name="test", starting_cash=1_000.0, lookback_minutes=60)
    strategy = StrategyConfig(
        name="edge_cash_pressure",
        family="edge_based",
        kelly_fraction=0.1,
        edge_threshold_bps=100.0,
        max_position_notional=250.0,
        max_holding_minutes=60,
    )
    engine.strategies = [strategy]
    engine._strategy_by_name = {strategy.name: strategy}
    engine.simulator = SimpleNamespace(
        _market_for_intent=lambda market, order: market,
        _simulate_aggressive=lambda market, order: SimpleNamespace(vwap_price=0.5, quantity=200.0),
        _simulate_passive=lambda market, next_market, order: SimpleNamespace(vwap_price=0.5, quantity=200.0),
    )
    engine._normalize_quotes = lambda market: market
    engine._ensure_orderbook = lambda market, reason: market
    engine._build_degraded_next_market = lambda market, order: market
    engine._strategy_config = ReplayEngine._strategy_config.__get__(engine, ReplayEngine)
    engine._record_missed_trade = ReplayEngine._record_missed_trade.__get__(engine, ReplayEngine)
    portfolio = StrategyPortfolio(cash=260.0)
    market = _make_market_state(best_bid=0.49, best_ask=0.51)
    edge_order = OrderIntent(
        strategy_name=strategy.name,
        market_id=market.market_id,
        ts=market.ts,
        side="buy",
        liquidity_intent="aggressive",
        limit_price=market.best_ask,
        requested_quantity=200.0,
        kelly_fraction=strategy.kelly_fraction,
        edge_bps=120.0,
        holding_period_minutes=strategy.max_holding_minutes,
        thesis="low edge under cash pressure",
    )

    ReplayEngine._execute_order(
        engine,
        portfolio,
        market,
        market,
        edge_order,
        entry_probability=0.6,
    )

    assert portfolio.missed_trades == 1
    assert portfolio.missed_edge_bps == pytest.approx(120.0)


def test_settle_resolved_positions_uses_no_payout_ratio() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE market_resolutions (
            market_id TEXT PRIMARY KEY,
            resolution_ts TEXT NOT NULL,
            resolved_outcome REAL NOT NULL,
            status TEXT NOT NULL,
            disputed INTEGER NOT NULL DEFAULT 0,
            clarification_issued INTEGER NOT NULL DEFAULT 0,
            resolution_note TEXT NOT NULL DEFAULT ''
        )
        """
    )
    resolution_ts = datetime(2026, 1, 16, 12, 0, tzinfo=UTC)
    conn.execute(
        """
        INSERT INTO market_resolutions (
            market_id, resolution_ts, resolved_outcome, status, disputed, clarification_issued, resolution_note
        ) VALUES (?, ?, ?, ?, 0, 0, '')
        """,
        ("test_market", resolution_ts.isoformat(), 0.0, "resolved"),
    )

    engine = object.__new__(ReplayEngine)
    engine.conn = conn
    engine.strategies = [
        StrategyConfig(
            name="resolution_no",
            family="resolution_convergence",
            kelly_fraction=0.1,
            edge_threshold_bps=25.0,
            max_position_notional=250.0,
            max_holding_minutes=None,
        )
    ]
    engine.portfolios = {
        "resolution_no": StrategyPortfolio(
            cash=5.0,
            positions={
                "test_market:NO": PositionState(
                    strategy_name="resolution_no",
                    market_id="test_market",
                    quantity=10.0,
                    avg_entry_price=0.3,
                    total_opened_quantity=10.0,
                    total_opened_notional=3.0,
                    opened_ts=resolution_ts - timedelta(hours=6),
                    is_no_bet=True,
                )
            },
        )
    }
    engine._resolution_cache = {}
    engine._persist_position = lambda position: None

    ReplayEngine._settle_resolved_positions(engine, resolution_ts)

    assert engine.portfolios["resolution_no"].cash == pytest.approx(15.0)
    assert engine.portfolios["resolution_no"].realized_pnl == pytest.approx(7.0)
    assert "test_market:NO" not in engine.portfolios["resolution_no"].positions


def test_mark_portfolio_tracks_yes_and_no_positions_on_same_market() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE pnl_marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            strategy_name TEXT NOT NULL,
            market_id TEXT,
            ts TEXT NOT NULL,
            cash REAL NOT NULL,
            position_qty REAL NOT NULL,
            mark_price REAL NOT NULL,
            inventory_value REAL NOT NULL,
            equity REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            unrealized_pnl REAL NOT NULL
        )
        """
    )
    engine = object.__new__(ReplayEngine)
    engine.conn = conn
    engine.experiment_id = 1
    engine._normalize_quotes = ReplayEngine._normalize_quotes.__get__(engine, ReplayEngine)
    portfolio = StrategyPortfolio(
        cash=100.0,
        positions={
            "test_market": PositionState(
                strategy_name="mark_test",
                market_id="test_market",
                quantity=2.0,
                avg_entry_price=0.50,
            ),
            "test_market:NO": PositionState(
                strategy_name="mark_test",
                market_id="test_market",
                quantity=3.0,
                avg_entry_price=0.40,
                is_no_bet=True,
            ),
        },
    )
    market = _make_market_state(best_bid=0.64, best_ask=0.66)

    ReplayEngine._mark_portfolio_from_market(
        engine,
        portfolio=portfolio,
        strategy_name="mark_test",
        market=market,
    )

    rows = conn.execute(
        """
        SELECT market_id, mark_price, inventory_value, equity
        FROM pnl_marks
        ORDER BY market_id
        """
    ).fetchall()

    assert [str(row["market_id"]) for row in rows] == ["test_market", "test_market:NO"]
    assert rows[0]["mark_price"] == pytest.approx(0.65)
    assert rows[0]["inventory_value"] == pytest.approx(1.3)
    assert rows[1]["mark_price"] == pytest.approx(0.35)
    assert rows[1]["inventory_value"] == pytest.approx(1.05)
    expected_equity = 100.0 + 1.3 + 1.05
    assert rows[0]["equity"] == pytest.approx(expected_equity)
    assert rows[1]["equity"] == pytest.approx(expected_equity)
    assert portfolio.last_known_mids["test_market"] == pytest.approx(0.65)
    assert portfolio.last_known_mids["test_market:NO"] == pytest.approx(0.35)


def test_compute_calibration_curve_skips_unresolved_markets() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE model_outputs (experiment_id INTEGER, probability_yes REAL, market_id TEXT)")
    conn.execute("CREATE TABLE market_resolutions (market_id TEXT, resolved_outcome REAL)")
    conn.executemany(
        "INSERT INTO model_outputs (experiment_id, probability_yes, market_id) VALUES (?, ?, ?)",
        [
            (1, 0.2, "resolved_market"),
            (1, 0.8, "unresolved_market"),
        ],
    )
    conn.execute(
        "INSERT INTO market_resolutions (market_id, resolved_outcome) VALUES (?, ?)",
        ("resolved_market", 1.0),
    )

    curve = compute_calibration_curve(conn, 1, bins=5)

    assert curve == [{"bucket": 1, "forecast_mean": 0.2, "realized_rate": 1.0, "n": 1}]


def test_compute_sharpe_like_uses_active_daily_marks_only() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE pnl_marks (
            experiment_id INTEGER,
            strategy_name TEXT,
            market_id TEXT,
            ts TEXT,
            cash REAL,
            position_qty REAL,
            mark_price REAL,
            inventory_value REAL,
            equity REAL,
            realized_pnl REAL,
            unrealized_pnl REAL
        )
        """
    )

    base_day = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    active_equities = [100.0]
    active_deltas = [8.0, -5.0, 14.0, -5.0, 9.0]
    for index in range(30):
        active_equities.append(active_equities[-1] + active_deltas[index % len(active_deltas)])
    rows: list[tuple[int, str, str, str, float, float, float, float, float, float, float]] = []

    for day_offset, equity in enumerate(active_equities):
        day = base_day + timedelta(days=day_offset)
        rows.extend(
            [
                (
                    1,
                    "active_strategy",
                    "market-a",
                    day.isoformat(),
                    1000.0,
                    1.0,
                    0.5,
                    10.0,
                    equity - 1.0,
                    0.0,
                    0.0,
                ),
                (
                    1,
                    "active_strategy",
                    "market-a",
                    day.replace(hour=15).isoformat(),
                    1000.0,
                    1.0,
                    0.5,
                    10.0,
                    equity - 0.5,
                    0.0,
                    0.0,
                ),
                (
                    1,
                    "active_strategy",
                    "market-b",
                    day.replace(hour=15).isoformat(),
                    1000.0,
                    0.0,
                    0.5,
                    0.0,
                    equity,
                    0.0,
                    0.0,
                ),
                (
                    1,
                    "active_strategy",
                    "market-c",
                    day.replace(hour=23).isoformat(),
                    1000.0,
                    0.0,
                    0.5,
                    0.0,
                    equity + 50.0,
                    0.0,
                    0.0,
                ),
            ]
        )

    for day_offset, equity in enumerate([200.0, 220.0, 210.0], start=len(active_equities)):
        day = base_day + timedelta(days=day_offset)
        rows.append(
            (
                1,
                "active_strategy",
                "market-flat",
                day.isoformat(),
                1000.0,
                0.0,
                0.5,
                0.0,
                equity,
                0.0,
                0.0,
            )
        )

    too_short_equities = [100.0]
    too_short_deltas = [1.0, 2.0, -1.0, 2.0, -1.0]
    for index in range(28):
        too_short_equities.append(too_short_equities[-1] + too_short_deltas[index % len(too_short_deltas)])

    for day_offset, equity in enumerate(too_short_equities):
        day = base_day + timedelta(days=day_offset)
        rows.append(
            (
                1,
                "too_short",
                "market-short",
                day.isoformat(),
                1000.0,
                1.0,
                0.5,
                10.0,
                equity,
                0.0,
                0.0,
            )
        )

    conn.executemany(
        """
        INSERT INTO pnl_marks (
            experiment_id, strategy_name, market_id, ts, cash, position_qty,
            mark_price, inventory_value, equity, realized_pnl, unrealized_pnl
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    sharpe_by_strategy = {item["strategy_name"]: item["sharpe_like"] for item in compute_sharpe_like(conn, 1)}
    daily_returns = [
        (current - previous) / previous for previous, current in zip(active_equities, active_equities[1:], strict=False)
    ]
    expected_sharpe = round((mean(daily_returns) / stdev(daily_returns)) * (365.0**0.5), 6)

    assert sharpe_by_strategy["active_strategy"] == expected_sharpe
    assert sharpe_by_strategy["too_short"] == 0.0


def test_compute_periodic_performance_groups_by_iso_week_and_month() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE pnl_marks (
            experiment_id INTEGER,
            strategy_name TEXT,
            market_id TEXT,
            ts TEXT,
            cash REAL,
            position_qty REAL,
            mark_price REAL,
            inventory_value REAL,
            equity REAL,
            realized_pnl REAL,
            unrealized_pnl REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE fills (
            order_id TEXT,
            experiment_id INTEGER,
            strategy_name TEXT,
            fill_ts TEXT
        )
        """
    )

    rows = [
        (
            1,
            "active_strategy",
            "market-a",
            datetime(2026, 1, 1, 9, 0, tzinfo=UTC).isoformat(),
            1000.0,
            1.0,
            0.5,
            10.0,
            100.0,
            0.0,
            0.0,
        ),
        (
            1,
            "active_strategy",
            "market-a",
            datetime(2026, 1, 3, 12, 0, tzinfo=UTC).isoformat(),
            1000.0,
            1.0,
            0.5,
            10.0,
            109.0,
            0.0,
            0.0,
        ),
        (
            1,
            "active_strategy",
            "market-b",
            datetime(2026, 1, 3, 12, 0, tzinfo=UTC).isoformat(),
            1000.0,
            0.0,
            0.5,
            0.0,
            110.0,
            0.0,
            0.0,
        ),
        (
            1,
            "active_strategy",
            "market-a",
            datetime(2026, 1, 5, 9, 0, tzinfo=UTC).isoformat(),
            1000.0,
            1.0,
            0.5,
            10.0,
            112.0,
            0.0,
            0.0,
        ),
        (
            1,
            "active_strategy",
            "market-c",
            datetime(2026, 1, 8, 18, 0, tzinfo=UTC).isoformat(),
            1000.0,
            0.0,
            0.5,
            0.0,
            125.0,
            0.0,
            0.0,
        ),
    ]
    conn.executemany(
        """
        INSERT INTO pnl_marks (
            experiment_id, strategy_name, market_id, ts, cash, position_qty,
            mark_price, inventory_value, equity, realized_pnl, unrealized_pnl
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.executemany(
        "INSERT INTO fills (order_id, experiment_id, strategy_name, fill_ts) VALUES (?, ?, ?, ?)",
        [
            ("order-1", 1, "active_strategy", datetime(2026, 1, 2, 10, 0, tzinfo=UTC).isoformat()),
            ("order-1", 1, "active_strategy", datetime(2026, 1, 2, 10, 1, tzinfo=UTC).isoformat()),
            ("order-2", 1, "active_strategy", datetime(2026, 1, 4, 14, 0, tzinfo=UTC).isoformat()),
            ("order-3", 1, "active_strategy", datetime(2026, 1, 7, 11, 0, tzinfo=UTC).isoformat()),
        ],
    )

    weekly = compute_periodic_performance(conn, 1, period="week")
    monthly = compute_periodic_performance(conn, 1, period="month")

    assert weekly == [
        {
            "strategy_name": "active_strategy",
            "period": "2026-W01",
            "period_start": "2025-12-29",
            "period_end": "2026-01-04",
            "starting_equity": 100.0,
            "ending_equity": 110.0,
            "pnl": 10.0,
            "n_trades": 2,
        },
        {
            "strategy_name": "active_strategy",
            "period": "2026-W02",
            "period_start": "2026-01-05",
            "period_end": "2026-01-11",
            "starting_equity": 112.0,
            "ending_equity": 125.0,
            "pnl": 13.0,
            "n_trades": 1,
        },
    ]
    assert monthly == [
        {
            "strategy_name": "active_strategy",
            "period": "2026-01",
            "period_start": "2026-01-01",
            "period_end": "2026-01-31",
            "starting_equity": 100.0,
            "ending_equity": 125.0,
            "pnl": 25.0,
            "n_trades": 3,
        }
    ]
