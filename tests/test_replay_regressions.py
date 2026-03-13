from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from polymarket_backtest.metrics import compute_calibration_curve
from polymarket_backtest.replay_engine import ReplayEngine, StrategyPortfolio
from polymarket_backtest.strategies import StrategyEngine
from polymarket_backtest.types import (
    FillResult,
    ForecastOutput,
    MarketState,
    OrderLevel,
    PositionState,
    StrategyConfig,
)


def _make_market_state(*, best_bid: float, best_ask: float) -> MarketState:
    ts = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    return MarketState(
        market_id="test_market",
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


def test_normalize_quotes_clamps_zero_quotes_and_preserves_spread() -> None:
    engine = object.__new__(ReplayEngine)
    market = _make_market_state(best_bid=0.0, best_ask=0.0)

    normalized = ReplayEngine._normalize_quotes(engine, market)

    assert normalized.best_bid == pytest.approx(0.001)
    assert normalized.best_ask > normalized.best_bid
    assert normalized.mid == pytest.approx((normalized.best_bid + normalized.best_ask) / 2.0)


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
