from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from statistics import mean, stdev

import pytest

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
                "test_market": PositionState(
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
    assert "test_market" not in engine.portfolios["resolution_no"].positions


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
        (current - previous) / previous
        for previous, current in zip(active_equities, active_equities[1:], strict=False)
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
        (1, "active_strategy", "market-a", datetime(2026, 1, 1, 9, 0, tzinfo=UTC).isoformat(), 1000.0, 1.0, 0.5, 10.0, 100.0, 0.0, 0.0),
        (1, "active_strategy", "market-a", datetime(2026, 1, 3, 12, 0, tzinfo=UTC).isoformat(), 1000.0, 1.0, 0.5, 10.0, 109.0, 0.0, 0.0),
        (1, "active_strategy", "market-b", datetime(2026, 1, 3, 12, 0, tzinfo=UTC).isoformat(), 1000.0, 0.0, 0.5, 0.0, 110.0, 0.0, 0.0),
        (1, "active_strategy", "market-a", datetime(2026, 1, 5, 9, 0, tzinfo=UTC).isoformat(), 1000.0, 1.0, 0.5, 10.0, 112.0, 0.0, 0.0),
        (1, "active_strategy", "market-c", datetime(2026, 1, 8, 18, 0, tzinfo=UTC).isoformat(), 1000.0, 0.0, 0.5, 0.0, 125.0, 0.0, 0.0),
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
