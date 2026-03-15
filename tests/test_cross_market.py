from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from polymarket_backtest import db
from polymarket_backtest.cross_market import CrossMarketTracker
from polymarket_backtest.replay_engine import ReplayEngine, StrategyPortfolio
from polymarket_backtest.strategies import StrategyEngine
from polymarket_backtest.types import (
    ForecastOutput,
    OrderIntent,
    OrderLevel,
    PositionState,
    ReplayConfig,
    StrategyConfig,
)


def _forecast(market_id: str, as_of: datetime) -> ForecastOutput:
    return ForecastOutput(
        agent_name="stub",
        model_id="stub",
        model_release="test",
        as_of=as_of,
        market_id=market_id,
        domain="test",
        probability_yes=0.55,
        confidence=0.75,
        expected_edge_bps=0.0,
        thesis="stub forecast",
        reasoning="stub forecast",
        evidence=[],
        raw_response={},
    )


def _strategy(name: str = "test_strategy", family: str = "edge_based") -> StrategyConfig:
    return StrategyConfig(
        name=name,
        family=family,  # type: ignore[arg-type]
        kelly_fraction=0.1,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=60,
        min_confidence=0.6,
    )


def _add_market_series(
    conn,
    *,
    market_id: str,
    event_id: str | None,
    mids: list[float],
    base_ts: datetime,
) -> datetime:
    resolution_ts = base_ts + timedelta(days=1)
    db.add_market(
        conn,
        market_id=market_id,
        title=f"Market {market_id}",
        domain="tests",
        market_type="binary",
        event_id=event_id,
        tags=["Politics", "Election"],
        open_ts=base_ts - timedelta(hours=1),
        close_ts=resolution_ts - timedelta(minutes=5),
        resolution_ts=resolution_ts,
        status="active",
        maker_rebate_rate=0.01,
    )
    db.add_rule_revision(
        conn,
        market_id=market_id,
        effective_ts=base_ts - timedelta(hours=1),
        rules_text="Fixture rules",
    )
    final_ts = base_ts
    for index, mid in enumerate(mids):
        ts = base_ts + timedelta(minutes=5 * index)
        final_ts = ts
        best_bid = round(max(0.001, mid - 0.01), 4)
        best_ask = round(min(0.999, mid + 0.01), 4)
        db.add_snapshot(
            conn,
            market_id=market_id,
            ts=ts,
            status="active",
            best_bid=best_bid,
            best_ask=best_ask,
            last_trade=mid,
            volume_1m=100.0 + index,
            volume_24h=5_000.0 + (index * 50.0),
            open_interest=1_000.0,
            tick_size=0.01,
            orderbook=[
                ("bid", 1, best_bid, 100.0),
                ("ask", 1, best_ask, 100.0),
            ],
        )
    return final_ts


def _make_engine(conn, *, strategy: StrategyConfig, experiment_name: str = "cross-market") -> ReplayEngine:
    experiment_id = db.create_experiment(
        conn,
        name=experiment_name,
        model_id="stub",
        model_release="test",
        system_prompt_hash="test-hash",
        config={"starting_cash": 1_000.0},
    )
    return ReplayEngine(
        conn=conn,
        config=ReplayConfig(experiment_name=experiment_name, starting_cash=1_000.0, lookback_minutes=60),
        grok=SimpleNamespace(experiment_id=experiment_id),
        strategies=[strategy],
    )


def test_cross_market_tracker_returns_event_correlations(tmp_path: Path) -> None:
    db_path = tmp_path / "cross_market.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        base_ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        _add_market_series(conn, market_id="market_a", event_id="event_shared", mids=[0.40, 0.45, 0.50], base_ts=base_ts)
        _add_market_series(conn, market_id="market_b", event_id="event_shared", mids=[0.20, 0.25, 0.30], base_ts=base_ts)
        _add_market_series(conn, market_id="market_c", event_id="event_other", mids=[0.60, 0.58, 0.56], base_ts=base_ts)

        tracker = CrossMarketTracker(conn)
        related = tracker.get_related_markets("market_a")

        assert [item.market_id for item in related] == ["market_b"]
        assert related[0].correlation == pytest.approx(1.0)


def test_replay_engine_passes_related_market_prices_to_strategies(tmp_path: Path) -> None:
    db_path = tmp_path / "replay_related.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        base_ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        final_ts = _add_market_series(
            conn,
            market_id="market_a",
            event_id="event_shared",
            mids=[0.40, 0.45, 0.50],
            base_ts=base_ts,
        )
        _add_market_series(
            conn,
            market_id="market_b",
            event_id="event_shared",
            mids=[0.42, 0.47, 0.52],
            base_ts=base_ts,
        )
        strategy = _strategy()
        engine = _make_engine(conn, strategy=strategy, experiment_name="replay-related")
        engine._preload_market_data({"market_a": [final_ts], "market_b": [final_ts]})

        captured: dict[str, object] = {}

        def _forecast_stub(market_id, as_of, *, market_state=None, prev_snapshots=None):
            return _forecast(market_id, as_of), "prompt-hash", "context-hash"

        def _capture_decide(**kwargs):
            captured["related_market_prices"] = kwargs["related_market_prices"]
            return []

        engine.grok.forecast = _forecast_stub
        engine.strategy_engine = SimpleNamespace(
            should_exit=lambda **_: False,
            decide=_capture_decide,
        )

        engine._process_market_snapshot("market_a", final_ts)

        related_market_prices = captured["related_market_prices"]
        assert isinstance(related_market_prices, dict)
        assert "market_b" in related_market_prices
        market_b = related_market_prices["market_b"]
        assert market_b["mid"] == pytest.approx(0.52)
        assert market_b["correlation"] == pytest.approx(1.0)
        assert market_b["event_id"] == "event_shared"


def test_market_making_strategy_quotes_both_sides_and_skews_inventory() -> None:
    strategy = StrategyConfig(
        name="mm_test",
        family="market_making",
        kelly_fraction=0.1,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=60,
        mm_spread_bps=200.0,
        mm_max_inventory=100.0,
    )
    market_ts = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    market = SimpleNamespace(
        market_id="market_mm",
        title="MM market",
        domain="tests",
        market_type="binary",
        ts=market_ts,
        status="active",
        best_bid=0.49,
        best_ask=0.51,
        mid=0.50,
        last_trade=0.50,
        volume_1m=100.0,
        volume_24h=5_000.0,
        open_interest=1_000.0,
        tick_size=0.01,
        rules_text="rules",
        additional_context="",
        resolution_ts=market_ts + timedelta(days=1),
        fees_enabled=True,
        fee_rate=0.02,
        fee_exponent=1.0,
        maker_rebate_rate=0.01,
        orderbook=[
            OrderLevel(side="bid", price=0.49, quantity=100.0, level_no=1),
            OrderLevel(side="ask", price=0.51, quantity=100.0, level_no=1),
        ],
        tags=[],
    )
    strategy_engine = StrategyEngine()
    forecast = _forecast("market_mm", market_ts)

    flat_orders = strategy_engine.decide(
        config=strategy,
        market=market,
        forecast=forecast,
        position=None,
        no_position=None,
        available_cash=1_000.0,
    )
    heavy_yes_orders = strategy_engine.decide(
        config=strategy,
        market=market,
        forecast=forecast,
        position=PositionState(strategy_name="mm_test", market_id="market_mm", quantity=90.0, avg_entry_price=0.50),
        no_position=None,
        available_cash=1_000.0,
    )

    assert len(flat_orders) == 2
    assert {order.side for order in flat_orders} == {"buy", "sell"}
    assert all(order.liquidity_intent == "passive" for order in flat_orders)

    flat_buy = next(order for order in flat_orders if order.side == "buy")
    flat_sell = next(order for order in flat_orders if order.side == "sell")
    heavy_buy = next(order for order in heavy_yes_orders if order.side == "buy")
    heavy_sell = next(order for order in heavy_yes_orders if order.side == "sell")

    assert flat_buy.limit_price < market.mid < flat_sell.limit_price
    assert heavy_buy.limit_price < flat_buy.limit_price
    assert heavy_sell.limit_price > flat_sell.limit_price


def test_execute_order_skips_sell_when_inventory_is_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "empty_sell.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        base_ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        _add_market_series(conn, market_id="market_a", event_id="event_shared", mids=[0.50], base_ts=base_ts)
        strategy = _strategy()
        engine = _make_engine(conn, strategy=strategy, experiment_name="empty-sell")
        portfolio = StrategyPortfolio(cash=1_000.0)
        market = db.get_market_state_as_of(conn, "market_a", base_ts)
        assert market is not None

        order = OrderIntent(
            strategy_name=strategy.name,
            market_id="market_a",
            ts=base_ts,
            side="sell",
            liquidity_intent="passive",
            limit_price=0.52,
            requested_quantity=10.0,
            kelly_fraction=strategy.kelly_fraction,
            edge_bps=10.0,
            holding_period_minutes=60,
            thesis="sell without inventory",
        )

        engine._execute_order(
            portfolio=portfolio,
            market=market,
            next_market=market,
            order=order,
            entry_probability=0.55,
        )

        order_count = conn.execute("SELECT COUNT(*) FROM orders WHERE experiment_id = ?", (engine.experiment_id,)).fetchone()[0]
        assert order_count == 0
        assert portfolio.cash == pytest.approx(1_000.0)
