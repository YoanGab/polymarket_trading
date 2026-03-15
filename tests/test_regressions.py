from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from polymarket_backtest import db
from polymarket_backtest.downloaders import goldsky
from polymarket_backtest.grid_search import expanded_strategy_grid
from polymarket_backtest.grok_replay import _parse_forecast_output
from polymarket_backtest.market_simulator import MarketSimulator
from polymarket_backtest.metaculus_validator import BASE_URL, _validate_next_url
from polymarket_backtest.ml_transport import MLModelTransport
from polymarket_backtest.replay_engine import ReplayEngine
from polymarket_backtest.snapshot_builder import _interpolate_snapshot
from polymarket_backtest.types import (
    FillResult,
    MarketState,
    OrderIntent,
    OrderLevel,
    PositionState,
    ReplayConfig,
    StrategyConfig,
    isoformat,
)


def _make_market_state(ts: datetime) -> MarketState:
    return MarketState(
        market_id="market-1",
        title="Regression Market",
        domain="tests",
        market_type="binary",
        ts=ts,
        status="active",
        best_bid=0.49,
        best_ask=0.51,
        mid=0.50,
        last_trade=0.50,
        volume_1m=200.0,
        volume_24h=10_000.0,
        open_interest=5_000.0,
        tick_size=0.01,
        rules_text="Regression fixture rules",
        additional_context="",
        resolution_ts=ts + timedelta(hours=24),
        fees_enabled=True,
        fee_rate=0.02,
        fee_exponent=1.0,
        maker_rebate_rate=0.01,
        orderbook=[
            OrderLevel(side="bid", price=0.49, quantity=500.0, level_no=1),
            OrderLevel(side="ask", price=0.51, quantity=500.0, level_no=1),
        ],
    )


def _make_strategy() -> StrategyConfig:
    return StrategyConfig(
        name="edge_test",
        family="edge_based",
        kelly_fraction=0.1,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=60,
        use_time_stop=True,
        min_confidence=0.6,
    )


def _raw_fill(fill_id: str, timestamp: str) -> dict[str, str]:
    return {
        "id": fill_id,
        "maker": "maker",
        "taker": "taker",
        "makerAssetId": "0",
        "takerAssetId": "123",
        "makerAmountFilled": "1000000",
        "takerAmountFilled": "2000000",
        "fee": "0",
        "timestamp": timestamp,
        "transactionHash": f"tx-{fill_id}",
    }


def _build_engine(tmp_path, *, resolved: bool = False) -> tuple[ReplayEngine, StrategyConfig, datetime, str]:
    db_path = tmp_path / "regressions.sqlite"
    conn = db.connect(db_path)
    db.init_db(conn)

    strategy = _make_strategy()
    base_ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    market_id = "engine-market"
    resolution_ts = base_ts + timedelta(hours=24)
    db.add_market(
        conn,
        market_id=market_id,
        title="Engine regression market",
        domain="tests",
        market_type="binary",
        open_ts=base_ts - timedelta(hours=1),
        close_ts=resolution_ts - timedelta(minutes=5),
        resolution_ts=resolution_ts,
        status="active",
    )
    if resolved:
        db.add_resolution(
            conn,
            market_id=market_id,
            resolution_ts=resolution_ts,
            resolved_outcome=1.0,
            status="resolved",
        )
    experiment_id = db.create_experiment(
        conn,
        name="regression-test",
        model_id="grok",
        model_release="grok-4.20-beta-0309-reasoning",
        system_prompt_hash="test-hash",
        config={"starting_cash": 1_000.0},
    )
    engine = ReplayEngine(
        conn=conn,
        config=ReplayConfig(experiment_name="regression-test", starting_cash=1_000.0, lookback_minutes=60),
        grok=SimpleNamespace(experiment_id=experiment_id),
        strategies=[strategy],
    )
    return engine, strategy, base_ts, market_id


def test_metaculus_next_url_must_stay_on_api_origin() -> None:
    _validate_next_url(f"{BASE_URL}/posts/?offset=25")
    with pytest.raises(ValueError, match="untrusted URL"):
        _validate_next_url("https://evil.example/api/posts/?offset=25")


def test_parse_forecast_output_clamps_probability_and_confidence() -> None:
    parsed = _parse_forecast_output(
        {"output_text": ('{"probability_yes": 1.7, "confidence": -0.25, "expected_edge_bps": "12.5", "evidence": []}')},
        model_release="test-release",
    )

    assert parsed["probability_yes"] == 0.999
    assert parsed["confidence"] == 0.0
    assert parsed["expected_edge_bps"] == 12.5
    assert parsed["agent_name"] == "grok_replay"


def test_parse_forecast_output_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _parse_forecast_output({"output_text": '["not-an-object"]'}, model_release="test-release")


def test_ml_transport_uses_bundle_as_of_for_resolution_features() -> None:
    transport = object.__new__(MLModelTransport)
    transport.agent_name = "ml_model"
    transport.model_id = "lightgbm"
    transport._feature_names = ["hours_to_resolution"]
    captured: dict[str, float] = {}

    def _capture_predict(X: np.ndarray) -> np.ndarray:
        captured["hours_to_resolution"] = float(X[0, 0])
        return np.array([0.62], dtype=np.float32)

    transport._predict = _capture_predict

    MLModelTransport.complete(
        transport,
        model_release="test-release",
        system_prompt="",
        context_bundle={
            "as_of": "2026-01-01T12:00:00+00:00",
            "market": {
                "market_id": "market-1",
                "best_bid": 0.49,
                "best_ask": 0.51,
                "mid": 0.50,
                "last_trade": 0.50,
                "volume_1m": 100.0,
                "volume_24h": 1_000.0,
                "open_interest": 500.0,
                "resolution_ts": "2026-01-03T12:00:00+00:00",
            },
        },
    )

    assert captured["hours_to_resolution"] == pytest.approx(48.0)


def test_ml_transport_uses_prev_snapshots_for_history_features() -> None:
    transport = object.__new__(MLModelTransport)
    transport.agent_name = "ml_model"
    transport.model_id = "lightgbm"
    transport._feature_names = [
        "momentum_3h",
        "momentum_24h",
        "volatility_24h",
        "volume_trend",
        "price_range_24h",
    ]
    captured: dict[str, float] = {}

    def _capture_predict(X: np.ndarray) -> np.ndarray:
        captured.update(dict(zip(transport._feature_names, X[0], strict=True)))
        return np.array([0.62], dtype=np.float32)

    transport._predict = _capture_predict

    prev_snapshots = []
    prev_mids = [0.20 + (i * 0.01) for i in range(24)]
    prev_volumes = [1_000.0 + (i * 50.0) for i in range(24)]
    base_ts = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    for index, (mid, volume_24h) in enumerate(zip(prev_mids, prev_volumes, strict=True)):
        prev_snapshots.append(
            {
                "market_id": "market-1",
                "ts": isoformat(base_ts + timedelta(hours=index)),
                "best_bid": mid - 0.01,
                "best_ask": mid + 0.01,
                "mid": mid,
                "last_trade": mid,
                "volume_1m": 100.0 + index,
                "volume_24h": volume_24h,
                "open_interest": 500.0,
                "resolution_ts": isoformat(base_ts + timedelta(days=3)),
            }
        )

    MLModelTransport.complete(
        transport,
        model_release="test-release",
        system_prompt="",
        context_bundle={
            "as_of": isoformat(base_ts + timedelta(hours=24)),
            "market": {
                "market_id": "market-1",
                "best_bid": 0.44,
                "best_ask": 0.46,
                "mid": 0.45,
                "last_trade": 0.45,
                "volume_1m": 200.0,
                "volume_24h": 2_250.0,
                "open_interest": 500.0,
                "resolution_ts": isoformat(base_ts + timedelta(days=3)),
            },
            "prev_snapshots": prev_snapshots,
        },
    )

    assert captured["momentum_3h"] == pytest.approx(0.04)
    assert captured["momentum_24h"] == pytest.approx(0.25)
    assert captured["volatility_24h"] == pytest.approx(float(np.std(prev_mids)))
    assert captured["volume_trend"] == pytest.approx((2100.0 - 1950.0) / 1950.0)
    assert captured["price_range_24h"] == pytest.approx(0.23)


def test_ml_transport_confidence_uses_tradable_edge() -> None:
    transport = object.__new__(MLModelTransport)
    transport.agent_name = "ml_model"
    transport.model_id = "lightgbm"
    transport._feature_names = []
    transport._predict = lambda X: np.array([0.55], dtype=np.float32)

    no_edge = MLModelTransport.complete(
        transport,
        model_release="test-release",
        system_prompt="",
        context_bundle={
            "as_of": "2026-01-01T12:00:00+00:00",
            "market": {
                "market_id": "market-1",
                "best_bid": 0.54,
                "best_ask": 0.56,
                "mid": 0.55,
                "resolution_ts": None,
            },
        },
    )
    positive_edge = MLModelTransport.complete(
        transport,
        model_release="test-release",
        system_prompt="",
        context_bundle={
            "as_of": "2026-01-01T12:00:00+00:00",
            "market": {
                "market_id": "market-1",
                "best_bid": 0.53,
                "best_ask": 0.54,
                "mid": 0.535,
                "resolution_ts": None,
            },
        },
    )

    assert no_edge["confidence"] == pytest.approx(0.5)
    assert positive_edge["confidence"] == pytest.approx(0.6)


def test_process_market_snapshot_passes_prev_snapshots_to_forecast() -> None:
    engine = object.__new__(ReplayEngine)
    engine.strategies = []
    engine.portfolios = {}

    market = _make_market_state(datetime(2026, 1, 15, 12, 0, tzinfo=UTC))
    history_entries: list[tuple[str, dict[str, float | str | None]]] = []
    for hours_ago in range(30, 0, -1):
        ts = market.ts - timedelta(hours=hours_ago)
        mid = 0.20 + ((30 - hours_ago) * 0.01)
        history_entries.append(
            (
                isoformat(ts),
                {
                    "market_id": market.market_id,
                    "ts": isoformat(ts),
                    "status": "active",
                    "best_bid": mid - 0.01,
                    "best_ask": mid + 0.01,
                    "mid": mid,
                    "last_trade": mid,
                    "volume_1m": 100.0,
                    "volume_24h": 1_000.0 + hours_ago,
                    "open_interest": 500.0,
                    "tick_size": 0.01,
                    "resolution_ts": isoformat(market.resolution_ts) if market.resolution_ts is not None else None,
                },
            )
        )
    history_entries.append(
        (
            isoformat(market.ts),
            {
                "market_id": market.market_id,
                "ts": isoformat(market.ts),
                "status": market.status,
                "best_bid": market.best_bid,
                "best_ask": market.best_ask,
                "mid": market.mid,
                "last_trade": market.last_trade,
                "volume_1m": market.volume_1m,
                "volume_24h": market.volume_24h,
                "open_interest": market.open_interest,
                "tick_size": market.tick_size,
                "resolution_ts": isoformat(market.resolution_ts) if market.resolution_ts is not None else None,
            },
        )
    )
    engine._history_by_market = {market.market_id: history_entries}
    engine._history_index_by_key = {
        (market.market_id, ts_str): index for index, (ts_str, _) in enumerate(history_entries)
    }
    engine._get_market_history = ReplayEngine._get_market_history.__get__(engine, ReplayEngine)
    engine._is_market_resolved_as_of = lambda market_id, timestamp: False
    engine._get_cached_market = lambda market_id, timestamp: market
    engine._get_cached_next_market = lambda market_id, timestamp: None
    engine._persist_model_output = lambda forecast, prompt_hash, context_hash: None

    captured: dict[str, Any] = {}

    def _forecast(
        market_id: str,
        timestamp: datetime,
        *,
        market_state: MarketState | None = None,
        prev_snapshots: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, str, str]:
        captured["market_id"] = market_id
        captured["timestamp"] = timestamp
        captured["market_state"] = market_state
        captured["prev_snapshots"] = list(prev_snapshots or [])
        return _make_forecast(), "prompt-hash", "context-hash"

    engine.grok = SimpleNamespace(forecast=_forecast)

    ReplayEngine._process_market_snapshot(engine, market.market_id, market.ts)

    assert captured["market_id"] == market.market_id
    assert captured["timestamp"] == market.ts
    assert captured["market_state"] == market
    assert len(captured["prev_snapshots"]) == 24
    assert captured["prev_snapshots"][0]["ts"] == isoformat(market.ts - timedelta(hours=24))
    assert captured["prev_snapshots"][-1]["ts"] == isoformat(market.ts - timedelta(hours=1))


def test_passive_fill_uses_delayed_fill_timestamp() -> None:
    base_ts = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    market = _make_market_state(base_ts)
    next_market = _make_market_state(base_ts + timedelta(minutes=5))
    intent = OrderIntent(
        strategy_name="edge_test",
        market_id=market.market_id,
        ts=market.ts,
        side="buy",
        liquidity_intent="passive",
        limit_price=market.best_bid,
        requested_quantity=10.0,
        kelly_fraction=0.1,
        edge_bps=50.0,
        holding_period_minutes=60,
        thesis="Regression test",
    )

    fills = MarketSimulator().simulate(order_id="order-1", market=market, next_market=next_market, intent=intent)

    assert fills
    assert fills[0].fill_delay_seconds > 0
    assert fills[0].fill_ts > market.ts


def test_goldsky_pagination_deduplicates_boundary_timestamp_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        [_raw_fill("fill-1", "100"), _raw_fill("fill-2", "100")],
        [_raw_fill("fill-2", "100"), _raw_fill("fill-3", "100")],
        [],
    ]

    class DummyClient:
        def __enter__(self) -> DummyClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(goldsky, "_build_client", lambda: DummyClient())
    monkeypatch.setattr(
        goldsky,
        "_request_order_fills_page",
        lambda client, *, token_id, after_timestamp, limit: pages.pop(0),
    )

    fills = goldsky.fetch_order_fills(token_id="123", after_timestamp="0", limit=2, max_fills=10)

    assert [fill["id"] for fill in fills] == ["fill-1", "fill-2", "fill-3"]


def test_expanded_strategy_grid_uses_honest_edge_names() -> None:
    strategies = expanded_strategy_grid()
    names = {s.name for s in strategies}
    families = {s.family for s in strategies}

    # No fantasy strategy names
    assert "deep_research_long" not in names
    assert "cross_market_arb" not in names
    # All current strategies should be valid families
    assert families <= {
        "resolution_convergence",
        "edge_based",
        "contrarian",
        "momentum",
        "mean_reversion",
        "carry_only",
        "news_driven",
        "volume_breakout",
    }


def test_expanded_strategy_grid_routes_by_category() -> None:
    strategies = {strategy.name: strategy for strategy in expanded_strategy_grid()}

    assert "political_event_wide" in strategies
    assert strategies["political_event_wide"].blocked_categories is not None
    assert "Crypto" in (strategies["political_event_wide"].blocked_categories or [])
    assert "Sports" in (strategies["political_event_wide"].blocked_categories or [])

    assert strategies["crypto_resolution_day"].allowed_categories == ["Crypto", "Crypto Prices"]
    assert strategies["crypto_edge_fast"].allowed_categories == ["Crypto", "Crypto Prices"]

    sports_categories = strategies["sports_resolution_day"].allowed_categories or []
    assert "Sports" in sports_categories
    assert "NBA" in sports_categories


def test_interpolate_snapshot_keeps_start_tick_size() -> None:
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    interpolated = _interpolate_snapshot(
        {
            "market_id": "market-1",
            "ts": ts,
            "status": "active",
            "best_bid": 0.4,
            "best_ask": 0.6,
            "last_trade": 0.5,
            "volume_1m": 100.0,
            "volume_24h": 1_000.0,
            "open_interest": 500.0,
            "tick_size": 0.001,
        },
        {
            "market_id": "market-1",
            "ts": ts + timedelta(minutes=30),
            "status": "active",
            "best_bid": 0.45,
            "best_ask": 0.65,
            "last_trade": 0.55,
            "volume_1m": 200.0,
            "volume_24h": 2_000.0,
            "open_interest": 600.0,
            "tick_size": 0.01,
        },
        ts + timedelta(minutes=15),
        0.5,
    )

    assert interpolated["tick_size"] == 0.001


def test_closed_sell_position_is_persisted_and_evicted(tmp_path) -> None:
    engine, strategy, base_ts, market_id = _build_engine(tmp_path)
    portfolio = engine.portfolios[strategy.name]

    engine._apply_fill(
        portfolio,
        FillResult(
            order_id="order-buy",
            market_id=market_id,
            strategy_name=strategy.name,
            fill_ts=base_ts,
            side="buy",
            liquidity_role="taker",
            price=0.40,
            quantity=10.0,
            fee_usdc=0.2,
            rebate_usdc=0.0,
            impact_bps=0.0,
            fill_delay_seconds=0.0,
        ),
        thesis="Open position",
        entry_probability=0.65,
    )
    engine._apply_fill(
        portfolio,
        FillResult(
            order_id="order-sell",
            market_id=market_id,
            strategy_name=strategy.name,
            fill_ts=base_ts + timedelta(hours=1),
            side="sell",
            liquidity_role="taker",
            price=0.60,
            quantity=10.0,
            fee_usdc=0.2,
            rebate_usdc=0.0,
            impact_bps=0.0,
            fill_delay_seconds=0.0,
        ),
        thesis="Close position",
        entry_probability=0.65,
    )

    row = engine.conn.execute(
        "SELECT status, quantity, closed_ts FROM positions WHERE experiment_id = ?",
        (engine.experiment_id,),
    ).fetchone()

    assert market_id not in portfolio.positions
    assert row is not None
    assert row["status"] == "closed"
    assert float(row["quantity"]) == 0.0
    assert str(row["closed_ts"]) == (base_ts + timedelta(hours=1)).isoformat(timespec="seconds")

    engine.conn.close()


def test_resolved_position_is_persisted_and_evicted(tmp_path) -> None:
    engine, strategy, base_ts, market_id = _build_engine(tmp_path, resolved=True)
    portfolio = engine.portfolios[strategy.name]
    portfolio.positions[market_id] = PositionState(
        strategy_name=strategy.name,
        market_id=market_id,
        quantity=5.0,
        avg_entry_price=0.40,
        total_opened_quantity=5.0,
        total_opened_notional=2.0,
        opened_ts=base_ts,
        entry_probability=0.65,
        thesis="Hold to resolution",
    )

    engine._settle_resolved_positions(base_ts + timedelta(hours=25))

    row = engine.conn.execute(
        "SELECT status, quantity, resolved_outcome FROM positions WHERE experiment_id = ?",
        (engine.experiment_id,),
    ).fetchone()

    assert market_id not in portfolio.positions
    assert row is not None
    assert row["status"] == "closed"
    assert float(row["quantity"]) == 0.0
    assert float(row["resolved_outcome"]) == 1.0

    engine.conn.close()
