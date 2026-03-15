from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from polymarket_backtest import db
from polymarket_backtest.gym_env import PolymarketGymEnv
from polymarket_backtest.trading_env import Action, MultiMarketEnvironment, TradingEnvironment


def _seed_market(
    conn,
    *,
    market_id: str,
    event_id: str,
    base_ts: datetime,
    mids: list[float],
    volume_1m: float = 20.0,
) -> None:
    resolution_ts = base_ts + timedelta(hours=len(mids) + 2)
    db.add_market(
        conn,
        market_id=market_id,
        title=f"Market {market_id}",
        domain="politics",
        market_type="binary",
        open_ts=base_ts - timedelta(hours=1),
        close_ts=resolution_ts - timedelta(minutes=5),
        resolution_ts=resolution_ts,
        status="active",
        event_id=event_id,
        tags=["Politics"],
    )
    db.add_rule_revision(
        conn,
        market_id=market_id,
        effective_ts=base_ts - timedelta(hours=1),
        rules_text="Fixture rules",
    )
    for index, mid in enumerate(mids):
        ts = base_ts + timedelta(hours=index)
        best_bid = round(max(0.01, mid - 0.02), 4)
        best_ask = round(min(0.99, mid + 0.02), 4)
        db.add_snapshot(
            conn,
            market_id=market_id,
            ts=ts,
            status="active",
            best_bid=best_bid,
            best_ask=best_ask,
            last_trade=mid,
            volume_1m=volume_1m,
            volume_24h=5_000.0 + index * 100.0,
            open_interest=1_000.0,
            tick_size=0.01,
            orderbook=[
                ("bid", 1, best_bid, 100.0),
                ("ask", 1, best_ask, 100.0),
            ],
        )
    db.add_resolution(
        conn,
        market_id=market_id,
        resolution_ts=resolution_ts,
        resolved_outcome=1.0,
        status="resolved",
    )
    conn.commit()


@pytest.fixture
def trading_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "trading_env.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        base_ts = datetime(2025, 11, 1, 12, 0, tzinfo=UTC)
        _seed_market(
            conn,
            market_id="market_a",
            event_id="event_shared",
            base_ts=base_ts,
            mids=[0.45, 0.50, 0.56, 0.60],
        )
        _seed_market(
            conn,
            market_id="market_b",
            event_id="event_shared",
            base_ts=base_ts,
            mids=[0.35, 0.32, 0.30, 0.28],
            volume_1m=8.0,
        )
    return db_path


def test_trading_environment_buy_yes_updates_state(trading_db: Path) -> None:
    env = TradingEnvironment(
        db_path=trading_db,
        starting_cash=100.0,
        market_ids=["market_a"],
        enable_ml_predictions=False,
        auto_order_cash_fraction=0.25,
    )

    state = env.reset("market_a")
    assert state.market_id == "market_a"
    assert state.ml_probability_yes is None
    assert state.to_array().dtype == np.float32

    result = env.step(Action.buy_yes(quantity=10.0))

    assert result.filled_quantity > 0.0
    assert result.fill_price > 0.0
    assert result.new_state.yes_position is not None
    assert result.new_state.no_position is None
    assert result.new_state.cash < 100.0
    assert result.new_state.n_open_positions == 1
    assert len(result.info["fills"]) >= 1


def test_trading_environment_buy_no_tracks_no_inventory(trading_db: Path) -> None:
    env = TradingEnvironment(
        db_path=trading_db,
        starting_cash=100.0,
        market_ids=["market_a"],
        enable_ml_predictions=False,
    )
    env.reset("market_a")

    result = env.step(Action.buy_no(quantity=5.0))

    assert result.filled_quantity > 0.0
    assert result.new_state.no_position is not None
    assert result.new_state.no_position.direction == "no"
    assert result.new_state.no_position.current_price == pytest.approx(1.0 - result.new_state.mid)


def test_trading_environment_mint_and_partial_redeem_pair(trading_db: Path) -> None:
    env = TradingEnvironment(
        db_path=trading_db,
        starting_cash=50.0,
        market_ids=["market_a"],
        enable_ml_predictions=False,
    )
    env.reset("market_a")

    mint_result = env.step(Action.mint_pair(quantity=5.0))
    assert mint_result.filled_quantity == pytest.approx(10.0)
    assert mint_result.new_state.yes_position is not None
    assert mint_result.new_state.no_position is not None

    redeem_result = env.step(Action.redeem_pair(quantity=2.0))
    assert redeem_result.info["redeemed_quantity"] == pytest.approx(2.0)
    assert redeem_result.new_state.yes_position is not None
    assert redeem_result.new_state.no_position is not None
    assert redeem_result.new_state.yes_position.quantity == pytest.approx(3.0)
    assert redeem_result.new_state.no_position.quantity == pytest.approx(3.0)


def test_limit_order_can_be_cancelled(trading_db: Path) -> None:
    env = TradingEnvironment(
        db_path=trading_db,
        starting_cash=100.0,
        market_ids=["market_b"],
        enable_ml_predictions=False,
    )
    env.reset("market_b")

    place_result = env.step(Action.buy_yes_limit(quantity=50.0, price=0.30))
    assert place_result.info["pending_orders_after"] >= 1

    cancel_result = env.step(Action.cancel_orders())
    assert cancel_result.info["cancelled_orders"] >= 1
    assert cancel_result.info["pending_orders_after"] == 0


def test_multi_market_environment_returns_all_market_states(trading_db: Path) -> None:
    env = MultiMarketEnvironment(
        db_path=trading_db,
        starting_cash=100.0,
        market_ids=["market_a", "market_b"],
        enable_ml_predictions=False,
    )

    state = env.reset(["market_a", "market_b"])
    assert set(state.markets) == {"market_a", "market_b"}
    assert state.cash == pytest.approx(100.0)

    results = env.step(
        {
            "market_a": Action.buy_yes(quantity=10.0),
            "market_b": Action.buy_no(quantity=5.0),
        }
    )

    assert set(results) == {"market_a", "market_b"}
    next_state = env.get_state()
    assert next_state.cash < 100.0
    assert len(next_state.positions) >= 1


def test_polymarket_gym_env_reset_and_step(trading_db: Path) -> None:
    env = PolymarketGymEnv(
        db_path=trading_db,
        starting_cash=100.0,
        market_ids=["market_a"],
        enable_ml_predictions=False,
    )

    obs, info = env.reset(options={"market_id": "market_a"})
    assert obs.shape == env.observation_space.shape
    assert info["state"].market_id == "market_a"

    next_obs, reward, terminated, truncated, step_info = env.step(1)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated in {True, False}
    assert truncated is False
    assert "state" in step_info
    env.close()
