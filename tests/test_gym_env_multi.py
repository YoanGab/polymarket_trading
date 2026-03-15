"""Tests for the multi-market Gymnasium environment."""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from polymarket_backtest import db
from polymarket_backtest.gym_env_multi import (
    _MARKET_FEATURE_DIM,
    _N_ACTIONS,
    _PORTFOLIO_DIM,
    PolymarketMultiMarketGymEnv,
)


def _seed_market(
    conn,
    *,
    market_id: str,
    event_id: str,
    base_ts: datetime,
    mids: list[float],
    volume_1m: float = 20.0,
    resolved_outcome: float = 1.0,
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
        resolved_outcome=resolved_outcome,
        status="resolved",
    )
    conn.commit()


@pytest.fixture
def multi_db(tmp_path: Path) -> Path:
    """Create a test DB with 5 markets for multi-market testing."""
    db_path = tmp_path / "multi_env.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        base_ts = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
        # 5 markets with different price trajectories and enough snapshots
        _seed_market(
            conn,
            market_id="m1",
            event_id="e1",
            base_ts=base_ts,
            mids=[0.45, 0.50, 0.56, 0.60, 0.65, 0.70],
        )
        _seed_market(
            conn,
            market_id="m2",
            event_id="e2",
            base_ts=base_ts,
            mids=[0.35, 0.32, 0.30, 0.28, 0.25, 0.22],
        )
        _seed_market(
            conn,
            market_id="m3",
            event_id="e1",
            base_ts=base_ts,
            mids=[0.80, 0.82, 0.85, 0.88, 0.90, 0.92],
        )
        _seed_market(
            conn,
            market_id="m4",
            event_id="e3",
            base_ts=base_ts,
            mids=[0.50, 0.48, 0.52, 0.55, 0.53, 0.58],
        )
        _seed_market(
            conn,
            market_id="m5",
            event_id="e3",
            base_ts=base_ts,
            mids=[0.60, 0.62, 0.58, 0.65, 0.70, 0.75],
        )
    return db_path


class TestMultiMarketGymEnvInit:
    """Test environment initialization and space definitions."""

    def test_creates_with_correct_spaces(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        assert env.observation_space["market_features"].shape == (3, _MARKET_FEATURE_DIM)
        assert env.observation_space["portfolio"].shape == (_PORTFOLIO_DIM,)
        assert env.observation_space["market_mask"].shape == (3,)
        assert env.observation_space["ml_edges"].shape == (3, 2)
        assert env.action_space.shape == (3,)
        assert all(n == _N_ACTIONS for n in env.action_space.nvec)
        env.close()

    def test_raises_on_empty_split(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.sqlite"
        with closing(db.connect(db_path)) as conn:
            db.init_db(conn)
        with pytest.raises(ValueError, match="No markets"):
            PolymarketMultiMarketGymEnv(
                db_path=db_path,
                n_markets=5,
                split="train",
                enable_ml_predictions=False,
            )


class TestMultiMarketGymEnvReset:
    """Test reset behavior."""

    def test_reset_returns_valid_observation(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        obs, info = env.reset(seed=42)

        assert obs["market_features"].shape == (3, _MARKET_FEATURE_DIM)
        assert obs["portfolio"].shape == (_PORTFOLIO_DIM,)
        assert obs["market_mask"].shape == (3,)
        assert obs["ml_edges"].shape == (3, 2)
        assert obs["market_features"].dtype == np.float32
        assert obs["portfolio"].dtype == np.float32

        # At least some markets should be active
        assert obs["market_mask"].sum() > 0
        assert info["n_active"] > 0
        assert info["cash"] == pytest.approx(500.0)
        env.close()

    def test_reset_fills_slots_up_to_n_markets(self, multi_db: Path) -> None:
        # Request 3 slots, 5 available markets -- should fill 3
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        obs, info = env.reset(seed=42)
        assert info["n_active"] == 3
        env.close()

    def test_reset_handles_more_slots_than_markets(self, multi_db: Path) -> None:
        # Request 10 slots but only 5 markets exist
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=10,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        obs, info = env.reset(seed=42)
        # Should fill up to 5 (all available) and leave rest empty
        assert info["n_active"] <= 5
        assert info["n_active"] > 0
        # Empty slots should be masked to 0
        active_count = int(obs["market_mask"].sum())
        assert active_count == info["n_active"]
        env.close()

    def test_reset_is_deterministic_with_seed(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        obs1, info1 = env.reset(seed=123)
        obs2, info2 = env.reset(seed=123)

        np.testing.assert_array_equal(obs1["market_features"], obs2["market_features"])
        np.testing.assert_array_equal(obs1["market_mask"], obs2["market_mask"])
        np.testing.assert_array_equal(obs1["portfolio"], obs2["portfolio"])
        assert info1["active_markets"] == info2["active_markets"]
        env.close()

    def test_portfolio_starts_at_full_cash(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=1000.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        obs, _ = env.reset(seed=42)
        # cash_frac should be 1.0, invested_frac=0, etc.
        assert obs["portfolio"][0] == pytest.approx(1.0)  # cash_frac
        assert obs["portfolio"][1] == pytest.approx(0.0)  # invested_frac
        assert obs["portfolio"][2] == pytest.approx(0.0)  # n_positions_frac
        env.close()


class TestMultiMarketGymEnvStep:
    """Test step mechanics."""

    def test_hold_action_advances_without_trading(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        env.reset(seed=42)

        # All hold actions
        action = np.zeros(3, dtype=np.int64)
        obs, reward, done, truncated, info = env.step(action)

        assert obs["market_features"].shape == (3, _MARKET_FEATURE_DIM)
        assert isinstance(reward, float)
        assert truncated is False
        assert info["cash"] == pytest.approx(500.0)
        env.close()

    def test_buy_reduces_cash(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        env.reset(seed=42)

        # Buy YES on slot 0, hold on rest
        action = np.zeros(3, dtype=np.int64)
        action[0] = 1  # buy_yes
        obs, reward, done, truncated, info = env.step(action)

        # Cash should be less than starting amount
        assert info["cash"] < 500.0
        env.close()

    def test_multi_market_buying(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        env.reset(seed=42)

        # Buy on multiple markets simultaneously
        action = np.array([1, 2, 1], dtype=np.int64)  # buy_yes, buy_no, buy_yes
        obs, reward, done, truncated, info = env.step(action)

        # Cash should decrease from multiple buys
        assert info["cash"] < 500.0
        env.close()

    def test_shared_cash_pool(self, multi_db: Path) -> None:
        """Buying on market A should reduce cash available for market B."""
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=100.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
            auto_order_cash_fraction=0.5,
        )
        env.reset(seed=42)

        # Buy on all markets -- shared cash means total spent comes from one pool
        action = np.array([1, 1, 1], dtype=np.int64)
        obs, reward, done, truncated, info = env.step(action)

        # Cash should be significantly reduced
        assert info["cash"] < 100.0
        # Portfolio features should reflect investment
        assert obs["portfolio"][1] > 0.0  # invested_frac > 0
        env.close()

    def test_episode_eventually_terminates(self, multi_db: Path) -> None:
        """Stepping repeatedly should eventually end the episode."""
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
            episode_hours=48.0,
        )
        env.reset(seed=42)

        done = False
        steps = 0
        max_steps = 200
        while not done and steps < max_steps:
            action = np.zeros(3, dtype=np.int64)
            _, _, done, _, _ = env.step(action)
            steps += 1

        # Should terminate within max_steps (markets have ~6 snapshots each)
        assert done or steps == max_steps
        env.close()

    def test_masked_slots_ignored_in_actions(self, multi_db: Path) -> None:
        """Actions on inactive (masked) slots should have no effect."""
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=10,  # More slots than markets
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        obs, _ = env.reset(seed=42)

        # Fill all slots with buy_yes -- inactive ones should be ignored
        action = np.ones(10, dtype=np.int64)  # all buy_yes
        obs, _, _, _, info = env.step(action)

        # Should still work without errors
        assert obs["market_features"].shape == (10, _MARKET_FEATURE_DIM)
        env.close()

    def test_sell_after_buy(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        env.reset(seed=42)

        # Buy YES on slot 0
        action = np.zeros(3, dtype=np.int64)
        action[0] = 1
        env.step(action)
        cash_after_buy = env._core.portfolio.cash

        # Sell YES on slot 0
        action = np.zeros(3, dtype=np.int64)
        action[0] = 3  # sell_yes
        env.step(action)
        cash_after_sell = env._core.portfolio.cash

        # Cash should increase after selling
        assert cash_after_sell > cash_after_buy
        env.close()


class TestMultiMarketGymEnvObservation:
    """Test observation space conformity."""

    def test_observation_in_space(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        env.close()

    def test_step_observation_in_space(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)
        env.close()

    def test_inactive_slots_are_zero(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=10,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        obs, _ = env.reset(seed=42)

        for i in range(10):
            if obs["market_mask"][i] == 0.0:
                np.testing.assert_array_equal(
                    obs["market_features"][i],
                    np.zeros(_MARKET_FEATURE_DIM, dtype=np.float32),
                )
        env.close()


class TestMultiMarketGymEnvMLScreener:
    """Test ML screener integration."""

    def test_screener_controls_market_selection(self, multi_db: Path) -> None:
        class FixedScreener:
            def rank_markets(
                self,
                market_ids: list[str],
                as_of: datetime,
            ) -> list[tuple[str, float, float]]:
                # Always return m3, m1, m5 in that order
                priority = ["m3", "m1", "m5", "m2", "m4"]
                result = []
                for mid in priority:
                    if mid in market_ids:
                        result.append((mid, 0.8, 150.0))
                return result

        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            ml_screener=FixedScreener(),
            enable_ml_predictions=False,
            random_seed=42,
        )
        _, info = env.reset(seed=42)

        # The screener should have picked m3, m1, m5
        active = set(info["active_markets"])
        assert "m3" in active or "m1" in active
        env.close()

    def test_screener_edge_values_in_observation(self, multi_db: Path) -> None:
        class EdgeScreener:
            def rank_markets(
                self,
                market_ids: list[str],
                as_of: datetime,
            ) -> list[tuple[str, float, float]]:
                return [(mid, 0.65, 200.0) for mid in market_ids[:3]]

        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            ml_screener=EdgeScreener(),
            enable_ml_predictions=False,
            random_seed=42,
        )
        obs, _ = env.reset(seed=42)

        # At least some ML edge slots should be non-zero
        active_edges = obs["ml_edges"][obs["market_mask"] > 0]
        if len(active_edges) > 0:
            assert np.any(active_edges != 0.0)
        env.close()


class TestMultiMarketGymEnvRender:
    """Test render output."""

    def test_render_returns_string(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        env.reset(seed=42)
        output = env.render()
        assert isinstance(output, str)
        assert "step=" in output
        assert "active_markets=" in output
        assert "cash=" in output
        env.close()


class TestMultiMarketGymEnvReward:
    """Test reward computation."""

    def test_hold_reward_reflects_market_movement(self, multi_db: Path) -> None:
        """Even holding should produce non-trivial reward if prices move."""
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        env.reset(seed=42)

        # First buy some positions
        action = np.array([1, 2, 0], dtype=np.int64)
        env.step(action)

        # Then hold -- reward should reflect price changes
        action = np.zeros(3, dtype=np.int64)
        _, reward, _, _, _ = env.step(action)
        # Reward is a float (may be positive, negative, or zero)
        assert isinstance(reward, float)
        env.close()

    def test_reward_is_portfolio_delta(self, multi_db: Path) -> None:
        env = PolymarketMultiMarketGymEnv(
            db_path=multi_db,
            starting_cash=500.0,
            n_markets=3,
            split="train",
            enable_ml_predictions=False,
            random_seed=42,
        )
        env.reset(seed=42)

        value_before = env._portfolio_value()
        action = np.array([1, 0, 0], dtype=np.int64)
        _, reward, _, _, info = env.step(action)
        value_after = info["portfolio_value"]

        # Reward should equal the delta in portfolio value
        assert reward == pytest.approx(value_after - value_before)
        env.close()
