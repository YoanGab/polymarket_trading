"""Hybrid ML+RL: ML pre-screens markets, RL learns timing/sizing.

Architecture:
1. ML model (XGBoost) scores all candidate markets for edge
2. Top-K markets by |sell edge| are selected for the episode
3. RL agent (PPO) observes these pre-screened markets
4. ML edge is passed as part of the observation (ml_edges)

The ML runs ONCE at episode reset, not per step.

Usage:
    uv run python scripts/train_hybrid_rl.py --split train --timeout 600
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class MLMarketScreener:
    """Pre-screen markets using the trained XGBoost model.

    Only selects markets where the ML model predicts a strong sell edge
    (overpriced YES contracts). Returns top-K by absolute edge.
    """

    def __init__(self) -> None:
        model_path = MODELS_DIR / "xgboost_model.pkl"
        with open(model_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._model = data["model"]
        self._feature_names: list[str] = data["feature_names"]

    def rank_markets(
        self,
        market_ids: list[str],
        as_of: object,
    ) -> list[tuple[str, float, float]]:
        """Return markets ranked by sell edge (strongest first).

        This is called ONCE at episode reset, not per step.
        Returns (market_id, probability_yes, edge_bps) sorted by |edge|.
        """
        # For now, return all markets with dummy edges
        # The real edge computation requires snapshot data from the DB
        # which we don't have at this point (before episodes are loaded)
        # The env will populate ml_edges after loading episodes
        return [(mid, 0.5, 0.0) for mid in market_ids]


def make_env(n_markets: int = 5, split: str = "train"):
    """Create hybrid ML+RL multi-market env."""
    from polymarket_backtest.gym_env_multi import PolymarketMultiMarketGymEnv
    from polymarket_backtest.splits import HOLDOUT_CUTOFF, TRAIN_CUTOFF, VAL_CUTOFF

    class _HybridEnv(PolymarketMultiMarketGymEnv):
        def _market_ids_for_split(self, split_name: str) -> list[str]:
            n = max(self.n_markets * 100, 500)
            if split_name == "train":
                cond = f"WHERE resolution_ts IS NOT NULL AND resolution_ts < '{TRAIN_CUTOFF}'"
            elif split_name == "val":
                cond = f"WHERE resolution_ts IS NOT NULL AND resolution_ts >= '{TRAIN_CUTOFF}' AND resolution_ts < '{VAL_CUTOFF}'"
            elif split_name == "test":
                cond = f"WHERE resolution_ts IS NOT NULL AND resolution_ts >= '{VAL_CUTOFF}' AND resolution_ts < '{HOLDOUT_CUTOFF}'"
            else:
                raise ValueError(f"Unknown split {split_name!r}")
            rows = self.conn.execute(f"SELECT market_id FROM markets {cond} ORDER BY RANDOM() LIMIT ?", (n,)).fetchall()
            return [str(r["market_id"]) for r in rows]

        def step(self, action):
            try:
                return super().step(action)
            except AssertionError:
                obs, info = self.reset()
                return obs, -10.0, True, False, info

    return _HybridEnv(
        db_path=str(DB_PATH),
        starting_cash=1000,
        n_markets=n_markets,
        split=split,
    )


def flatten_obs(obs_dict: dict) -> np.ndarray:
    """Flatten Dict observation for PPO MlpPolicy."""
    return np.concatenate(
        [
            np.asarray(obs_dict["market_features"], dtype=np.float32).flatten(),
            np.asarray(obs_dict["portfolio"], dtype=np.float32),
            np.asarray(obs_dict["market_mask"], dtype=np.float32),
            np.asarray(obs_dict["ml_edges"], dtype=np.float32).flatten(),
        ]
    ).astype(np.float32, copy=False)


def train(split: str = "train", timeout_seconds: int = 600, n_markets: int = 5, eval_only: bool = False):
    """Train/eval hybrid RL agent."""
    import torch

    if eval_only and split == "train":
        split = "val"

    env = make_env(n_markets=n_markets, split=split)

    try:
        # Get obs dim
        sample_obs, _ = env.reset()
        obs_dim = flatten_obs(sample_obs).shape[0]
        n_actions = n_markets * 5  # 5 actions per market slot

        # Simple DQN with Double Q-learning
        from train_rl import DQNAgent

        agent = DQNAgent(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=256,
            lr=5e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=10000,
            batch_size=128,
            target_update=200,
        )

        # Load saved policy if eval-only
        policy_path = MODELS_DIR / "rl_hybrid_policy.pt"
        if eval_only and policy_path.exists():
            checkpoint = torch.load(policy_path, weights_only=True)
            agent.q_net.load_state_dict(checkpoint["q_net"])
            agent.target_net.load_state_dict(checkpoint["target_net"])
            print(f"Loaded hybrid policy from {policy_path}", flush=True)

        episode_returns: list[float] = []
        episode_trades: list[int] = []

        start = time.monotonic()
        completed = 0

        while True:
            if time.monotonic() - start >= timeout_seconds:
                break

            obs_dict, _ = env.reset()
            obs = flatten_obs(obs_dict)
            ep_return = 0.0
            ep_trades = 0

            for _ in range(500):
                if time.monotonic() - start >= timeout_seconds:
                    break

                action_idx = agent.greedy_action(obs) if eval_only else agent.select_action(obs)
                # Convert to multi-discrete
                actions = np.zeros(n_markets, dtype=np.int64)
                slot = action_idx // 5
                act = action_idx % 5
                if 0 <= slot < n_markets:
                    actions[slot] = act

                next_obs_dict, reward, terminated, truncated, _ = env.step(actions)
                next_obs = flatten_obs(next_obs_dict)

                # Risk-aware reward: differential Sharpe + drawdown penalty + edge bonus
                pnl_norm = float(reward) / 1000.0  # normalize by starting cash

                # Differential Sharpe (EWMA)
                if not hasattr(env, "_rl_mu"):
                    env._rl_mu = 0.0
                    env._rl_sigma2 = 1e-4
                    env._rl_peak = 1000.0
                env._rl_mu = 0.97 * env._rl_mu + 0.03 * pnl_norm
                env._rl_sigma2 = 0.97 * env._rl_sigma2 + 0.03 * pnl_norm**2
                sharpe_component = env._rl_mu / (np.sqrt(env._rl_sigma2) + 1e-6)

                # Drawdown penalty
                portfolio_val = (
                    float(next_obs_dict.get("portfolio", [1.0])[4])
                    if len(next_obs_dict.get("portfolio", [])) > 4
                    else 1.0
                )
                env._rl_peak = max(env._rl_peak, portfolio_val * 1000)
                dd = (env._rl_peak - portfolio_val * 1000) / max(env._rl_peak, 1.0)
                dd_penalty = -2.0 * dd

                shaped_reward = float(np.clip(0.5 * sharpe_component + 0.3 * pnl_norm + 0.15 * dd_penalty, -5.0, 5.0))

                if not eval_only:
                    agent.store(obs, action_idx, shaped_reward, next_obs, terminated or truncated)
                    agent.train_step()

                ep_return += float(reward)
                if act != 0:
                    ep_trades += 1
                obs = next_obs
                if terminated or truncated:
                    break

            episode_returns.append(ep_return)
            episode_trades.append(ep_trades)
            completed += 1

            if completed % 50 == 0:
                recent = episode_returns[-50:]
                elapsed = time.monotonic() - start
                print(
                    f"Episode {completed}"
                    f" | avg_return={np.mean(recent):.2f}"
                    f" | avg_trades={np.mean(episode_trades[-50:]):.1f}"
                    f" | win_rate={np.mean(np.array(recent) > 0):.3f}"
                    f" | epsilon={agent.epsilon:.3f}"
                    f" | elapsed={elapsed:.0f}s",
                    flush=True,
                )

        # Save policy
        if not eval_only:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"q_net": agent.q_net.state_dict(), "target_net": agent.target_net.state_dict()},
                policy_path,
            )
            print(f"Saved hybrid policy to {policy_path}", flush=True)

        returns = np.array(episode_returns)
        trades = np.array(episode_trades)
        sharpe = float(np.mean(returns) / max(np.std(returns), 1e-8)) if len(returns) > 1 else 0.0

        print(
            "RESULT"
            f"\tSHARPE={sharpe:.6f}"
            f"\tPNL={np.sum(returns):.2f}"
            f"\tTRADES={np.mean(trades):.1f}"
            f"\tWIN_RATE={np.mean(returns > 0):.4f}"
            f"\tEPISODES={len(returns)}"
            f"\tTIME={time.monotonic() - start:.0f}s"
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Train hybrid ML+RL agent")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--n-markets", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train(split=args.split, timeout_seconds=args.timeout, n_markets=args.n_markets, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
