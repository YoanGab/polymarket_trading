"""Train a PPO agent for single-market Polymarket trading.

Uses Stable-Baselines3 PPO with the single-market gym env (Discrete(12)).
Much simpler action space than multi-market DQN (12 vs 50 actions).

Usage:
    uv run python scripts/train_rl_ppo.py --split train --timeout 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def make_env(split: str = "train", n_markets: int = 5):
    """Create multi-market gym env (fast, pre-cached market states)."""
    from polymarket_backtest.gym_env_multi import PolymarketMultiMarketGymEnv
    from polymarket_backtest.splits import HOLDOUT_CUTOFF, TRAIN_CUTOFF, VAL_CUTOFF

    class _PPOEnv(PolymarketMultiMarketGymEnv):
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

    return _PPOEnv(db_path=str(DB_PATH), starting_cash=1000, n_markets=n_markets, split=split)


def train(split: str = "train", timeout_seconds: int = 300, eval_only: bool = False):
    """Train PPO agent."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    if eval_only and split == "train":
        split = "val"

    env = make_env(split=split)

    class TimeoutCallback(BaseCallback):
        def __init__(self, timeout: float):
            super().__init__()
            self.start = time.monotonic()
            self.timeout = timeout

        def _on_step(self) -> bool:
            return (time.monotonic() - self.start) < self.timeout

    policy_path = MODELS_DIR / "rl_ppo_policy.zip"

    if eval_only and policy_path.exists():
        model = PPO.load(str(policy_path), env=env)
        print(f"Loaded PPO policy from {policy_path}", flush=True)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=1,
        )

    start = time.monotonic()

    if not eval_only:
        callback = TimeoutCallback(timeout_seconds)
        model.learn(total_timesteps=10_000_000, callback=callback, progress_bar=False)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(policy_path))
        print(f"Saved PPO policy to {policy_path}", flush=True)

    # Evaluate
    episode_returns = []
    episode_trades = []
    eval_start = time.monotonic()
    eval_timeout = min(120.0, timeout_seconds * 0.3)

    for ep in range(500):
        if time.monotonic() - eval_start > eval_timeout:
            break
        obs, info = env.reset()
        ep_return = 0.0
        ep_trades = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_trades += int(np.sum(action != 0))
            done = terminated or truncated
        episode_returns.append(ep_return)
        episode_trades.append(ep_trades)

    env.close()

    elapsed = time.monotonic() - start
    returns = np.array(episode_returns)
    trades = np.array(episode_trades)
    sharpe = float(np.mean(returns) / max(np.std(returns), 1e-8)) if len(returns) > 1 else 0.0

    print(
        "RESULT"
        f"\tEPISODE_RETURN={np.mean(returns):.6f}"
        f"\tTRADES={np.mean(trades):.6f}"
        f"\tWIN_RATE={np.mean(returns > 0):.6f}"
        f"\tSHARPE={sharpe:.6f}"
        f"\tPNL={np.sum(returns):.2f}"
        f"\tEPISODES={len(returns)}"
        f"\tTRAIN_TIME={elapsed:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO RL agent")
    parser.add_argument("--timeout", type=int, default=300, help="Training time budget in seconds")
    parser.add_argument("--split", default="train", help="Split to use: train, val, test")
    parser.add_argument("--eval-only", action="store_true", help="Eval with frozen policy")
    args = parser.parse_args()

    train(split=args.split, timeout_seconds=args.timeout, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
