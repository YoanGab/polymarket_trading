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


def make_env(split: str = "train"):
    """Create a single-market Polymarket gym environment."""
    # Get liquid market IDs for the split
    from polymarket_backtest import db
    from polymarket_backtest.gym_env import PolymarketGymEnv
    from polymarket_backtest.splits import HOLDOUT_CUTOFF, TRAIN_CUTOFF, VAL_CUTOFF

    conn = db.connect(DB_PATH)
    if split == "train":
        condition = f"AND m.resolution_ts < '{TRAIN_CUTOFF}'"
    elif split == "val":
        condition = f"AND m.resolution_ts >= '{TRAIN_CUTOFF}' AND m.resolution_ts < '{VAL_CUTOFF}'"
    else:
        condition = f"AND m.resolution_ts >= '{VAL_CUTOFF}' AND m.resolution_ts < '{HOLDOUT_CUTOFF}'"

    rows = conn.execute(
        f"""
        SELECT m.market_id
        FROM markets m
        JOIN market_resolutions mr ON mr.market_id = m.market_id
        WHERE 1=1 {condition}
        ORDER BY RANDOM()
        LIMIT 500
        """
    ).fetchall()
    market_ids = [str(r["market_id"]) for r in rows]
    conn.close()

    if not market_ids:
        raise ValueError(f"No markets found for split {split}")

    return PolymarketGymEnv(
        db_path=str(DB_PATH),
        starting_cash=1000,
        market_ids=market_ids,
        split=split,
    )


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
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=0,
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
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_return += float(reward)
            if int(action) != 0:
                ep_trades += 1
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
