"""Train a DQN agent for Polymarket multi-market trading.

Usage:
    uv run python scripts/train_rl.py [--timeout 300] [--n-markets 10]

The autoresearch agent can edit this file to try different:
- RL algorithms (DQN, Rainbow, PPO)
- Network architectures
- Reward shaping
- Observation engineering
- Hyperparameters
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"

ObsDict = dict[str, np.ndarray]
_ACTION_TYPES_PER_SLOT = 5


def make_env(n_markets: int = 10, split: str = "train"):
    """Create the default multi-market Polymarket gym environment."""
    from polymarket_backtest.gym_env_multi import PolymarketMultiMarketGymEnv
    from polymarket_backtest.splits import HOLDOUT_CUTOFF, TRAIN_CUTOFF, VAL_CUTOFF

    class _TrainingPolymarketMultiMarketGymEnv(PolymarketMultiMarketGymEnv):
        def _market_ids_for_split(self, split_name: str) -> list[str]:
            market_universe_size = max(self.n_markets * 100, 500)
            if split_name == "all":
                condition = ""
                params: tuple[str, ...] = ()
            elif split_name == "train":
                condition = "WHERE resolution_ts IS NOT NULL AND resolution_ts < ?"
                params = (TRAIN_CUTOFF,)
            elif split_name == "val":
                condition = "WHERE resolution_ts IS NOT NULL AND resolution_ts >= ? AND resolution_ts < ?"
                params = (TRAIN_CUTOFF, VAL_CUTOFF)
            elif split_name == "test":
                condition = "WHERE resolution_ts IS NOT NULL AND resolution_ts >= ? AND resolution_ts < ?"
                params = (VAL_CUTOFF, HOLDOUT_CUTOFF)
            else:
                raise ValueError(f"Unknown split {split_name!r}")

            rows = self.conn.execute(
                f"""
                SELECT market_id
                FROM markets
                {condition}
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (*params, market_universe_size),
            ).fetchall()
            return [str(row["market_id"]) for row in rows]

    return _TrainingPolymarketMultiMarketGymEnv(
        db_path=str(DB_PATH),
        starting_cash=1000,
        n_markets=n_markets,
        split=split,
    )


def flatten_obs(obs_dict: ObsDict) -> np.ndarray:
    """Flatten the multi-market Dict observation into a single DQN vector."""
    return np.concatenate(
        [
            np.asarray(obs_dict["market_features"], dtype=np.float32).flatten(),
            np.asarray(obs_dict["portfolio"], dtype=np.float32),
            np.asarray(obs_dict["market_mask"], dtype=np.float32),
            np.asarray(obs_dict["ml_edges"], dtype=np.float32).flatten(),
        ]
    ).astype(np.float32, copy=False)


def action_index_to_multidiscrete(action_idx: int, n_markets: int) -> np.ndarray:
    """Map a single DQN action index to a MultiDiscrete action vector."""
    actions = np.zeros(n_markets, dtype=np.int64)
    bounded_idx = int(np.clip(action_idx, 0, max(n_markets * _ACTION_TYPES_PER_SLOT - 1, 0)))
    slot_idx = bounded_idx // _ACTION_TYPES_PER_SLOT
    action_type = bounded_idx % _ACTION_TYPES_PER_SLOT
    if 0 <= slot_idx < n_markets:
        actions[slot_idx] = action_type
    return actions


def _position_signature(env: Any, market_id: str | None) -> tuple[float, float]:
    """Capture YES/NO quantities for a single market."""
    if market_id is None:
        return (0.0, 0.0)

    yes_qty = 0.0
    no_qty = 0.0
    for position in env._core.portfolio.positions.values():
        if position.market_id != market_id or position.quantity <= 0:
            continue
        if position.is_no_bet:
            no_qty += float(position.quantity)
        else:
            yes_qty += float(position.quantity)
    return (yes_qty, no_qty)


def _did_execute_trade(before_sig: tuple[float, float], after_sig: tuple[float, float], tolerance: float = 1e-9) -> int:
    return int(any(abs(after - before) > tolerance for before, after in zip(before_sig, after_sig)))


# ── Simple DQN Agent ──────────────────────────────────────────────


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Simple DQN with epsilon-greedy exploration."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        batch_size: int = 64,
        target_update: int = 100,
    ):
        import torch
        import torch.nn as nn

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self.target_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer()
        self.torch = torch

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.steps_done / self.epsilon_decay
        )

    def greedy_action(self, state: np.ndarray) -> int:
        with self.torch.no_grad():
            state_t = self.torch.tensor(state, dtype=self.torch.float32).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self.greedy_action(state)

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = self.torch.tensor(states, dtype=self.torch.float32)
        actions_t = self.torch.tensor(actions, dtype=self.torch.int64).unsqueeze(1)
        rewards_t = self.torch.tensor(rewards, dtype=self.torch.float32)
        next_states_t = self.torch.tensor(next_states, dtype=self.torch.float32)
        dones_t = self.torch.tensor(dones, dtype=self.torch.float32)

        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        with self.torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


# ── Training Loop ─────────────────────────────────────────────────


def train(
    n_episodes: int = 0,
    max_steps: int = 500,
    n_markets: int = 10,
    eval_every: int = 50,
    split: str = "train",
    eval_only: bool = False,
    timeout_seconds: int = 300,
):
    if eval_only and split == "train":
        split = "val"

    env = make_env(n_markets=n_markets, split=split)
    try:
        sample_obs, _ = env.reset()
        obs_dim = flatten_obs(sample_obs).shape[0]
        n_actions = n_markets * _ACTION_TYPES_PER_SLOT

        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)

        episode_returns: list[float] = []
        episode_trades: list[int] = []
        best_return = float("-inf")

        start = time.monotonic()
        completed_episodes = 0

        while True:
            elapsed = time.monotonic() - start
            if elapsed >= timeout_seconds:
                break
            if n_episodes > 0 and completed_episodes >= n_episodes:
                break

            obs_dict, _ = env.reset()
            obs = flatten_obs(obs_dict)
            ep_return = 0.0
            ep_trades = 0
            steps_taken = 0
            timed_out = False

            for _step in range(max_steps):
                if time.monotonic() - start >= timeout_seconds:
                    timed_out = True
                    break

                action_idx = agent.greedy_action(obs) if eval_only else agent.select_action(obs)
                multi_action = action_index_to_multidiscrete(action_idx, n_markets)

                slot_idx = action_idx // _ACTION_TYPES_PER_SLOT
                action_type = action_idx % _ACTION_TYPES_PER_SLOT
                targeted_market_id = None
                before_signature = (0.0, 0.0)
                if 0 <= slot_idx < n_markets:
                    targeted_market_id = env._slot_market_ids[slot_idx]
                    before_signature = _position_signature(env, targeted_market_id)

                next_obs_dict, reward, terminated, truncated, _info = env.step(multi_action)
                next_obs = flatten_obs(next_obs_dict)

                if action_type != 0:
                    after_signature = _position_signature(env, targeted_market_id)
                    ep_trades += _did_execute_trade(before_signature, after_signature)

                if not eval_only:
                    agent.store(obs, action_idx, reward, next_obs, terminated or truncated)
                    agent.train_step()

                ep_return += float(reward)
                steps_taken += 1
                obs = next_obs
                if terminated or truncated:
                    break

            if timed_out and steps_taken == 0:
                break

            episode_returns.append(ep_return)
            episode_trades.append(ep_trades)
            best_return = max(best_return, ep_return)
            completed_episodes += 1

            if completed_episodes % eval_every == 0:
                recent_returns = episode_returns[-eval_every:]
                recent_trades = episode_trades[-eval_every:]
                avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                avg_trades = float(np.mean(recent_trades)) if recent_trades else 0.0
                win_rate = float(np.mean(np.asarray(recent_returns) > 0)) if recent_returns else 0.0
                elapsed = time.monotonic() - start
                print(
                    f"Episode {completed_episodes}"
                    f" | avg_return={avg_return:.4f}"
                    f" | avg_trades={avg_trades:.2f}"
                    f" | win_rate={win_rate:.4f}"
                    f" | epsilon={agent.epsilon:.3f}"
                    f" | buffer={len(agent.replay)}"
                    f" | elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if timed_out:
                break

        elapsed = time.monotonic() - start
        avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        avg_trades = float(np.mean(episode_trades)) if episode_trades else 0.0
        win_rate = float(np.mean(np.asarray(episode_returns) > 0)) if episode_returns else 0.0
        best_return = 0.0 if best_return == float("-inf") else float(best_return)

        print(
            "RESULT"
            f"\tEPISODE_RETURN={avg_return:.6f}"
            f"\tTRADES={avg_trades:.6f}"
            f"\tWIN_RATE={win_rate:.6f}"
            f"\tBEST_RETURN={best_return:.6f}"
            f"\tTRAIN_TIME={elapsed:.2f}"
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument("--episodes", type=int, default=0, help="Max episodes (0 = run until timeout)")
    parser.add_argument(
        "--timeout", type=int, default=300, help="Training time budget in seconds (default: 300 = 5 min)"
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n-markets", type=int, default=10, help="Number of simultaneous market slots")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--split", default="train", help="Split to use: train, val, test (default: train)")
    parser.add_argument("--eval-only", action="store_true", help="Run greedy evaluation on the validation split")
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        n_markets=args.n_markets,
        eval_every=args.eval_every,
        split=args.split,
        eval_only=args.eval_only,
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    main()
