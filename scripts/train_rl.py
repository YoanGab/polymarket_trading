"""Train an RL agent for Polymarket trading.

Usage:
    uv run python scripts/train_rl.py [--episodes 500] [--algo dqn]

The autoresearch agent can edit this file to try different:
- RL algorithms (DQN, Rainbow, PPO)
- Network architectures
- Reward shaping
- Observation engineering
- Hyperparameters
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"


def make_env(max_markets: int = 100, split: str = "val"):
    """Create the Polymarket gym environment."""
    import sqlite3

    from polymarket_backtest.gym_env import PolymarketGymEnv
    from polymarket_backtest.splits import TRAIN_CUTOFF, VAL_CUTOFF

    # Pre-select market_ids for the split
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    split_filter = {
        "train": f"m.resolution_ts < '{TRAIN_CUTOFF}'",
        "val": f"m.resolution_ts >= '{TRAIN_CUTOFF}' AND m.resolution_ts < '{VAL_CUTOFF}'",
        "test": f"m.resolution_ts >= '{VAL_CUTOFF}'",
    }
    filt = split_filter.get(split, split_filter["val"])
    query = f"SELECT m.market_id FROM markets m JOIN market_resolutions mr ON mr.market_id = m.market_id WHERE {filt}"
    if max_markets > 0:
        query += f" ORDER BY RANDOM() LIMIT {max_markets}"
    rows = conn.execute(query).fetchall()
    conn.close()
    market_ids = [str(r["market_id"]) for r in rows]

    return PolymarketGymEnv(
        db_path=str(DB_PATH),
        starting_cash=1000,
        market_ids=market_ids,
        split=split,
    )


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

        # Q-network
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Target network
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

    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with self.torch.no_grad():
            state_t = self.torch.tensor(state, dtype=self.torch.float32).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = self.torch.tensor(states)
        actions_t = self.torch.tensor(actions).unsqueeze(1)
        rewards_t = self.torch.tensor(rewards)
        next_states_t = self.torch.tensor(next_states)
        dones_t = self.torch.tensor(dones)

        # Current Q values
        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q values (Double DQN: use q_net to select, target_net to evaluate)
        with self.torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


# ── Training Loop ─────────────────────────────────────────────────


def train(
    n_episodes: int = 100_000,
    max_steps: int = 500,
    max_markets: int = 0,
    eval_every: int = 50,
    split: str = "train",
    eval_only: bool = False,
    timeout_seconds: int = 300,
):
    env = make_env(max_markets=max_markets, split=split)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions)

    # Training metrics
    episode_returns = []
    episode_trades = []
    best_avg_return = float("-inf")

    start = time.monotonic()

    for ep in range(n_episodes):
        if time.monotonic() - start > timeout_seconds:
            print(f"Timeout after {ep} episodes ({timeout_seconds}s)", flush=True)
            break
        obs, _ = env.reset()
        ep_return = 0.0
        ep_trades = 0

        for step in range(max_steps):
            if eval_only:
                # Greedy action, no exploration
                with agent.torch.no_grad():
                    state_t = agent.torch.tensor(obs, dtype=agent.torch.float32).unsqueeze(0)
                    action = int(agent.q_net(state_t).argmax(dim=1).item())
            else:
                action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            if not eval_only:
                agent.store(obs, action, reward, next_obs, terminated or truncated)
                agent.train_step()

            ep_return += reward
            if info.get("filled_quantity", 0) > 0:
                ep_trades += 1

            obs = next_obs
            if terminated or truncated:
                break

        episode_returns.append(ep_return)
        episode_trades.append(ep_trades)

        # Periodic evaluation
        if (ep + 1) % eval_every == 0:
            recent_returns = episode_returns[-eval_every:]
            recent_trades = episode_trades[-eval_every:]
            avg_return = np.mean(recent_returns)
            avg_trades = np.mean(recent_trades)
            win_rate = np.mean([1 if r > 0 else 0 for r in recent_returns])

            elapsed = time.monotonic() - start
            print(
                f"Episode {ep + 1}/{n_episodes} | "
                f"avg_return={avg_return:+.2f} | "
                f"avg_trades={avg_trades:.1f} | "
                f"win_rate={win_rate:.0%} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"buffer={len(agent.replay)} | "
                f"elapsed={elapsed:.0f}s"
            )

            if avg_return > best_avg_return:
                best_avg_return = avg_return

    # Final summary
    elapsed = time.monotonic() - start
    avg_return = np.mean(episode_returns[-100:])
    avg_trades = np.mean(episode_trades[-100:])
    win_rate = np.mean([1 if r > 0 else 0 for r in episode_returns[-100:]])

    print(
        f"\nRESULT\tEPISODE_RETURN={avg_return:+.4f}\tTRADES={avg_trades:.1f}\t"
        f"WIN_RATE={win_rate:.2%}\tBEST_RETURN={best_avg_return:+.4f}\t"
        f"TRAIN_TIME={elapsed:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument("--episodes", type=int, default=0, help="Max episodes (0 = run until timeout)")
    parser.add_argument(
        "--timeout", type=int, default=300, help="Training time budget in seconds (default: 300 = 5 min)"
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--max-markets", type=int, default=0, help="Max markets to pre-select (0 = ALL markets in split)"
    )
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--split", default="train", help="Split to use: 'train' for training, 'val' for evaluation (default: train)"
    )
    parser.add_argument("--eval-only", action="store_true", help="Evaluate only (no learning). Use with --split val.")
    args = parser.parse_args()

    train(
        n_episodes=args.episodes if args.episodes > 0 else 100_000,
        max_steps=args.max_steps,
        max_markets=args.max_markets,
        eval_every=args.eval_every,
        split=args.split,
        eval_only=args.eval_only,
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    main()
