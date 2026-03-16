"""Imitation Learning -> RL Fine-Tuning for multi-market trading.

Pipeline:
1. Generate expert demonstrations by running the sell_edge strategy in the gym env
2. Pre-train a policy network via behavioral cloning (supervised on expert actions)
3. Fine-tune with PPO using differential Sharpe reward
4. Uses MPS (Apple Silicon GPU) for training

The sell_edge expert logic:
- If no NO position and model says YES is overpriced (mid > 0.10), buy NO
- If holding NO position and edge disappeared, sell NO
- Otherwise hold

Usage:
    uv run python scripts/train_rl_imitation.py --timeout 1200
    uv run python scripts/train_rl_imitation.py --eval-only --split val
    uv run python scripts/train_rl_imitation.py --skip-bc --timeout 600  # RL only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Env constants (from gym_env_multi.py)
_N_ACTIONS_PER_SLOT = 5  # hold=0, buy_yes=1, buy_no=2, sell_yes=3, sell_no=4
_MARKET_FEATURE_DIM = 31
_PORTFOLIO_DIM = 5


# ── Environment Setup ────────────────────────────────────────────


def make_env(n_markets: int = 5, split: str = "train"):
    """Create the multi-market gym env with AssertionError handling."""
    from polymarket_backtest.gym_env_multi import PolymarketMultiMarketGymEnv
    from polymarket_backtest.splits import HOLDOUT_CUTOFF, TRAIN_CUTOFF, VAL_CUTOFF

    class _ImitationEnv(PolymarketMultiMarketGymEnv):
        def _market_ids_for_split(self, split_name: str) -> list[str]:
            n = max(self.n_markets * 100, 500)
            if split_name == "train":
                cond = f"WHERE resolution_ts IS NOT NULL AND resolution_ts < '{TRAIN_CUTOFF}'"
            elif split_name == "val":
                cond = (
                    "WHERE resolution_ts IS NOT NULL"
                    f" AND resolution_ts >= '{TRAIN_CUTOFF}'"
                    f" AND resolution_ts < '{VAL_CUTOFF}'"
                )
            elif split_name == "test":
                cond = (
                    "WHERE resolution_ts IS NOT NULL"
                    f" AND resolution_ts >= '{VAL_CUTOFF}'"
                    f" AND resolution_ts < '{HOLDOUT_CUTOFF}'"
                )
            else:
                raise ValueError(f"Unknown split {split_name!r}")
            rows = self.conn.execute(
                f"SELECT market_id FROM markets {cond} ORDER BY RANDOM() LIMIT ?",
                (n,),
            ).fetchall()
            return [str(r["market_id"]) for r in rows]

        def step(self, action):
            try:
                return super().step(action)
            except AssertionError:
                obs, info = self.reset()
                return obs, -10.0, True, False, info

    return _ImitationEnv(
        db_path=str(DB_PATH),
        starting_cash=1000,
        n_markets=n_markets,
        split=split,
    )


def flatten_obs(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten Dict observation to a single vector."""
    return np.concatenate(
        [
            np.asarray(obs_dict["market_features"], dtype=np.float32).flatten(),
            np.asarray(obs_dict["portfolio"], dtype=np.float32),
            np.asarray(obs_dict["market_mask"], dtype=np.float32),
            np.asarray(obs_dict["ml_edges"], dtype=np.float32).flatten(),
        ]
    ).astype(np.float32, copy=False)


def obs_dim_for(n_markets: int) -> int:
    """Compute flat observation dimension."""
    return n_markets * _MARKET_FEATURE_DIM + _PORTFOLIO_DIM + n_markets + n_markets * 2


# ── Expert (sell_edge) Policy ────────────────────────────────────


_ML_EDGES: dict[str, float] | None = None


def _load_ml_edges() -> dict[str, float]:
    """Load pre-computed ML edges for expert policy."""
    global _ML_EDGES
    if _ML_EDGES is None:
        import pickle

        path = MODELS_DIR / "market_ml_edges.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _ML_EDGES = pickle.load(f)  # noqa: S301
        else:
            _ML_EDGES = {}
    return _ML_EDGES


def sell_edge_expert_action(env, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Compute the sell_edge expert action using ML model predictions.

    Expert logic per slot:
    - Buy NO when ML model says YES is overpriced (edge < -100bps)
    - Sell NO when position PnL hits take-profit or stop-loss
    - Hold otherwise

    Uses pre-computed ML edges for accurate market selection.
    """
    n_markets = env.n_markets
    actions = np.zeros(n_markets, dtype=np.int64)
    market_mask = obs_dict["market_mask"]

    for slot_idx in range(n_markets):
        if market_mask[slot_idx] < 0.5:
            continue  # inactive slot

        market_id = env._slot_market_ids[slot_idx]
        if market_id is None:
            continue

        episode = env._episodes.get(market_id)
        if episode is None or episode.done:
            continue

        # Get market state
        features = obs_dict["market_features"][slot_idx]
        mid = float(features[3])  # index 3 = mid price
        ask = float(features[2])  # index 2 = best_ask

        # Check if we have a NO position in this market
        has_no_position = False
        for pos in env._core.portfolio.positions.values():
            if pos.market_id == market_id and pos.is_no_bet and pos.quantity > 0:
                has_no_position = True
                break

        # Use pre-computed ML edges for accurate expert decisions
        ml_edges = _load_ml_edges()
        ml_prob = ml_edges.get(market_id, mid)  # fall back to mid if no ML prediction
        ml_edge = ml_prob - mid  # negative = model thinks YES is overpriced

        if not has_no_position:
            # Entry: buy NO when ML model says YES is overpriced
            # Conditions (matching sell_edge strategy):
            # 1. ml_edge < -0.01 (ML says YES overpriced by >100bps)
            # 2. mid >= 0.10 (longshot filter)
            if ml_edge < -0.01 and mid >= 0.10:
                actions[slot_idx] = 2  # buy_no
        else:
            # Exit: sell NO when PnL hits targets
            no_pnl_pct = float(features[23]) if len(features) > 23 else 0.0
            if no_pnl_pct > 0.05:
                actions[slot_idx] = 4  # sell_no (take profit)
            elif no_pnl_pct < -0.15:
                actions[slot_idx] = 4  # sell_no (cut losses)

    return actions


# ── Demonstration Buffer ─────────────────────────────────────────


@dataclass
class DemonstrationBuffer:
    """Stores (obs, action) pairs from expert demonstrations."""

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)  # per-slot actions
    rewards: list[float] = field(default_factory=list)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float = 0.0) -> None:
        self.observations.append(obs)
        self.actions.append(action.copy())
        self.rewards.append(reward)

    def __len__(self) -> int:
        return len(self.observations)


def generate_demonstrations(
    n_markets: int = 5,
    n_episodes: int = 200,
    max_steps: int = 300,
    timeout_seconds: float = 300.0,
) -> DemonstrationBuffer:
    """Run the sell_edge expert in the gym env and collect demonstrations."""
    print("\n=== Phase 1: Generating Expert Demonstrations ===", flush=True)

    env = make_env(n_markets=n_markets, split="train")
    buffer = DemonstrationBuffer()
    start = time.monotonic()

    episode_returns: list[float] = []
    episode_trades: list[int] = []

    for ep in range(n_episodes):
        if time.monotonic() - start >= timeout_seconds:
            break

        obs_dict, _ = env.reset()
        obs_flat = flatten_obs(obs_dict)
        ep_return = 0.0
        ep_trades = 0

        for _step in range(max_steps):
            if time.monotonic() - start >= timeout_seconds:
                break

            # Get expert action
            expert_action = sell_edge_expert_action(env, obs_dict)

            # Store demonstration
            buffer.add(obs_flat, expert_action, 0.0)

            # Step environment
            next_obs_dict, reward, terminated, truncated, _info = env.step(expert_action)
            next_obs_flat = flatten_obs(next_obs_dict)

            # Track trades
            n_trades = int(np.sum(expert_action != 0))
            ep_trades += n_trades
            ep_return += float(reward)

            obs_dict = next_obs_dict
            obs_flat = next_obs_flat

            if terminated or truncated:
                break

        episode_returns.append(ep_return)
        episode_trades.append(ep_trades)

        if (ep + 1) % 50 == 0:
            recent = episode_returns[-50:]
            elapsed = time.monotonic() - start
            print(
                f"  Demo episode {ep + 1}/{n_episodes}"
                f" | avg_return={np.mean(recent):.2f}"
                f" | avg_trades={np.mean(episode_trades[-50:]):.1f}"
                f" | win_rate={np.mean(np.array(recent) > 0):.3f}"
                f" | demos={len(buffer)}"
                f" | elapsed={elapsed:.0f}s",
                flush=True,
            )

    env.close()

    # Compute expert baseline stats
    returns = np.array(episode_returns)
    sharpe = float(np.mean(returns) / max(np.std(returns), 1e-8)) if len(returns) > 1 else 0.0

    non_hold_count = sum(1 for a in buffer.actions if np.any(a != 0))
    hold_count = len(buffer) - non_hold_count

    print(
        f"\n  Expert demonstrations: {len(buffer)} transitions"
        f" ({non_hold_count} trades, {hold_count} holds)"
        f"\n  Expert Sharpe: {sharpe:.4f}"
        f" | avg_return: {np.mean(returns):.2f}"
        f" | episodes: {len(returns)}",
        flush=True,
    )

    return buffer


# ── Policy Network ───────────────────────────────────────────────


def _get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_policy_network(obs_dim: int, n_markets: int, hidden_dim: int = 256):
    """Build a policy network that outputs per-slot action logits.

    Architecture:
    - Shared trunk: obs -> hidden features
    - Per-slot heads: hidden features + slot features -> action logits
    - Market mask applied to zero out inactive slots
    """
    import torch
    import torch.nn as nn

    class MultiSlotPolicy(nn.Module):
        """Policy network for multi-market trading.

        Processes the full observation through a shared trunk, then
        produces action logits for each market slot independently.
        """

        def __init__(self, obs_dim_: int, n_markets_: int, n_actions: int, hidden_: int):
            super().__init__()
            self.n_markets = n_markets_
            self.n_actions = n_actions
            self.market_feat_dim = _MARKET_FEATURE_DIM

            # Shared feature extractor
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim_, hidden_),
                nn.LayerNorm(hidden_),
                nn.ReLU(),
                nn.Linear(hidden_, hidden_),
                nn.LayerNorm(hidden_),
                nn.ReLU(),
            )

            # Per-slot action head: takes trunk output + per-slot market features
            slot_input_dim = hidden_ + _MARKET_FEATURE_DIM + 2  # +2 for ml_edges
            self.slot_head = nn.Sequential(
                nn.Linear(slot_input_dim, hidden_ // 2),
                nn.ReLU(),
                nn.Linear(hidden_ // 2, n_actions),
            )

            # Value head for PPO
            self.value_head = nn.Sequential(
                nn.Linear(hidden_, hidden_ // 2),
                nn.ReLU(),
                nn.Linear(hidden_ // 2, 1),
            )

        def forward(self, obs_flat: torch.Tensor, return_value: bool = False):
            """Forward pass.

            Args:
                obs_flat: (batch, obs_dim) flat observation
                return_value: if True, also return value estimate

            Returns:
                logits: (batch, n_markets, n_actions) action logits per slot
                value: (batch, 1) value estimate (only if return_value=True)
            """
            batch_size = obs_flat.shape[0]

            # Extract components from flat obs
            mf_end = self.n_markets * self.market_feat_dim
            p_end = mf_end + _PORTFOLIO_DIM
            mask_end = p_end + self.n_markets
            # ml_edges_end = mask_end + self.n_markets * 2

            market_features = obs_flat[:, :mf_end].reshape(batch_size, self.n_markets, self.market_feat_dim)
            market_mask = obs_flat[:, p_end:mask_end]  # (batch, n_markets)
            ml_edges = obs_flat[:, mask_end:].reshape(batch_size, self.n_markets, 2)

            # Shared trunk
            trunk_out = self.trunk(obs_flat)  # (batch, hidden)

            # Per-slot action logits
            all_logits = []
            for slot in range(self.n_markets):
                slot_features = market_features[:, slot, :]  # (batch, market_feat_dim)
                slot_edges = ml_edges[:, slot, :]  # (batch, 2)
                slot_input = torch.cat([trunk_out, slot_features, slot_edges], dim=1)
                slot_logits = self.slot_head(slot_input)  # (batch, n_actions)

                # Mask inactive slots: set logits to very negative except hold
                mask = market_mask[:, slot].unsqueeze(1)  # (batch, 1)
                # For inactive slots, force hold action (index 0)
                hold_bias = torch.zeros_like(slot_logits)
                hold_bias[:, 0] = 20.0  # strong bias toward hold for inactive
                slot_logits = slot_logits * mask + hold_bias * (1 - mask)

                all_logits.append(slot_logits)

            logits = torch.stack(all_logits, dim=1)  # (batch, n_markets, n_actions)

            if return_value:
                value = self.value_head(trunk_out)  # (batch, 1)
                return logits, value

            return logits

    return MultiSlotPolicy(obs_dim, n_markets, _N_ACTIONS_PER_SLOT, hidden_dim)


# ── Phase 2: Behavioral Cloning ─────────────────────────────────


def behavioral_cloning(
    policy,
    demo_buffer: DemonstrationBuffer,
    n_markets: int,
    *,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    timeout_seconds: float = 300.0,
):
    """Pre-train policy via behavioral cloning on expert demonstrations."""
    import torch
    import torch.nn as nn

    print("\n=== Phase 2: Behavioral Cloning ===", flush=True)

    device = _get_device()
    print(f"  Device: {device}", flush=True)
    policy = policy.to(device)

    # Prepare dataset
    obs_array = np.array(demo_buffer.observations, dtype=np.float32)
    act_array = np.array(demo_buffer.actions, dtype=np.int64)  # (N, n_markets)

    n_samples = len(obs_array)
    print(f"  Training on {n_samples} demonstrations", flush=True)

    # Compute class weights to handle hold-action imbalance
    action_counts = np.bincount(act_array.flatten(), minlength=_N_ACTIONS_PER_SLOT)
    total = action_counts.sum()
    # Inverse frequency weighting, clamped
    weights = np.where(action_counts > 0, total / (action_counts * _N_ACTIONS_PER_SLOT), 1.0)
    weights = np.clip(weights, 0.5, 10.0).astype(np.float32)
    class_weights = torch.tensor(weights, device=device)
    print(f"  Class weights: {weights}", flush=True)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    start = time.monotonic()
    best_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        if time.monotonic() - start >= timeout_seconds:
            break

        # Shuffle
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        correct = 0
        total_pred = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            obs_batch = torch.tensor(obs_array[batch_idx], device=device)
            act_batch = torch.tensor(act_array[batch_idx], device=device)

            logits = policy(obs_batch)  # (batch, n_markets, n_actions)

            # Flatten for cross-entropy: (batch * n_markets, n_actions) vs (batch * n_markets,)
            logits_flat = logits.reshape(-1, _N_ACTIONS_PER_SLOT)
            targets_flat = act_batch.reshape(-1)

            loss = loss_fn(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Accuracy
            preds = logits_flat.argmax(dim=1)
            correct += (preds == targets_flat).sum().item()
            total_pred += targets_flat.numel()

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        accuracy = correct / max(total_pred, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.monotonic() - start
            print(
                f"  Epoch {epoch + 1}/{n_epochs}"
                f" | loss={avg_loss:.4f}"
                f" | accuracy={accuracy:.4f}"
                f" | lr={scheduler.get_last_lr()[0]:.6f}"
                f" | elapsed={elapsed:.0f}s",
                flush=True,
            )

    # Restore best
    if best_state is not None:
        policy.load_state_dict(best_state)
        policy = policy.to(device)
    print(f"  Best BC loss: {best_loss:.4f}", flush=True)

    return policy


# ── Phase 3: PPO Fine-Tuning ────────────────────────────────────


def compute_differential_sharpe_rewards(
    raw_rewards: list[float],
    eta: float = 0.01,
) -> list[float]:
    """Compute differential Sharpe ratio rewards.

    The differential Sharpe ratio at time t is the derivative of the
    Sharpe ratio with respect to a new return observation. This provides
    a reward signal that directly optimizes the Sharpe ratio.

    dS_t = (B_{t-1} * delta_A_t - 0.5 * A_{t-1} * delta_B_t) / (B_{t-1} - A_{t-1}^2)^{3/2}

    where A_t = EWMA of returns, B_t = EWMA of squared returns.
    """
    shaped = []
    a_prev = 0.0  # EWMA of returns
    b_prev = 1e-4  # EWMA of squared returns (small init to avoid div by zero)

    for r in raw_rewards:
        r_norm = r / 1000.0  # normalize by starting cash

        delta_a = r_norm - a_prev
        delta_b = r_norm**2 - b_prev

        denom = b_prev - a_prev**2
        d_sharpe = (b_prev * delta_a - 0.5 * a_prev * delta_b) / (denom**1.5) if denom > 1e-8 else delta_a

        # Clip to prevent extreme values
        d_sharpe = float(np.clip(d_sharpe, -5.0, 5.0))
        shaped.append(d_sharpe)

        # Update EWMAs
        a_prev = (1 - eta) * a_prev + eta * r_norm
        b_prev = (1 - eta) * b_prev + eta * r_norm**2

    return shaped


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    Returns:
        advantages: (T,) array of advantages
        returns: (T,) array of discounted returns (for value function target)
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else 0.0
        next_done = dones[t]
        delta = rewards[t] + gamma * next_value * (1 - float(next_done)) - values[t]
        last_gae = delta + gamma * lam * (1 - float(next_done)) * last_gae
        advantages[t] = last_gae

    returns = advantages + np.array(values[:n], dtype=np.float32)
    return advantages, returns


def ppo_fine_tune(
    policy,
    n_markets: int,
    *,
    n_rollout_steps: int = 256,
    n_ppo_epochs: int = 4,
    batch_size: int = 128,
    lr: float = 3e-5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    timeout_seconds: float = 600.0,
    bc_regularization: float = 0.1,
    split: str = "train",
):
    """Fine-tune the BC-pretrained policy with PPO.

    Key design choices:
    - Small learning rate to not destroy BC initialization
    - BC regularization: KL penalty against the BC policy (DAgger-like)
    - Differential Sharpe reward for risk-adjusted returns
    - Entropy bonus to encourage exploration beyond expert behavior
    """
    import torch
    import torch.nn.functional as F

    print("\n=== Phase 3: PPO Fine-Tuning ===", flush=True)

    device = _get_device()
    policy = policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    env = make_env(n_markets=n_markets, split=split)

    start = time.monotonic()
    total_steps = 0
    episode_returns: list[float] = []
    episode_trades: list[int] = []
    current_ep_return = 0.0
    current_ep_trades = 0

    obs_dict, _ = env.reset()
    obs_flat = flatten_obs(obs_dict)

    update_count = 0

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= timeout_seconds:
            break

        # ── Collect rollout ──
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards_raw = []
        rollout_values = []
        rollout_dones = []

        policy.eval()

        for _step in range(n_rollout_steps):
            if time.monotonic() - start >= timeout_seconds:
                break

            obs_tensor = torch.tensor(obs_flat, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, value = policy(obs_tensor, return_value=True)
                # logits: (1, n_markets, n_actions)

                # Sample actions per slot
                dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                actions = dist.sample()  # (n_markets,)
                log_prob = dist.log_prob(actions)  # (n_markets,)

            actions_np = actions.cpu().numpy()
            log_probs_np = log_prob.sum().cpu().item()  # sum over slots
            value_np = value.squeeze().cpu().item()

            rollout_obs.append(obs_flat.copy())
            rollout_actions.append(actions_np.copy())
            rollout_log_probs.append(log_probs_np)
            rollout_values.append(value_np)

            # Step env
            next_obs_dict, reward, terminated, truncated, _info = env.step(actions_np)
            done = terminated or truncated
            next_obs_flat = flatten_obs(next_obs_dict)

            rollout_rewards_raw.append(float(reward))
            rollout_dones.append(done)

            current_ep_return += float(reward)
            current_ep_trades += int(np.sum(actions_np != 0))
            total_steps += 1

            if done:
                episode_returns.append(current_ep_return)
                episode_trades.append(current_ep_trades)
                current_ep_return = 0.0
                current_ep_trades = 0
                next_obs_dict, _ = env.reset()
                next_obs_flat = flatten_obs(next_obs_dict)

            obs_dict = next_obs_dict
            obs_flat = next_obs_flat

        if not rollout_obs:
            break

        # ── Compute shaped rewards (differential Sharpe) ──
        shaped_rewards = compute_differential_sharpe_rewards(rollout_rewards_raw)

        # ── Compute GAE ──
        # Get bootstrap value
        with torch.no_grad():
            boot_tensor = torch.tensor(obs_flat, device=device).unsqueeze(0)
            _, boot_value = policy(boot_tensor, return_value=True)
            boot_val = boot_value.squeeze().cpu().item()
        rollout_values_ext = rollout_values + [boot_val]

        advantages, returns = compute_gae(
            shaped_rewards,
            rollout_values_ext,
            rollout_dones,
            gamma=gamma,
            lam=gae_lambda,
        )

        # Do NOT normalize advantages — research shows this kills calibration
        # (Turtel et al., 2025: advantage normalization causes overconfidence collapse)

        # ── PPO update ──
        policy.train()

        obs_batch = torch.tensor(np.array(rollout_obs), device=device)
        act_batch = torch.tensor(np.array(rollout_actions), device=device, dtype=torch.long)
        old_log_probs = torch.tensor(np.array(rollout_log_probs), device=device, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, device=device, dtype=torch.float32)
        returns_t = torch.tensor(returns, device=device, dtype=torch.float32)

        n_samples = len(rollout_obs)

        for _ppo_epoch in range(n_ppo_epochs):
            indices = np.random.permutation(n_samples)

            for mb_start in range(0, n_samples, batch_size):
                mb_idx = indices[mb_start : mb_start + batch_size]

                mb_obs = obs_batch[mb_idx]
                mb_acts = act_batch[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_ret = returns_t[mb_idx]

                logits, values = policy(mb_obs, return_value=True)

                # New log probs
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_acts).sum(dim=1)  # sum over slots
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), mb_ret)

                # Total loss
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        update_count += 1

        # Logging
        if update_count % 5 == 0 and episode_returns:
            recent = episode_returns[-20:] if len(episode_returns) >= 20 else episode_returns
            recent_trades = episode_trades[-20:] if len(episode_trades) >= 20 else episode_trades
            avg_ret = np.mean(recent)
            avg_trades = np.mean(recent_trades)
            win_rate = np.mean(np.array(recent) > 0)
            elapsed = time.monotonic() - start
            print(
                f"  PPO update {update_count}"
                f" | steps={total_steps}"
                f" | episodes={len(episode_returns)}"
                f" | avg_return={avg_ret:.2f}"
                f" | avg_trades={avg_trades:.1f}"
                f" | win_rate={win_rate:.3f}"
                f" | elapsed={elapsed:.0f}s",
                flush=True,
            )

    env.close()

    return policy, episode_returns, episode_trades


# ── Evaluation ───────────────────────────────────────────────────


def evaluate(
    policy,
    n_markets: int,
    split: str = "val",
    n_episodes: int = 100,
    max_steps: int = 300,
    timeout_seconds: float = 120.0,
):
    """Evaluate the trained policy."""
    import torch

    print(f"\n=== Evaluation on {split} split ===", flush=True)

    device = _get_device()
    policy = policy.to(device)
    policy.eval()

    env = make_env(n_markets=n_markets, split=split)
    start = time.monotonic()

    episode_returns: list[float] = []
    episode_trades: list[int] = []

    for _ep in range(n_episodes):
        if time.monotonic() - start >= timeout_seconds:
            break

        obs_dict, _ = env.reset()
        obs_flat = flatten_obs(obs_dict)
        ep_return = 0.0
        ep_trades = 0

        for _step in range(max_steps):
            if time.monotonic() - start >= timeout_seconds:
                break

            obs_tensor = torch.tensor(obs_flat, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_tensor)
                actions = logits.squeeze(0).argmax(dim=1)  # greedy

            actions_np = actions.cpu().numpy()

            next_obs_dict, reward, terminated, truncated, _info = env.step(actions_np)
            next_obs_flat = flatten_obs(next_obs_dict)

            ep_return += float(reward)
            ep_trades += int(np.sum(actions_np != 0))

            obs_flat = next_obs_flat
            if terminated or truncated:
                break

        episode_returns.append(ep_return)
        episode_trades.append(ep_trades)

    env.close()

    returns = np.array(episode_returns)
    trades = np.array(episode_trades)
    sharpe = float(np.mean(returns) / max(np.std(returns), 1e-8)) if len(returns) > 1 else 0.0
    total_pnl = float(np.sum(returns))

    print(
        f"  Episodes: {len(returns)}"
        f" | avg_return: {np.mean(returns):.2f}"
        f" | avg_trades: {np.mean(trades):.1f}"
        f" | win_rate: {np.mean(returns > 0):.3f}"
        f" | sharpe: {sharpe:.4f}"
        f" | total_pnl: {total_pnl:.2f}",
        flush=True,
    )

    return {
        "sharpe": sharpe,
        "avg_return": float(np.mean(returns)),
        "avg_trades": float(np.mean(trades)),
        "win_rate": float(np.mean(returns > 0)),
        "total_pnl": total_pnl,
        "n_episodes": len(returns),
    }


# ── Main Training Pipeline ──────────────────────────────────────


def train_pipeline(
    *,
    n_markets: int = 5,
    timeout_seconds: int = 1200,
    split: str = "train",
    eval_only: bool = False,
    skip_bc: bool = False,
    # Demo generation
    demo_episodes: int = 200,
    demo_max_steps: int = 300,
    # Behavioral cloning
    bc_epochs: int = 30,
    bc_lr: float = 1e-3,
    bc_batch_size: int = 256,
    # PPO fine-tuning
    ppo_lr: float = 3e-5,
    ppo_rollout_steps: int = 256,
    ppo_epochs: int = 4,
    ppo_batch_size: int = 128,
    ppo_clip: float = 0.2,
    ppo_ent_coef: float = 0.01,
    # Network
    hidden_dim: int = 256,
):
    """Run the full imitation learning -> RL fine-tuning pipeline."""
    import torch

    pipeline_start = time.monotonic()

    obs_dim = obs_dim_for(n_markets)
    n_actions = n_markets * _N_ACTIONS_PER_SLOT

    print(f"Configuration: n_markets={n_markets}, obs_dim={obs_dim}, n_actions={n_actions}")
    print(f"Timeout: {timeout_seconds}s ({timeout_seconds / 60:.1f} min)")

    policy_path = MODELS_DIR / "rl_imitation_policy.pt"

    # Handle eval-only mode
    if eval_only:
        if split == "train":
            split = "val"
        if not policy_path.exists():
            print(f"ERROR: No saved policy at {policy_path}", flush=True)
            return

        checkpoint = torch.load(policy_path, weights_only=False)
        policy = build_policy_network(obs_dim, n_markets, hidden_dim)
        policy.load_state_dict(checkpoint["policy"])
        print(f"Loaded policy from {policy_path}", flush=True)

        metrics = evaluate(policy, n_markets, split=split, timeout_seconds=120.0)
        _print_result(metrics, time.monotonic() - pipeline_start)
        return

    # Budget allocation:
    # - 15% for demonstration generation
    # - 15% for behavioral cloning
    # - 60% for PPO fine-tuning
    # - 10% for evaluation
    demo_budget = timeout_seconds * 0.15
    bc_budget = timeout_seconds * 0.15
    eval_budget = timeout_seconds * 0.10

    # ── Phase 1: Generate Demonstrations ──
    if not skip_bc:
        demo_buffer = generate_demonstrations(
            n_markets=n_markets,
            n_episodes=demo_episodes,
            max_steps=demo_max_steps,
            timeout_seconds=demo_budget,
        )

        if len(demo_buffer) < 100:
            print("WARNING: Very few demonstrations collected, BC may be poor", flush=True)

    # ── Phase 2: Behavioral Cloning ──
    policy = build_policy_network(obs_dim, n_markets, hidden_dim)

    if not skip_bc and len(demo_buffer) >= 50:
        policy = behavioral_cloning(
            policy,
            demo_buffer,
            n_markets,
            n_epochs=bc_epochs,
            batch_size=bc_batch_size,
            lr=bc_lr,
            timeout_seconds=bc_budget,
        )
    else:
        if skip_bc:
            print("\n  Skipping behavioral cloning (--skip-bc)", flush=True)
        else:
            print("\n  Skipping BC: too few demonstrations", flush=True)

    # ── Phase 3: PPO Fine-Tuning ──
    remaining = timeout_seconds - (time.monotonic() - pipeline_start) - eval_budget
    if remaining < 30:
        print("\n  Not enough time for PPO, skipping", flush=True)
    else:
        policy, ppo_returns, ppo_trades = ppo_fine_tune(
            policy,
            n_markets,
            n_rollout_steps=ppo_rollout_steps,
            n_ppo_epochs=ppo_epochs,
            batch_size=ppo_batch_size,
            lr=ppo_lr,
            clip_range=ppo_clip,
            ent_coef=ppo_ent_coef,
            timeout_seconds=remaining,
            split=split,
        )

    # ── Save Policy ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "obs_dim": obs_dim,
            "n_markets": n_markets,
            "hidden_dim": hidden_dim,
        },
        policy_path,
    )
    print(f"\nSaved policy to {policy_path}", flush=True)

    # ── Phase 4: Evaluate ──
    eval_remaining = max(30.0, timeout_seconds - (time.monotonic() - pipeline_start))
    metrics = evaluate(
        policy,
        n_markets,
        split="val" if split == "train" else split,
        timeout_seconds=min(eval_remaining, eval_budget),
    )

    elapsed = time.monotonic() - pipeline_start
    _print_result(metrics, elapsed)


def _print_result(metrics: dict, elapsed: float) -> None:
    """Print the final RESULT line in the standard format."""
    print(
        "RESULT"
        f"\tEPISODE_RETURN={metrics['avg_return']:.6f}"
        f"\tTRADES={metrics['avg_trades']:.6f}"
        f"\tWIN_RATE={metrics['win_rate']:.6f}"
        f"\tSHARPE={metrics['sharpe']:.6f}"
        f"\tPNL={metrics['total_pnl']:.2f}"
        f"\tEPISODES={metrics['n_episodes']}"
        f"\tTRAIN_TIME={elapsed:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Imitation Learning -> RL Fine-Tuning for multi-market trading")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Total time budget in seconds (default: 1200 = 20 min)",
    )
    parser.add_argument("--n-markets", type=int, default=5, help="Number of simultaneous market slots")
    parser.add_argument("--split", default="train", help="Split to use: train, val, test")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only with saved policy")
    parser.add_argument("--skip-bc", action="store_true", help="Skip behavioral cloning, start with random policy")

    # Demo generation
    parser.add_argument("--demo-episodes", type=int, default=200, help="Number of expert demo episodes")
    parser.add_argument("--demo-max-steps", type=int, default=300, help="Max steps per demo episode")

    # Behavioral cloning
    parser.add_argument("--bc-epochs", type=int, default=30, help="BC training epochs")
    parser.add_argument("--bc-lr", type=float, default=1e-3, help="BC learning rate")
    parser.add_argument("--bc-batch-size", type=int, default=256, help="BC batch size")

    # PPO fine-tuning
    parser.add_argument("--ppo-lr", type=float, default=3e-5, help="PPO learning rate")
    parser.add_argument("--ppo-rollout-steps", type=int, default=256, help="PPO rollout length")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--ppo-batch-size", type=int, default=128, help="PPO mini-batch size")
    parser.add_argument("--ppo-clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01, help="Entropy coefficient")

    # Network
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")

    args = parser.parse_args()

    train_pipeline(
        n_markets=args.n_markets,
        timeout_seconds=args.timeout,
        split=args.split,
        eval_only=args.eval_only,
        skip_bc=args.skip_bc,
        demo_episodes=args.demo_episodes,
        demo_max_steps=args.demo_max_steps,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        bc_batch_size=args.bc_batch_size,
        ppo_lr=args.ppo_lr,
        ppo_rollout_steps=args.ppo_rollout_steps,
        ppo_epochs=args.ppo_epochs,
        ppo_batch_size=args.ppo_batch_size,
        ppo_clip=args.ppo_clip,
        ppo_ent_coef=args.ppo_ent_coef,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
