"""RL-based trading agent using REINFORCE policy gradient.

Instead of fixed strategy rules, learns WHEN and HOW MUCH to trade
based on model predictions and market state.

The RL agent receives:
  - State: [model_prediction, mid, spread, volatility, edge_bps, ...]
  - Outputs: action (trade_fraction: 0=don't trade, positive=buy YES)

Trains on historical outcomes using REINFORCE with baseline.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_backtest.features import (
    build_dataset,
    extract_snapshot_features,
    walk_forward_split,
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest.sqlite"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class TradingPolicy(nn.Module):
    """Simple policy network: state → action distribution."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Output: mean and log_std for position sizing (Gaussian policy)
        self.mean_head = nn.Linear(32, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean = torch.sigmoid(self.mean_head(h))  # position fraction [0, 1]
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def sample_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return torch.clamp(action, 0.0, 0.2), log_prob  # max 20% position

    def get_action_deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self(state)
        return mean


def load_ml_model():
    """Load the trained ML model for predictions."""
    model_path = MODELS_DIR / "logistic_model.pkl"
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_names"]


def predict_ml(model: dict, X: np.ndarray) -> float:
    """Get ML model prediction for a single sample."""
    X_scaled = model["scaler"].transform(X.reshape(1, -1))
    prob = model["model"].predict_proba(X_scaled)[0, 1]
    return float(np.clip(prob, 0.001, 0.999))


def build_trading_episodes(db_path: str | Path) -> list[dict]:
    """Build trading episodes from the database.

    Each episode is a market with snapshots. The agent sees each snapshot
    and decides how much to trade. At resolution, PnL is computed.
    """
    from polymarket_backtest import db as dbmod

    conn = dbmod.connect(db_path)
    market_ids = dbmod.get_market_ids(conn)

    ml_model, feature_names = load_ml_model()

    episodes = []
    for market_id in market_ids:
        resolution = dbmod.get_resolution(conn, market_id)
        if resolution is None:
            continue
        outcome = float(resolution["resolved_outcome"])

        rows = conn.execute(
            """
            SELECT s.*, m.resolution_ts
            FROM market_snapshots s
            JOIN markets m ON m.market_id = s.market_id
            WHERE s.market_id = ?
            ORDER BY s.ts ASC
            """,
            (market_id,),
        ).fetchall()

        if len(rows) < 10:
            continue

        # Use every 6th snapshot
        sampled = rows[::6]
        if sampled[-1] != rows[-1]:
            sampled.append(rows[-1])

        snapshots = []
        for i, row in enumerate(sampled):
            prev_rows = sampled[max(0, i - 24) : i]
            features = extract_snapshot_features(row, prev_rows)
            feature_vector = np.array([features.get(name, 0.0) for name in feature_names], dtype=np.float32)

            mid = float(row["mid"])
            best_ask = float(row["best_ask"])
            spread = float(row["best_ask"]) - float(row["best_bid"])

            # Get ML prediction
            prediction = predict_ml(ml_model, feature_vector)
            edge_bps = (prediction - best_ask) * 10000.0

            # Build RL state
            state = np.array(
                [
                    prediction,
                    mid,
                    spread,
                    edge_bps / 1000.0,  # normalize
                    abs(prediction - mid),  # confidence
                    features.get("volatility_24h", 0.0),
                    features.get("volume_oi_ratio", 0.0),
                    features.get("resolution_proximity", 0.0),
                    features.get("price_vs_half", 0.0),
                    features.get("momentum_24h_pct", 0.0),
                ],
                dtype=np.float32,
            )

            snapshots.append(
                {
                    "state": state,
                    "mid": mid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "prediction": prediction,
                    "edge_bps": edge_bps,
                }
            )

        episodes.append(
            {
                "market_id": market_id,
                "outcome": outcome,
                "snapshots": snapshots,
                "last_ts": str(rows[-1]["ts"]),
            }
        )

    conn.close()
    return episodes


def compute_trade_pnl(
    action: float,
    best_ask: float,
    outcome: float,
    max_notional: float = 100.0,
) -> float:
    """Compute PnL from a single trade decision.

    action: fraction of capital to deploy (0-0.2)
    best_ask: price to buy YES
    outcome: 0 or 1 (resolution)
    """
    if action < 0.01:  # don't trade threshold
        return 0.0

    notional = action * max_notional
    qty = notional / best_ask  # YES shares bought
    cost = notional
    payout = qty * outcome  # $1 per share if outcome=1, $0 if outcome=0
    return payout - cost


def train_rl(episodes: list[dict], n_epochs: int = 200) -> TradingPolicy:
    """Train RL policy using REINFORCE with baseline."""
    state_dim = 10
    policy = TradingPolicy(state_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    # Use last snapshot per market for training (closest to resolution)
    states = []
    outcomes = []
    asks = []
    for ep in episodes:
        last_snap = ep["snapshots"][-1]
        states.append(last_snap["state"])
        outcomes.append(ep["outcome"])
        asks.append(last_snap["best_ask"])

    states_t = torch.tensor(np.array(states), dtype=torch.float32)
    outcomes_np = np.array(outcomes)
    asks_np = np.array(asks)

    best_sharpe = -float("inf")
    best_state_dict = None

    for epoch in range(n_epochs):
        # Sample actions from policy
        actions, log_probs = policy.sample_action(states_t)
        actions_np = actions.squeeze().detach().numpy()

        # Compute PnL for each trade
        pnls = np.array([compute_trade_pnl(a, ask, out) for a, ask, out in zip(actions_np, asks_np, outcomes_np)])

        # Sharpe-like reward (penalize variance)
        mean_pnl = pnls.mean()
        std_pnl = pnls.std() + 1e-8
        sharpe = mean_pnl / std_pnl

        # Per-trade advantage (reward - baseline)
        baseline = mean_pnl
        advantages = pnls - baseline

        # REINFORCE loss
        advantages_t = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)
        policy_loss = -(log_probs * advantages_t).mean()

        # Entropy bonus for exploration
        mean_a, std_a = policy(states_t)
        entropy = 0.5 * torch.log(2 * np.pi * np.e * std_a**2).mean()
        loss = policy_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_state_dict = {k: v.clone() for k, v in policy.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            n_trades = (actions_np > 0.01).sum()
            total_pnl = pnls.sum()
            print(
                f"  Epoch {epoch + 1:3d}: PnL={total_pnl:+8.2f}, "
                f"Sharpe={sharpe:+.4f}, Trades={n_trades}/{len(actions_np)}, "
                f"MeanAction={actions_np.mean():.4f}"
            )

    policy.load_state_dict(best_state_dict)  # type: ignore[arg-type]
    policy.eval()
    return policy


def evaluate_policy(policy: TradingPolicy, episodes: list[dict]) -> dict:
    """Evaluate a trained policy on a set of episodes."""
    total_pnl = 0.0
    n_trades = 0
    pnls = []

    for ep in episodes:
        last_snap = ep["snapshots"][-1]
        state_t = torch.tensor(last_snap["state"], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = policy.get_action_deterministic(state_t).item()

        pnl = compute_trade_pnl(action, last_snap["best_ask"], ep["outcome"])
        pnls.append(pnl)
        total_pnl += pnl
        if action > 0.01:
            n_trades += 1

    pnls_arr = np.array(pnls)
    sharpe = pnls_arr.mean() / (pnls_arr.std() + 1e-8)

    return {
        "total_pnl": round(total_pnl, 2),
        "n_trades": n_trades,
        "n_markets": len(episodes),
        "sharpe": round(float(sharpe), 4),
        "mean_pnl_per_trade": round(total_pnl / max(n_trades, 1), 2),
    }


def main() -> None:
    print("Building trading episodes...")
    episodes = build_trading_episodes(DB_PATH)
    print(f"  Total episodes (markets): {len(episodes)}")

    # Split into train/test by time (same as walk-forward)
    episodes.sort(key=lambda e: e["last_ts"])
    n = len(episodes)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_eps = episodes[:train_end]
    val_eps = episodes[train_end:val_end]
    test_eps = episodes[val_end:]

    print(f"  Train: {len(train_eps)}, Val: {len(val_eps)}, Test: {len(test_eps)}")

    # Train RL policy
    print("\nTraining RL policy...")
    policy = train_rl(train_eps, n_epochs=300)

    # Evaluate
    print("\n" + "=" * 60)
    for name, eps in [("TRAIN", train_eps), ("VAL", val_eps), ("TEST", test_eps)]:
        result = evaluate_policy(policy, eps)
        print(
            f"  {name}: PnL={result['total_pnl']:+.2f}, Sharpe={result['sharpe']:+.4f}, Trades={result['n_trades']}/{result['n_markets']}"
        )

    # Compare with simple heuristic: trade when edge > 30bps
    print("\n--- Heuristic baseline (edge > 30bps) ---")
    for name, eps in [("TRAIN", train_eps), ("VAL", val_eps), ("TEST", test_eps)]:
        total_pnl = 0.0
        n_trades = 0
        pnls = []
        for ep in eps:
            last_snap = ep["snapshots"][-1]
            if last_snap["edge_bps"] > 30:
                pnl = compute_trade_pnl(0.05, last_snap["best_ask"], ep["outcome"])
                pnls.append(pnl)
                total_pnl += pnl
                n_trades += 1
            else:
                pnls.append(0.0)
        pnls_arr = np.array(pnls)
        sharpe = pnls_arr.mean() / (pnls_arr.std() + 1e-8) if len(pnls_arr) > 0 else 0.0
        print(f"  {name}: PnL={total_pnl:+.2f}, Sharpe={sharpe:+.4f}, Trades={n_trades}/{len(eps)}")

    # Save policy
    policy_path = MODELS_DIR / "rl_policy.pkl"
    torch.save({"policy": policy.state_dict(), "state_dim": 10}, policy_path)
    print(f"\nPolicy saved to: {policy_path}")


if __name__ == "__main__":
    main()
