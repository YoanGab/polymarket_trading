# autotrader ML+RL — Autoresearch Program

This is the research memo for autonomous ML and RL experimentation on the Polymarket trading bot.

## Setup

To set up a new experiment run:

1. **Agree on a run tag** (e.g. `ml-rl-mar15`). Create branch `autotrader/<tag>`.
2. **Read these files** for full context:
   - `program_ml_rl.md` — this file. The ground truth.
   - `src/polymarket_backtest/trading_env.py` — the universal trading environment
   - `src/polymarket_backtest/gym_env.py` — Gymnasium wrapper for RL
   - `src/polymarket_backtest/features.py` — feature extraction (ML features)
   - `scripts/train_model.py` — ML model training
   - `src/polymarket_backtest/ml_transport.py` — how the ML model makes predictions
   - `src/polymarket_backtest/strategies.py` — strategy logic
   - `src/polymarket_backtest/grid_search.py` — strategy grid + eval
3. **Verify data**: `ls -la data/polymarket_backtest_v2.sqlite` (~19GB)
4. **Verify prepared data**: `ls data/prepared/` (train.npz, val.npz, test.npz, meta.json)
5. **Run baseline ML**: `uv run python scripts/train_model.py --model xgboost > run.log 2>&1`
6. **Run baseline strategy eval**: `uv run python scripts/eval_strategies.py --forecast-mode ml_model --max-markets 100 --split val > eval.log 2>&1`
7. **Record baselines** in results.tsv and results_rl.tsv
8. **Go.**

## Architecture

```
ML SCREENER (XGBoost)          RL AGENT (policy network)
  Scans all markets              Sees top opportunities
  Predicts probability_yes       Decides: what, how much, when
  Fast (~1ms per market)         Uses TradingEnvironment
         │                               │
         └──────── both feed into ───────┘
                        │
              TradingEnvironment
              (state/action/reward)
```

The ML model and RL agent are **complementary**, not competing:
- **ML** predicts WHAT will happen (probability)
- **RL** learns HOW to trade (timing, sizing, exits)

## What you CAN edit

### ML track
- `src/polymarket_backtest/features.py` — feature extraction, new features
- `scripts/train_model.py` — model architecture, hyperparameters, calibration
- `src/polymarket_backtest/ml_transport.py` — how predictions are made at inference

### RL track
- `src/polymarket_backtest/gym_env.py` — observation space, action space, reward
- `src/polymarket_backtest/trading_env.py` — ONLY the observation/reward parts (NOT the simulation logic)
- New files: `scripts/train_rl.py`, `src/polymarket_backtest/rl_agent.py`

### Strategy track
- `src/polymarket_backtest/strategies.py` — strategy logic
- `src/polymarket_backtest/grid_search.py` — strategy grid parameters

## What you CANNOT edit

- `src/polymarket_backtest/replay_engine.py` — simulation engine (fixed)
- `src/polymarket_backtest/market_simulator.py` — order execution (fixed)
- `src/polymarket_backtest/db.py` — database layer (fixed)
- `src/polymarket_backtest/types.py` — type definitions (fixed)
- `src/polymarket_backtest/metrics.py` — metric calculations (fixed)
- `src/polymarket_backtest/production_guards.py` — safety guards (fixed)
- `src/polymarket_backtest/splits.py` — train/val/test definitions (fixed)
- `scripts/prepare.py` — dataset preparation (fixed)
- `data/` — read-only

## Metrics (in order of importance)

### ML metrics
1. **Val Brier** — lower is better (model prediction quality)
2. **Calibration ECE** — lower is better (prediction reliability)

### Trading metrics (both ML and RL)
1. **Sharpe ratio** — must be positive, higher is better. Computed on val split.
2. **Trade count** — must be > 0. Target: 100+ trades.
3. **PnL** — must be positive. Higher is better.
4. **Max drawdown** — lower is better. Target: < 20%.

### RL-specific metrics
1. **Episode return** — average cumulative reward per episode
2. **Win rate** — % of episodes with positive total reward
3. **Capital utilization** — % of cash actually deployed

**Sharpe > 2.0 likely indicates a bug, not a breakthrough.**

A Sharpe of 0.5-1.5 is realistic and good. Focus on consistency across seeds.

## Data

```
data/polymarket_backtest_v2.sqlite (READ ONLY, ~19 GB)
```

- **181,275 resolved binary markets** (YES/NO)
- **68,219,420 hourly price snapshots**
- **Time range**: Nov 2022 to Mar 2026
- **Split**: train < 2025-10-01, val Oct-Dec 2025, test >= 2026-01-01, holdout >= 2026-02-01

The TradingEnvironment loads markets from this DB. The Gym wrapper provides observation/action/reward.

## The Experiment Loop

There are THREE experiment types. Each follows the same commit-run-keep/revert loop.

### ML Experiment

```bash
# Edit features.py or train_model.py
git commit -m "feat: description of change"
uv run python scripts/train_model.py --model xgboost > run.log 2>&1
grep "RESULT\|TEST_BRIER" run.log
# Then run strategy eval to see trading impact:
uv run python scripts/eval_strategies.py --forecast-mode ml_model --max-markets 100 --split val > eval.log 2>&1
grep "BEST_STRATEGY\|SHARPE\|PNL\|TRADES" eval.log
```

### RL Experiment

```bash
# Edit gym_env.py or scripts/train_rl.py
git commit -m "feat: description of change"
uv run python scripts/train_rl.py > run.log 2>&1
grep "EPISODE_RETURN\|SHARPE\|TRADES\|DRAWDOWN" run.log
```

### Strategy Experiment

```bash
# Edit strategies.py or grid_search.py
git commit -m "feat: description of change"
uv run python scripts/eval_strategies.py --forecast-mode ml_model --max-markets 100 --split val > eval.log 2>&1
grep "BEST_STRATEGY\|SHARPE\|PNL\|TRADES" eval.log
```

### The Loop (applies to ALL experiment types)

LOOP FOREVER:

1. Read results.tsv / results_rl.tsv. Don't repeat failures.
2. Pick ONE experimental idea. ONE change per experiment.
3. `git commit` the change.
4. Run the experiment (see commands above).
5. Read out results from run.log / eval.log.
6. If empty output → crash. Read `tail -n 50 run.log`. Fix if trivial, log crash if fundamental.
7. Record in results.tsv (ML/strategy) or results_rl.tsv (RL).
8. If improved → **keep** the commit (advance the branch).
9. If equal or worse → `git reset --hard HEAD~1` to revert. Then retrain if needed.
10. Every 5 experiments → re-read `program_ml_rl.md` from disk.
11. Every 5 experiments → run multi-seed robustness check.
12. Go to step 1.

## Logging Results

### results.tsv (ML + strategy experiments)

```
commit	val_brier	sharpe	pnl	trades	status	description
```

### results_rl.tsv (RL experiments)

```
commit	episode_return	sharpe	pnl	trades	max_drawdown	status	description
```

Status: `keep`, `discard`, or `crash`.
Do NOT commit results files — leave untracked.

## RL Research Guide

### Getting Started

The RL infrastructure is ready:

```python
from polymarket_backtest.gym_env import PolymarketGymEnv

env = PolymarketGymEnv(
    db_path='data/polymarket_backtest_v2.sqlite',
    starting_cash=1000,
    split='val',
    max_markets=100,
)

obs, info = env.reset()           # shape: (31,) float32
obs, reward, done, trunc, info = env.step(action)  # action: 0-11
```

### Observation Space

31-dimensional vector from `TradingState.to_array()`:
- Market features: bid, ask, mid, spread, volume, momentum (3h-24h), volatility
- ML prediction: probability_yes, confidence, edge_bps
- Portfolio: cash_pct, total_invested_pct, n_positions, unrealized_pnl
- Time: hours_to_resolution

### Action Space

Discrete(12):
```
0: hold           6: buy_yes_limit
1: buy_yes        7: buy_no_limit
2: buy_no         8: sell_yes_limit
3: sell_yes       9: sell_no_limit
4: sell_no       10: mint_pair
5: cancel        11: redeem_pair
```

### Reward

`reward = Δ portfolio_value` at each step (dense, immediate).
`portfolio_value = cash + sum(position_qty * mark_price)`

### Episode Structure

One episode = one market, from first snapshot to resolution (or max 500 steps).
Reset loads a new random market from the val split.

### Algorithms to Try (in order)

1. **DQN** (start here) — simple, discrete actions, replay buffer
2. **Rainbow DQN** — better: double, dueling, prioritized replay, n-step
3. **PPO** — if DQN plateaus, try on-policy
4. **Decision Transformer** — offline RL, treat trajectories as sequences

### RL Ideas to Explore

- [ ] Basic DQN with default obs/reward
- [ ] Reward shaping: penalize churn (too many trades), penalize drawdown
- [ ] Action masking: disable buy when no cash, disable sell when no position
- [ ] Curriculum learning: start with short markets (< 7 days), then longer
- [ ] Multi-market episodes: rotate markets within one episode
- [ ] Add ML prediction to observation (the RL uses the ML screener's signal)
- [ ] Feature engineering: add raw price history as extra obs channels
- [ ] Larger networks: 256-512 hidden units, 3+ layers
- [ ] Prioritized replay buffer biased toward non-hold actions
- [ ] Warm-start from imitation of the best hand-crafted strategy
- [ ] Per-category RL models (political vs crypto vs sports)

## ML Research Guide

### Current Model

XGBoost + isotonic calibration, 27 features. Beats mid-price by ~7%.

### ML Ideas to Explore

- [ ] Add tag features (50 one-hot category tags) — infrastructure ready
- [ ] Better calibration: Platt scaling, temperature, Venn-ABERS
- [ ] Feature importance / SHAP analysis → drop dead features
- [ ] LightGBM / CatBoost comparison
- [ ] Neural net (MLP, TabNet) on tabular features
- [ ] Time-series model (LSTM/Transformer on raw price history)
- [ ] Per-category specialist models
- [ ] Ensemble: blend XGBoost + neural net
- [ ] Train on resolution_proximity (using scheduled_close_ts, not actual)
- [ ] More lookback features (72h, 168h, 720h momentum — infrastructure ready)

## Simplicity Criterion

All else being equal, simpler is better:
- A 0.001 improvement + 20 lines of hacky code? Probably not worth it.
- A 0.001 improvement from deleting code? Always keep.
- Same performance but much simpler code? Keep.

## Timeout

- ML experiments: ~5 minutes (train) + ~2 minutes (eval) = ~7 min total
- RL experiments: ~10-15 minutes (training loop)
- Strategy experiments: ~2 minutes (eval only)
- If anything exceeds 20 minutes, kill it and log crash.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. The human may be asleep. Continue working autonomously until manually stopped. If you run out of ideas, re-read this file, inspect failures, try combining near-misses, try more radical approaches. The loop runs until interrupted.

## CURRENT PRIORITY

1. **Establish corrected ML baseline** with all new features (27 features + 50 tags + multi-scale momentum)
2. **Build basic RL agent** (DQN) that beats the hand-crafted strategy
3. **Compare ML vs RL vs ML+RL** on the same val markets
4. Target: Sharpe > 0.5 with 100+ trades, consistent across 3+ seeds
5. Do NOT deploy to production until 500+ trades over 6+ months of out-of-sample data
