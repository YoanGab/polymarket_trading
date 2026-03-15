# autotrader ML+RL — Autoresearch Program

Autonomous research on ML and RL approaches for the Polymarket trading bot.

## Goal

**Maximize Sharpe ratio on the val split with 100+ trades.** You choose the method. You choose the features. You choose the architecture. The only constraints are the data splits, the environment interface, and the experiment loop discipline.

## Data Splits (STRICT — no exceptions)

| Split | Markets | Period | Usage |
|-------|---------|--------|-------|
| **train** | ~63K | resolved < Oct 2025 | Train models, train RL agents |
| **val** | ~44K | Oct-Dec 2025 | Evaluate experiments, tune hyperparams |
| **test** | ~11K | Jan 2026 | Final evaluation only (never for tuning) |
| **holdout** | ~63K | Feb 2026+ | LOCKED. `--final-eval` required. NEVER use during research. |

Defined in `src/polymarket_backtest/splits.py`. Do NOT modify or bypass.

**Critical rule**: Train on train, evaluate on val. NEVER train on val data. NEVER peek at test until you have a final candidate.

## Setup

1. Create branch `autotrader/<tag>` from current HEAD.
2. Read this file and understand the environment.
3. Verify: `uv run pytest tests/ -q` → 122 tests pass.
4. Verify data: `ls -la data/polymarket_backtest_v2.sqlite` (~19GB).
5. Run baseline, record in results.tsv, and go.

## The Environment

The `TradingEnvironment` is your interface to the market simulation:

```python
from polymarket_backtest.trading_env import TradingEnvironment, Action
from polymarket_backtest.gym_env import PolymarketGymEnv  # for RL

# Agent sees: TradingState (31 features + raw data)
# Agent does: Action (12 types: buy/sell YES/NO, limits, mint/redeem, cancel, hold)
# Agent gets: StepResult (reward = Δ portfolio value, new state, done)
```

The environment exposes **raw data** — the agent decides what features to extract from it. Don't hardcode features; let the agent discover what works.

**Available actions**: hold, buy_yes, buy_no, sell_yes, sell_no, buy_yes_limit, buy_no_limit, sell_yes_limit, sell_no_limit, mint_pair, redeem_pair, cancel_orders.

**step_multi(actions)**: Execute multiple actions at the same timestamp.

## What you CAN edit

- `scripts/train_rl.py` — RL training loop, algorithm, architecture, reward shaping
- `scripts/train_model.py` — ML model, features, hyperparameters
- `src/polymarket_backtest/gym_env.py` — single-market gym env (Discrete(12))
- `src/polymarket_backtest/gym_env_multi.py` — multi-market gym env (N markets, shared cash, MultiDiscrete)
- `src/polymarket_backtest/features.py` — feature extraction
- `src/polymarket_backtest/ml_transport.py` — ML inference
- `src/polymarket_backtest/strategies.py` — hand-crafted strategy logic
- `src/polymarket_backtest/grid_search.py` — strategy configs
- New files: `scripts/train_*.py`, `src/polymarket_backtest/models/`

## What you CANNOT edit

- `src/polymarket_backtest/trading_env.py` — the environment simulation (fixed)
- `src/polymarket_backtest/replay_engine.py` — simulation engine (fixed)
- `src/polymarket_backtest/market_simulator.py` — order execution (fixed)
- `src/polymarket_backtest/types.py` — type definitions (fixed)
- `src/polymarket_backtest/metrics.py` — metric calculations (fixed)
- `src/polymarket_backtest/splits.py` — data split definitions (fixed)
- `src/polymarket_backtest/db.py` — database layer (fixed)
- `scripts/prepare.py` — dataset preparation (fixed)
- `data/` — read-only

## Training and Evaluation (STRICT separation)

### For ML experiments

```bash
# 1. TRAIN on train split
uv run python scripts/train_model.py --model xgboost > run.log 2>&1

# 2. EVALUATE on val split (ALL val markets, frozen model)
uv run python scripts/eval_strategies.py --forecast-mode ml_model --split val > eval.log 2>&1

# 3. Read metrics
grep "RESULT\|TEST_BRIER" run.log
grep "BEST_STRATEGY\|SHARPE\|PNL\|TRADES" eval.log
```

### For RL experiments

```bash
# 1. TRAIN on train markets (the agent plays and learns)
uv run python scripts/train_rl.py --split train > run.log 2>&1

# 2. EVALUATE on val markets (frozen policy, no learning)
uv run python scripts/train_rl.py --split val --eval-only > eval.log 2>&1

# 3. Read metrics
grep "RESULT\|EPISODE_RETURN\|SHARPE" run.log
grep "EVAL\|SHARPE\|TRADES" eval.log
```

**Gym environment**: `PolymarketMultiMarketGymEnv` — N markets simultaneous, shared cash, MultiDiscrete actions. 1700 episodes/min after 96s init. This is the realistic environment.

**Time budget**: Each experiment (train + eval) should complete within ~10 minutes wall clock. If it exceeds 15 minutes, kill it and log crash. The training script should stop by time, not by episode count.

**The RL agent must train on train markets and be evaluated on val markets.** The `--split` flag controls which markets are used. `--eval-only` freezes the policy (no gradient updates).

## The Experiment Loop

LOOP FOREVER:

1. Look at results.tsv. Don't repeat failures unless with a fundamentally different approach.
2. Come up with ONE experimental idea. You decide what to try. Be creative.
3. `git commit` the code change.
4. Run the experiment (train + eval, see commands above).
5. Read results.
6. If empty → crash. Read `tail -n 50 run.log`, fix if trivial, log crash if fundamental.
7. Record in results.tsv.
8. If improved (higher Sharpe) → **keep** the commit.
9. If equal or worse → `git reset --hard HEAD~1`.
10. Every 5 experiments → re-read this file from disk.
11. Go to step 1.

## Logging Results

### results.tsv

```
commit	val_brier	sharpe	pnl	trades	status	description
```

Status: `keep`, `discard`, or `crash`. Do NOT commit this file.

## Metrics (in priority order)

1. **Sharpe ratio on val** — must be positive. Higher is better.
2. **Trade count** — must be > 0. Target: 100+.
3. **PnL on val** — must be positive.
4. **Consistency** — must hold across 3+ random seeds.

**Sharpe > 2.0 likely indicates a bug.** A Sharpe of 0.5-1.5 is realistic.

## What NOT to do

- Do NOT train on val data. Ever.
- Do NOT peek at test or holdout during research.
- Do NOT bundle multiple changes in one experiment.
- Do NOT exceed 15 minutes per experiment.
- Do NOT hardcode strategies when you can learn them.
- Do NOT assume one approach is best — test and compare.

## Simplicity criterion

All else being equal, simpler is better. A tiny improvement with lots of complexity? Probably not worth it. Same performance with less code? Always keep.

## NEVER STOP

Once the loop begins, do NOT pause. The human may be asleep. Continue working autonomously until manually stopped. If you run out of ideas, re-read this file, analyze what features drive predictions, try combining near-misses, try radically different approaches. The loop runs until interrupted.

## Available infrastructure

- **181K markets** with 68M hourly snapshots
- **27 ML features** already computed (momentum, volatility, resolution proximity)
- **50 category tags** available as features
- **720h lookback** for temporal features
- **12 action types** in the environment
- **Two gym environments**: `PolymarketGymEnv` (single market, Discrete(12)) and `PolymarketMultiMarketGymEnv` (N markets simultaneous, MultiDiscrete, shared cash)
- **Cross-market correlation** via event_id (33K events)
- **Category routing** (political, crypto, sports — each with different dynamics and fees)
- **Production guards** (staleness, drawdown, distribution shift)
- **600x eval speedup** (stride + pre-load + skip audit)

## Hardware and acceleration

- **Apple Silicon Mac** with 128GB unified memory
- **MLX** (Apple's ML framework) is available for GPU-accelerated training on Apple Silicon. Consider using it when the neural network becomes the bottleneck (large models, Transformer/LSTM on long sequences). Library: `mlx` + `rlx` (CleanRL-inspired RL on MLX). Install: `uv add mlx`.
- **PyTorch MPS** backend also works on Apple Silicon (slightly slower than MLX for most workloads).
- **Current bottleneck** is data loading / feature computation, NOT the neural network. Focus on data pipeline speed first. Switch to MLX/GPU only when the model becomes the bottleneck.

## Research tools

- **Perplexity** (`mcp__perplexity-ask__perplexity_ask`): Use for researching best practices, library docs, algorithm comparisons. Example: "What is the best RL algorithm for financial trading?" or "How to implement prioritized experience replay in MLX?"
- **Firecrawl** (`mcp__Firecrawl__firecrawl_search`): Use for scraping documentation, GitHub repos, papers.
- **Context7** (`mcp__plugin_context7_context7__resolve-library-id`): Use for library-specific documentation (PyTorch, MLX, gymnasium, etc.)
- You are encouraged to research before implementing. Iterate — look up for new ideas, don't take what's on the research for granted, do your own experiments.
