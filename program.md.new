# autotrader

This is an experiment to have the LLM do its own research on Polymarket trading strategies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`). The branch `autotrader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `program.md` — this file. The ground truth.
   - `src/polymarket_backtest/features.py` — feature extraction, dataset building, splits.
   - `scripts/train_model.py` — model training, hyperparameters.
   - `src/polymarket_backtest/ml_transport.py` — how the model is used as a forecaster.
   - `src/polymarket_backtest/strategies.py` — strategy logic.
   - `src/polymarket_backtest/grid_search.py` — strategy grid parameters.
4. **Verify data exists**: `ls -la data/polymarket_backtest_v2.sqlite`
5. **Run the baseline**: `uv run python scripts/train_model.py --model logistic 2>&1 > run.log`
6. **Record baseline metrics**: Extract val_brier and test_brier from run.log.
7. **Run baseline strategy eval**: `uv run python scripts/eval_strategies.py --forecast-mode ml_model --max-markets 200 2>&1 > eval.log`
8. **Initialize results.tsv**: Create it with the header row. Record the baseline.
9. **Confirm and go.**

Once you get confirmation, kick off the experimentation.

## Experimentation

**What you CAN edit:**
- `src/polymarket_backtest/features.py` — Feature extraction, dataset building, splits
- `scripts/train_model.py` — Model training, hyperparameters, architectures, calibration
- `src/polymarket_backtest/ml_transport.py` — How the trained model is used as a forecaster
- `src/polymarket_backtest/strategies.py` — Strategy logic, entry/exit rules
- `src/polymarket_backtest/grok_replay.py` — SmartRuleTransport rules ONLY
- `src/polymarket_backtest/grid_search.py` — Strategy grid parameters
- You can create new files: `scripts/train_*.py`, `src/polymarket_backtest/models/`, helper scripts.

**What you CANNOT edit:**
- `src/polymarket_backtest/replay_engine.py` — Simulation engine (fixed)
- `src/polymarket_backtest/db.py` — Database layer (fixed)
- `src/polymarket_backtest/market_simulator.py` — Order execution (fixed)
- `src/polymarket_backtest/types.py` — Type definitions (fixed)
- `src/polymarket_backtest/metrics.py` — Metric calculations (fixed)
- `data/` — The database (fixed, read-only)

**The goal is: build a strategy that makes money with acceptable risk.**

Concretely, maximize a composite score:
1. Positive PnL — must make money
2. Positive Sharpe (or at least > -1) — must be risk-adjusted profitable
3. Low Brier score — better predictions = bigger edge
4. Good calibration (low ECE) — so Kelly sizing works

**Sharpe matters more than PnL.** A strategy with +5K PnL and Sharpe +0.5 beats a strategy with +30K PnL and Sharpe -30. Reject any strategy with Sharpe below -5 regardless of PnL. A strategy with 0 trades is a failure regardless of Brier score.

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep.

## Data

```
data/polymarket_backtest_v2.sqlite (READ ONLY)
```

This database contains **real Polymarket data** — no synthetic prices, no fake volume.

**Dataset stats** (verified 2026-03-13):
- **181,275 resolved binary markets** (YES/NO)
- **68,219,420 hourly price snapshots** (avg 376/market, min 5, max 14,188)
- **Time range**: Nov 2022 → Mar 2026
- **Outcome split**: 72.7% NO, 27.3% YES
- **0 invalid prices, 0 crossed bid/ask, 0 synthetic data, 0 duplicate timestamps**
- **DB size**: 19.4 GB

Key facts:
- Price data is real hourly CLOB prices
- Bid/ask spreads are estimated from price levels (not from real orderbook)
- `volume_24h` is total Gamma volume (same for all snapshots of a market) — not per-snapshot
- `volume_1m` is 0 — we don't have per-minute volume data
- `open_interest` is 0 — we don't have historical open interest
- **Fees = 0 in the backtest.** In production, Polymarket charges ~2% on net winnings.
- Domain is "general" for all markets (Gamma API didn't return granular categories)

Run `uv run python scripts/verify_data.py` to re-verify dataset integrity.

## Architecture

```
data/polymarket_backtest_v2.sqlite ─── READ ONLY
       │
       ▼
src/polymarket_backtest/features.py ─── Feature extraction
       │
       ▼
scripts/train_model.py ─── Train models
       │
       ▼
models/*.pkl ─── Saved trained models
       │
       ▼
src/polymarket_backtest/ml_transport.py ─── Uses model as forecaster
       │
       ▼
scripts/eval_strategies.py ─── Backtest with all strategies
       │
       ▼
results_ml.tsv / results.tsv ─── Experiment tracking
```

## Logging results

When an experiment is done, log it to `experiments.jsonl` (one JSON line per experiment, win or lose).

```json
{"id": 1, "timestamp": "2026-03-13T14:00:00", "phase": "baseline", "hypothesis": "baseline logistic", "files_changed": [], "metric_before": {}, "metric_after": {"test_brier": 0.15, "sharpe": 0.5}, "outcome": "KEPT", "commit": "abc1234", "notes": "baseline established"}
```

This is your institutional memory. Read it before starting each experiment.

## The experiment loop

LOOP FOREVER:

1. Read `experiments.jsonl` to see what was tried. Don't repeat failures unless you have a fundamentally different approach.
2. Make a single focused change to the code. One idea per experiment. Do NOT commit yet.
3. Run the evaluation:
   - For ML changes: `uv run python scripts/train_model.py --model logistic 2>&1 > run.log`
   - For strategy changes: `uv run python scripts/eval_strategies.py --forecast-mode ml_model --max-markets 200 2>&1 > eval.log`
   - Every 5 experiments, also run `--max-markets 1000` for validation.
4. Read the results: `grep "TEST_BRIER\|SHARPE\|PNL\|COMPOSITE" run.log eval.log`
5. If improved: keep the change, `git commit` the changed files only.
6. If not improved or unclear: `git checkout -- .` to revert, then **retrain the model** (`uv run python scripts/train_model.py --model logistic 2>&1 > run.log`) to restore the correct model pkl.
7. Log the result to `experiments.jsonl`. `git add experiments.jsonl && git commit -m "log: experiment #N — kept/discarded — summary"`
8. Go back to step 1.

**CRITICAL**: `git checkout -- .` reverts code but NOT model pkl (it's in .gitignore). If you trained during evaluation, the pkl contains the failed experiment's model. You MUST retrain after reverting.

**NEVER STOP.** The loop runs until you are manually stopped. If you run out of ideas, think harder — try combining near-misses, try more radical changes, analyze what features drive predictions.

## Validation Rules (no data leakage)

- **Walk-forward split**: Train on early markets, validate on middle, test on latest
- **Never peek at test set during development** — Only evaluate test at the end
- **Use validation set for all tuning decisions**
- **Markets in train/val/test must not overlap**
- **No future data in features** — Features at time T can only use data up to time T

## Rules

- One idea per experiment. Don't bundle changes.
- Evaluate BEFORE committing. Only winners enter git history.
- Check that strategies produce trades. TRADES=0 means the eval is meaningless.
- Keep it simple. Fewer features is better if performance is similar.
- No overfitting. If train Brier << test Brier, add regularization.
- Run tests periodically: `uv run pytest tests/ -q 2>&1`
- Install deps with `uv add <package>` if needed.
- Never `git add -A` or `git add .` — stage only files you changed.
- Don't spend >10 min on one experiment.
