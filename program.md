# autotrader

This is an experiment to have the LLM do its own research on Polymarket trading strategies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`). The branch `autotrader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `program.md` — this file. The ground truth.
   - `src/polymarket_backtest/features.py` — feature extraction, dataset building, splits.
   - `scripts/train_model.py` — model training, hyperparameters.
   - `src/polymarket_backtest/ml_transport.py` — how the model is used as a forecaster.
   - `src/polymarket_backtest/strategies.py` — strategy logic.
   - `src/polymarket_backtest/grid_search.py` — strategy grid parameters.
4. **Verify data exists**: `ls -la data/polymarket_backtest_v2.sqlite` (should be ~19GB). If not, tell the human to run `uv run python scripts/download_exhaustive.py --fresh --workers 4` (~9h) then `uv run python scripts/backfill_events.py` (~20min).
5. **Prepare dataset**: `uv run python scripts/prepare.py` (~10-15min). Pre-extracts features from the 19GB SQLite into fast-loading .npz files in `data/prepared/`. Only needs to run once.
6. **Run the baseline**: `uv run python scripts/train_model.py --model logistic > run.log 2>&1`
7. **Record baseline metrics**: `grep "TEST_BRIER\|VAL_BRIER" run.log`
8. **Run baseline strategy eval**: `uv run python scripts/eval_strategies.py --forecast-mode ml_model --max-markets 500 > eval.log 2>&1`
9. **Record baseline strategy**: `grep "SHARPE\|PNL\|TRADES" eval.log`
10. **Initialize results.tsv**: Create with header row. Record baseline.
11. **Confirm and go.**

Once you get confirmation, kick off the experimentation.

## Experimentation

There are two experiment types: **model experiments** (improving predictions) and **strategy experiments** (improving trading). Each should complete within ~10 minutes.

**What you CAN edit:**
- `src/polymarket_backtest/features.py` — feature extraction, dataset building, splits
- `scripts/train_model.py` — model training, hyperparameters, architectures, calibration
- `src/polymarket_backtest/ml_transport.py` — how the trained model is used as a forecaster
- `src/polymarket_backtest/strategies.py` — strategy logic, entry/exit rules
- `src/polymarket_backtest/grid_search.py` — strategy grid parameters
- You can create new files: `scripts/train_*.py`, `src/polymarket_backtest/models/`, helper scripts.

**What you CANNOT edit:**
- `scripts/prepare.py` — dataset preparation (fixed, run once)
- `src/polymarket_backtest/replay_engine.py` — simulation engine (fixed)
- `src/polymarket_backtest/db.py` — database layer (fixed)
- `src/polymarket_backtest/market_simulator.py` — order execution (fixed)
- `src/polymarket_backtest/types.py` — type definitions (fixed)
- `src/polymarket_backtest/metrics.py` — metric calculations (fixed)
- `data/` — the database and prepared datasets (fixed, read-only)

**The goal: build a strategy that makes money with acceptable risk.**

The primary metrics, in order of importance:
1. **Sharpe ratio** — must be positive. Higher is better.
2. **PnL** — must be positive. Higher is better.
3. **Brier score** — lower is better. Better predictions = bigger edge.

**Sharpe matters more than PnL.** A strategy with +5K PnL and Sharpe +0.5 beats a strategy with +30K PnL and Sharpe -30. A strategy with 0 trades is a failure regardless of Brier score.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

## Data

```
data/polymarket_backtest_v2.sqlite (READ ONLY, ~19 GB)
```

This database contains **real Polymarket data** — no synthetic prices, no fake volume.

**Dataset stats** (verified 2026-03-13):
- **181,275 resolved binary markets** (YES/NO)
- **68,219,420 hourly price snapshots** (avg 376/market, min 5, max 14,188)
- **246,854 events** with tags (4,751 unique tags — Sports, Politics, Crypto, etc.)
- **Time range**: Nov 2022 → Mar 2026
- **Outcome split**: 72.7% NO, 27.3% YES
- **0 invalid prices, 0 crossed bid/ask, 0 synthetic data, 0 duplicate timestamps**
- **DB size**: 19.4 GB

Key facts:
- Price data is real hourly CLOB prices
- Bid/ask spreads are estimated from price levels (not from real orderbook)
- `volume_24h` is total Gamma volume (same for all snapshots of a market) — not per-snapshot
- `volume_1m` = 0 — we don't have per-minute volume data
- `open_interest` = 0 — we don't have historical open interest
- **Fees = 0 in the backtest.** In production, Polymarket charges ~2% on net winnings.
- Event tags available via `events`/`event_tags`/`market_events` tables (JOIN on event_id)

Run `uv run python scripts/verify_data.py` to re-verify dataset integrity.

## Output format

After training:
```
grep "TEST_BRIER\|VAL_BRIER\|TRAIN_BRIER" run.log
```

After strategy eval:
```
grep "SHARPE\|PNL\|TRADES\|COMPOSITE" eval.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	val_brier	sharpe	pnl	trades	status	description
```

1. git commit hash (short, 7 chars)
2. val_brier achieved (e.g. 0.1234) — use 0.0000 for crashes
3. sharpe ratio (e.g. 1.23) — use 0.00 for crashes or model-only experiments
4. pnl in USD (e.g. 5432.10) — use 0.00 for crashes or model-only experiments
5. number of trades — use 0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	val_brier	sharpe	pnl	trades	status	description
a1b2c3d	0.1850	0.00	0.00	0	keep	baseline logistic
b2c3d4e	0.1790	0.52	3210.50	847	keep	add price momentum features
c3d4e5f	0.1920	-2.10	-500.00	234	discard	switch to random forest
d4e5f6g	0.0000	0.00	0.00	0	crash	neural net OOM
```

Do NOT commit results.tsv — leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autotrader/mar13`).

LOOP FOREVER:

1. Look at results.tsv to see what's been tried. Don't repeat failures unless you have a fundamentally different approach.
2. Modify the code with an experimental idea. **One idea per experiment.** Do NOT bundle changes.
3. git commit the change.
4. Run the experiment:
   - For model changes: `uv run python scripts/train_model.py --model logistic > run.log 2>&1`
   - For strategy changes: `uv run python scripts/eval_strategies.py --forecast-mode ml_model --max-markets 500 > eval.log 2>&1`
   - Every 5 experiments, also run `--max-markets 2000` for robustness check.
5. Read out the results: `grep "TEST_BRIER\|VAL_BRIER" run.log` and/or `grep "SHARPE\|PNL\|TRADES" eval.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` (or `eval.log`) to read the stack trace and attempt a fix.
7. Record the results in results.tsv.
8. If improved (lower Brier or higher Sharpe), you "advance" the branch, keeping the commit.
9. If equal or worse, `git reset --hard HEAD~1` to revert. Then **retrain the model** (`uv run python scripts/train_model.py --model logistic > run.log 2>&1`) to restore the correct model pkl. This is critical: git reset reverts code but NOT the model pkl (it's in .gitignore).
10. Go back to step 1.

**Timeout**: Each experiment should complete within ~10 minutes. If a run exceeds 20 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: if it's something dumb and easy to fix (typo, missing import), fix and re-run. If the idea itself is fundamentally broken, log "crash" and move on.

**Check trades**: After strategy eval, always verify TRADES > 0. A strategy with 0 trades is meaningless — the eval tells you nothing.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes, analyze what features drive predictions, try different model architectures. The loop runs until the human interrupts you, period.

## Validation Rules (no data leakage)

- **Chronological split by resolution date**:
  - **Train**: markets resolved ≤ Q3 2025 (Sep 30, 2025) — ~63K markets (35%)
  - **Validation**: markets resolved Q4 2025 (Oct-Dec 2025) — ~44K markets (24%)
  - **Test**: markets resolved 2026+ (Jan 2026 onward) — ~74K markets (41%)
- **Never peek at test set during development** — only evaluate test at the final review
- **Use validation set for all tuning decisions** (model selection, hyperparameters, strategy thresholds)
- **Markets in train/val/test must not overlap**
- **No future data in features** — features at time T can only use data up to time T

## Ideas to explore

Rough priority order. Cross off as you try them. Start with quick wins, escalate to complex models.

### Phase 1: Feature engineering & classic ML
- [ ] Price velocity, acceleration, volatility features
- [ ] Time-to-resolution features (hours remaining, % of lifetime elapsed)
- [ ] Event tags as categorical features (JOIN event_tags table)
- [ ] Volume features: total volume, volume rank
- [ ] Market age features: days since open, days active
- [ ] LightGBM (handles categoricals natively, fast to train)
- [ ] XGBoost (strong baseline, GPU-acceleratable)
- [ ] CatBoost (native categorical support — perfect for event tags)
- [ ] Feature selection: drop low-importance features, SHAP analysis
- [ ] Calibration: Platt scaling, isotonic regression, temperature scaling

### Phase 2: Deep learning & sequence models
- [ ] MLP on tabular features (simple feedforward, start small)
- [ ] TabNet (attention-based deep learning for tabular data)
- [ ] LSTM / GRU on price time series (learn temporal patterns)
- [ ] Transformer on price sequences (self-attention over snapshots)
- [ ] 1D-CNN on price history (learn local patterns)
- [ ] Temporal Fusion Transformer (TFT — state of the art for time series)
- [ ] NLP embeddings: encode market titles with sentence-transformers → numeric features
- [ ] Cross-market features: correlation between markets in the same event

### Phase 3: Ensemble & meta-learning
- [ ] Ensemble: blend logistic + LightGBM + neural net (weighted average)
- [ ] Stacking: use model predictions as features for a meta-learner
- [ ] Per-category specialists: train separate models for sports/politics/crypto
- [ ] Online learning: update model weights as new data arrives
- [ ] Bayesian model averaging across ensemble members

### Phase 4: RL & advanced strategies
- [ ] RL agent (PPO/DQN): learn entry/exit timing from price trajectories
- [ ] Multi-armed bandit: dynamically allocate capital across strategies
- [ ] Kelly criterion with calibrated probabilities
- [ ] Contrarian: buy when price diverges from model probability
- [ ] Hold-to-resolution vs. dynamic exit (trailing stop, target profit)
- [ ] Category-specific strategies (sports vs politics vs crypto behave differently)
- [ ] Spread-aware entry: only enter when expected edge > spread cost
- [ ] Portfolio-level risk: max exposure per market, sector diversification
- [ ] Regime detection: identify trending vs mean-reverting markets

### Phase 5: Market microstructure & alpha
- [ ] Bid-ask spread dynamics as features (wide spread = uncertainty)
- [ ] Price impact estimation from historical spread patterns
- [ ] Event-driven features: markets that resolve near same date cluster
- [ ] Contrarian signal: extreme prices (>0.95 or <0.05) tend to mean-revert
- [ ] Momentum signal: trending markets continue trending
- [ ] Sentiment proxy: rapid price moves = new information arriving

### Infrastructure
- [ ] Faster eval: sample markets instead of running all 181K
- [ ] Feature importance analysis after each model change
- [ ] Overfit detection: compare train vs val Brier, early stopping
- [ ] Hyperparameter search with Optuna (Bayesian optimization)
- [ ] GPU training for neural nets (torch, check CUDA/MPS availability)
- [ ] Parallel model training: train multiple architectures concurrently
