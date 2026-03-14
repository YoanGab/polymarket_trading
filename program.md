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

The previous autoresearch cycle ran 410+ experiments against partially broken trading infrastructure. Historical strategy Sharpe and PnL numbers from before the fixes are not comparable to current results. Treat strategy evaluation as re-baselined from here, while preserving only the findings below that still survive the infrastructure audit.

**Infrastructure that is now fixed:**
- Sharpe now uses daily returns with position-only filtering. Old Sharpe values were inflated by roughly 5-10x.
- Resolution-proximity inference features now work correctly.
- The confidence formula has been simplified; the old one mostly measured distance from 0.5 and was not useful.
- The synthetic orderbook now uses `volume_24h` as the liquidity proxy. The old v2 1-unit book prevented many trades from executing.
- Strategy evaluation now runs on v2 data (181,275 markets), not the old v1 subset (459 markets).
- Market selection now uses stratified random sampling. Never go back to top-N-by-snapshots sampling.

**What remains valid from prior research:**
- XGBoost + isotonic calibration beats mid-price by about 7% on test Brier (about 0.091 vs 0.098).
- The `resolution_convergence` family is still the strongest known strategy family.
- `thesis_stop = 0.12` is the best known stop for the current model.
- The current grid likely contains about 29 configs and should be collapsed to roughly 10-15 materially different configs.

**What you CAN edit:**
- `src/polymarket_backtest/features.py` — feature extraction, dataset building, splits
- `scripts/train_model.py` — model training, hyperparameters, architectures, calibration
- `src/polymarket_backtest/ml_transport.py` — how the trained model is used as a forecaster
- `src/polymarket_backtest/strategies.py` — strategy logic, entry/exit rules
- `src/polymarket_backtest/grid_search.py` — strategy grid parameters, stratified sampling behavior, experiment plumbing
- You can create new files: `scripts/train_*.py`, `scripts/*.py`, `src/polymarket_backtest/models/`, helper scripts.

**What you CANNOT edit:**
- `scripts/prepare.py` — dataset preparation (fixed, do not touch)
- `src/polymarket_backtest/replay_engine.py` — simulation engine (fixed, do not touch)
- `src/polymarket_backtest/db.py` — database layer (fixed, do not touch)
- `src/polymarket_backtest/market_simulator.py` — execution model (fixed, now with the corrected orderbook proxy)
- `src/polymarket_backtest/types.py` — type definitions (fixed, do not touch)
- `src/polymarket_backtest/metrics.py` — metric calculations (fixed, now with the corrected Sharpe)
- `data/` — the database and prepared datasets (fixed, read-only)

**The goal: build a strategy that still works after the infrastructure fixes.**

The primary metrics, in order of importance:
1. **Sharpe ratio** — on the corrected daily-return basis. A Sharpe of 0.5 is good. A Sharpe of 1.0 is excellent. Sharpe above 2.0 is suspicious and should trigger a bug audit, not celebration.
2. **Trade count** — target 100+ trades on v2. A high Sharpe on a tiny number of trades is not enough.
3. **PnL** — must be positive. Higher is better.
4. **Brier score** — lower is better. Better predictions matter because if strategies plateau, model improvement is the next lever.

**Sharpe matters more than raw PnL, and robustness matters more than one lucky sample.** A strategy with Sharpe +0.62 across several stratified seeds beats a strategy with Sharpe +1.10 on one seed and ~0 on the next two. A strategy with 0 trades is a failure regardless of Brier score.

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

For corrected-infrastructure strategy experiments, use the `description` field to capture the family name and seed when relevant (for example `baseline corrected stratified seed=3` or `contrarian extreme fade seed=11`).

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autotrader/mar13`).

Work in phases. Do not skip ahead.

### Phase 0: Verify the corrected baseline

1. Re-read `program.md` and `results.tsv`.
2. Train the current baseline model.
3. Run the corrected strategy eval on v2 with stratified market sampling.
4. Record the corrected Sharpe, PnL, and trade count as the new baseline. Historical pre-fix strategy metrics are obsolete.
5. Run the baseline on multiple market-sampling seeds. If the code does not expose a seed, add that capability in editable code before trusting any strategy result.

### Phase 1: Deduplicate the strategy grid

1. Analyze the current strategy configs for trade overlap, PnL correlation, and parameter redundancy.
2. Collapse the current grid from about 29 configs down to roughly 10-15 materially distinct configs.
3. Remove economically identical variants before spending experiments on new thresholds.
4. Keep `resolution_convergence` as the control family, with `thesis_stop = 0.12` as the default reference point.

### Phase 2: Explore new strategy families

1. Explore families beyond `resolution_convergence`.
2. Priority order:
   - `contrarian` — fade extreme prices or exhausted moves.
   - `momentum` — follow directional moves with volume confirmation.
   - `mean_reversion` — fade dislocations after temporary volume/price shocks.
   - `edge_based` — trade pure forecast edge with simple risk controls.
3. Keep experiments clean: one family or one mechanism change per experiment.

### Phase 3: Improve the model if strategies plateau

1. If corrected Sharpe stays near 0 (roughly between -0.1 and +0.1) after several strategy experiments, switch effort to model work.
2. Prioritize improvements that can plausibly improve ranking and calibration: better tree models, calibration, feature cleanup, feature additions, and inference consistency.
3. Start from the known good direction (`xgboost` + isotonic) before trying exotic models.
4. Do not spend many experiments threshold-tuning a weak model.

### Ongoing loop

1. Look at `results.tsv` to see what's been tried. Don't repeat failures unless you have a fundamentally different approach.
2. Modify the code with one experimental idea. Do NOT bundle changes.
3. Commit the change.
4. Run the experiment:
   - For model changes: `uv run python scripts/train_model.py --model logistic > run.log 2>&1`
   - For strategy changes: `uv run python scripts/eval_strategies.py --forecast-mode ml_model --max-markets 500 --no-in-memory > eval.log 2>&1`
   - The eval must use the v2 + stratified-sampling path. Never switch back to the old v1 or top-N evaluation logic.
   - Promising strategy results are not real until they survive multiple stratified sampling seeds.
5. Read out the results from `run.log` and/or `eval.log`.
6. If the expected metrics are missing, the run crashed. Read the stack trace and attempt a fix.
7. Record the results in `results.tsv`.
8. If improved, keep the commit and advance.
9. If equal or worse, restore the previous code state cleanly. Then retrain the model if needed to realign ignored model artifacts with the checked-out code.
10. Every 5 experiments, re-run a robustness check with different random seeds for market sampling.
11. Every 5 experiments, re-read `program.md` from disk before choosing the next idea.
12. Go back to step 1.

**Timeout**: Each experiment should complete within ~10 minutes. If a run exceeds 20 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: if it's something dumb and easy to fix (typo, missing import), fix and re-run. If the idea itself is fundamentally broken, log "crash" and move on.

**Check trades**: After strategy eval, always verify `TRADES > 0`. A strategy with 0 trades is meaningless — the eval tells you nothing.

**CURRENT PRIORITY (updated by human — READ THIS)**:
1. First verify the corrected baseline. The first post-fix milestone is a clean baseline eval recorded in `results.tsv`.
2. Second deduplicate the strategy grid using trade overlap and redundancy analysis. Shrink from about 29 configs to 10-15 distinct ones.
3. Third explore new strategy families beyond `resolution_convergence`: `contrarian`, `momentum`, `mean_reversion`, and `edge_based`.
4. Fourth, if corrected Sharpe is near 0 or the new strategy families stall, improve the model before trying many more strategy variants.
5. The target is corrected Sharpe `> 0.5` with `100+ trades` on v2 using stratified market sampling, and the result must survive multiple random seeds.
6. Do NOT expect Sharpe `> 2.0`. That likely indicates a bug, not a breakthrough.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue unless you are blocked by missing data, a broken environment, or conflicting instructions from `program.md`. The human may be asleep and expects you to continue working autonomously until manually stopped. If you run out of ideas, re-read the in-scope files, inspect failures, analyze what features actually drive the model, and shift from strategy work to model work when the corrected Sharpe says the strategies have plateaued.

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

Rough priority order. Cross off as you try them. Start with baseline verification and strategy dedupe before broadening the search.

### Phase 0: Baseline & robustness
- [ ] Record the corrected baseline Sharpe, PnL, and trade count on v2 with stratified sampling
- [ ] Add or verify random-seed control for market sampling
- [ ] Run the corrected baseline across multiple seeds and record variance
- [ ] Build a simple overlap/correlation report for the current strategy configs

### Phase 1: Deduplicate existing strategy configs
- [ ] Cluster configs by trade overlap / PnL correlation
- [ ] Remove configs that differ only trivially in thresholds but take the same trades
- [ ] Keep a compact control set around `resolution_convergence`
- [ ] Re-confirm `thesis_stop = 0.12` under corrected infrastructure as the reference stop

### Phase 2: New strategy families
- [ ] `contrarian`: fade extreme prices or exhaustion moves
- [ ] `momentum`: follow price trends with volume confirmation
- [ ] `mean_reversion`: fade temporary price dislocations after spikes
- [ ] `edge_based`: trade pure forecast edge against price / spread
- [ ] Category- or tag-specific variants only if the base family works first
- [ ] Portfolio/risk overlays only after a family shows standalone edge

### Phase 3: Model improvement
- [ ] Re-establish `xgboost` + isotonic as the main model baseline
- [ ] Improve resolution and time-to-resolution features now that inference is fixed
- [ ] Add better liquidity / volume features keyed to `volume_24h`
- [ ] Add event-tag features or category specialists if they improve validation Brier
- [ ] Compare calibration methods: isotonic vs Platt vs temperature scaling
- [ ] Run feature importance / SHAP / ablation to remove dead features

### Phase 4: Support tooling
- [ ] Multi-seed eval harness for stratified sampling
- [ ] Auto-log per-seed results and summary stats
- [ ] Overfit checks: compare train vs val vs test Brier before trusting strategy changes
- [ ] Faster scripts for trade-overlap analysis, config dedupe, and family comparison
