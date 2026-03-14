# Audit Report — Polymarket Trading Bot
**Date**: 2026-03-14
**Branch**: `autotrader/mar13`
**Audited by**: 12 parallel agents (10 completed, 2 failed OOM — failure confirmed findings)

---

## 1. Executive Summary

After 410+ autoresearch experiments, the trading bot shows **promising but significantly overstated results**. The model has genuine predictive signal (+7% over market mid-price), but the evaluation infrastructure has critical measurement bugs that inflate the Sharpe ratio by 5-10x and limit testing to 459 out of 181,275 available markets.

**Bottom line**: The strategy is likely viable (positive edge exists), but we cannot trust the current metrics. Fixes required before production decisions.

---

## 2. What We Built

### Model
- **Architecture**: XGBoost + isotonic calibration (post-hoc on validation set)
- **Features**: 16 features (price, spread, volume, resolution proximity, interactions)
- **Training data**: 4.2M samples from 49K markets (v2 DB, pre-Oct 2025)
- **Validation**: 2.8M samples from 44K markets (Oct-Dec 2025)
- **Test**: 4.6M samples from 74K markets (Jan 2026+)
- **Real training**: Yes, genuine ML pipeline with proper walk-forward validation

### Strategies
- **Family**: All 29 configs are `resolution_convergence` variants
- **Logic**: Buy YES when model predicts higher probability than market price, near resolution time
- **Parameters**: Vary by time horizon (72h-4320h), edge threshold (20-1000 bps), confidence (0.65-0.75), Kelly fraction (0.25-1.0)
- **Exit**: Thesis stop (exit if forecast drops by 9-12 cents from entry)

### Capital Management
The system has proper capital guards:
- **Pre-buy check** (replay_engine.py:472): Rejects orders if notional > available cash
- **Strict cash guard** (replay_engine.py:568-605): Reduces fill quantity if insufficient cash
- **Assert + clamp** (replay_engine.py:604-605): `assert cash >= 0`, clamp floating point dust
- **Kelly sizing**: Position size = `cash * kelly_fraction * kelly * mid_factor * confidence_factor`, capped by `max_position_notional`
- **Independent portfolios**: Each strategy gets $1,000 starting cash, operates independently (29 strategies = $29K total virtual capital)

**Limitations**:
- No portfolio-level risk management (no max drawdown, no correlation limits)
- No cross-strategy capital sharing (strategies don't compete for capital)
- Kelly fraction=1.0 on 2 strategies can attempt to bet >100% via confidence_factor 3x multiplier (clamped by execution guard)

---

## 3. Reported vs Real Metrics

| Metric | Reported | Estimated Real | Why |
|--------|----------|---------------|-----|
| **Sharpe** | 3.95 (best) | **0.4-0.8** | Zero-return dilution inflates 5-10x |
| **Combined PnL** | $17,246 | **$16,300-16,700** | After realistic fees (3-5.5%) |
| **Trades** | 1,401 | Correct | But heavy duplication across strategies |
| **Markets evaluated** | 500 | **459** (v1 DB) | Biased toward political/NO/long-running |
| **Brier (val)** | 0.1029 | Correct | Genuine, no leakage |
| **Model vs mid** | +7% | Correct | Real predictive signal |

---

## 4. Findings by Severity

### CRITICAL (3)

#### C1: Sharpe Ratio Inflated 5-10x
**File**: `src/polymarket_backtest/metrics.py` (compute_sharpe_like, L426-465)

The Sharpe calculation marks equity at every market snapshot for every strategy, even when no position exists. This creates 70-99% zero returns that compress standard deviation. Combined with irregular interval annualization (sqrt(8018) multiplier), all 26 strategies show Sharpe > 2.88.

For context: Sharpe > 3 is top 0.1% of hedge funds. Having ALL 26 strategies above 2.88 is systematic measurement bias.

**Evidence**:
- `resolution_convergence`: 99.3% zero returns out of 14,402 observations
- 49.5% of consecutive marks are <1 minute apart from different markets

#### C2: Market Selection Bias
**File**: `src/polymarket_backtest/grid_search.py` (L517-530)

Top 500 by snapshot count selects only long-running, political, liquid markets:
- 92% NO outcomes (vs 73% in random sample)
- 55% resolve in January 2026 (single correlated event)
- Median duration 335 days (vs 7 days for random markets)
- Median price $0.098 (long-shot political options)
- 100% of test-period markets were active during training

#### C3: Eval on Wrong Database
**File**: `scripts/eval_strategies.py` (L25)

Eval hardcoded to v1 DB (459 markets, real orderbooks) while model trained on v2 (181K markets, no orderbooks). The v1 markets ARE a subset of v2, so predictions work, but coverage is 0.25%.

On v2 with synthetic orderbooks: **0 trades** (confirmed by running eval on v2 — synthetic depth of 1 unit prevents any meaningful fills).

### HIGH (4)

#### H1: Thesis Stop Contradicts Hold-to-Resolution
**Files**: `strategies.py:553` vs `replay_engine.py:217-235`

`_resolution_convergence()` claims "Pure hold-to-resolution: no early exits" but `should_exit()` triggers thesis_stop BEFORE `decide()` is called. All 29 strategies have `use_thesis_stop=True`. The "hold-to-resolution" claim is false.

#### H2: Kelly=1.0 + 3x Confidence Multiplier
**File**: `strategies.py:580-584`, `grid_search.py:150,345`

Two strategies (`resolution_core`, `high_conviction`) with kelly_fraction=1.0. Combined with confidence_factor up to 3.0, they request $2400 from $1000 cash. Execution guard clamps silently, but sizing is degenerate.

#### H3: Resolution Proximity Broken at Inference
**File**: `ml_transport.py:182`

`ts` resolves to empty string `""` because `context_bundle["as_of"]` is at top level, not in `market` sub-dict. Result: `hours_to_resolution=720.0` constant (3/16 features are wrong at inference).

#### H4: Confidence Formula is Meaningless
**File**: `ml_transport.py:77`

`confidence = min(0.95, max(0.10, 0.5 + abs(probability - 0.5)))` just measures distance from 0.5. It correlates 0.86 with `abs(mid - 0.5)`. 85.6% of predictions pass the 0.65 threshold. Not a real uncertainty measure.

### MEDIUM (4)

#### M1: Edge Threshold 20 bps Too Aggressive
Two strategies (`nearterm_core`, `resolution_core`) enter on almost any positive edge. After fees (~60-100 bps), net edge requirement is essentially zero.

#### M2: Synthetic Orderbook = 1 Unit on v2
`volume_1m = 0` for all 68M snapshots → `min(50, max(1, 0*0.1)) = 1`. Every v2 fill is against a 1-unit synthetic book.

#### M3: Zero Strategy Diversification
All 29 strategies are `resolution_convergence` variants. Zero diversification against model failure.

#### M4: Brier Metric Doesn't Discriminate
All strategies share identical brier_improvement (forecasts computed once per market). The 30% brier weight in composite score adds zero discrimination.

### POSITIVE (5)

#### OK1: No Critical Data Leakage
Features don't use resolution outcomes. Chronological split is correct. Train/val/test don't overlap.

#### OK2: Model Beats Mid-Price
Test Brier: 0.0913 vs mid-price Brier: 0.0979 (~7% improvement). Real signal exists.

#### OK3: Fees Are Manageable
Realistic fees (trading + settlement) would reduce PnL by only 3-5.5% ($545-$940 on $17,246).

#### OK4: Spread Costs Implicitly Modeled
`aggressive_entry=True` buys at best_ask (2-4 cent spread). This partially substitutes for missing taker fees.

#### OK5: Model Overfitting is Moderate
Train Brier (0.068) << Test Brier (0.091), but model still generalizes. Val Brier (0.102) > Test Brier = no leakage sign.

---

## 5. Capital & Risk Assessment

### What Works
- Cash can never go negative (assert + clamp)
- Orders rejected if notional > available cash
- Fill quantity reduced to affordable amount if insufficient cash
- Kelly sizing limits position size relative to bankroll

### What's Missing
- No portfolio-level max drawdown
- No correlation limits between strategies
- No max exposure per market across strategies
- 29 independent $1K portfolios ≠ 1 realistic $29K portfolio
- No fee budget tracking

### Risk Profile (if deployed as-is)
- **Best case**: Model has real edge, Sharpe ~0.5-0.8, modest but consistent profits
- **Worst case**: Edge disappears on non-political markets, losses on unfamiliar market types
- **Structural risk**: All strategies fail simultaneously if model's resolution_convergence edge degrades

---

## 6. Robustness Checks (Debunked)

The autoresearch ran 8+ "robustness checks" comparing 500 vs 2000 markets. **These are invalid**: the v1 DB only has 459 markets, so both queries return the exact same universe. "Identical results" is trivially true, not evidence of robustness.

---

## 7. Action Plan

### Phase 1: Fix Critical Bugs (~1h)

| Fix | File | What |
|-----|------|------|
| **Sharpe** | `metrics.py:426-465` | Exclude zero-position periods, daily resampling |
| **ts bug** | `ml_transport.py:182` | Propagate `context_bundle["as_of"]` |
| **Confidence** | `ml_transport.py:77` | Replace with calibrated uncertainty or remove gating |
| **Orderbook** | `replay_engine.py:328` | Use `volume_24h/1440` instead of `volume_1m*0.1` |

### Phase 2: Corrected Evaluation (~30min)

- Fix market selection (stratified sampling)
- Migrate eval to v2 (with fixed orderbook)
- Re-run eval with corrected Sharpe

**Decision Gate**:
- Corrected Sharpe 0.4-0.8 → Continue building (Phase 3)
- Corrected Sharpe ~0 → Fundamental rethink

### Phase 3: Resume Autoresearch (autonomous)

- Deduplicate strategies (29 → 10-15)
- New strategy families (not just resolution_convergence)
- Updated program.md with corrected objectives
- v1 = execution realism, v2 = coverage/generalization

---

## 8. Files Reference

| File | Status | Role |
|------|--------|------|
| `metrics.py` | NEEDS FIX | Sharpe calculation |
| `ml_transport.py` | NEEDS FIX | ts bug + confidence |
| `replay_engine.py` | NEEDS FIX | Synthetic orderbook |
| `eval_strategies.py` | NEEDS FIX | Wrong DB path |
| `grid_search.py` | NEEDS FIX | Market selection + dedup |
| `strategies.py` | OK (misleading comment) | Strategy logic |
| `features.py` | OK | Feature extraction |
| `train_model.py` | OK | Model training |
| `prepare.py` | OK (FIXED) | Dataset preparation |
| `market_simulator.py` | OK | Fill simulation |
| `types.py` | OK | Type definitions |
| `db.py` | OK | Database layer |
