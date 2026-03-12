# Polymarket Backtester — Implementation Plan

| Field | Value |
|-------|-------|
| **Feature** | Historical backtesting engine for Grok-powered Polymarket trading strategies |
| **Status** | Approved |
| **Complexity** | COMPLEX (18 tasks) |
| **Mode** | Normal (interactive) |
| **Interaction** | Terminal |
| **Created** | 2026-03-12 |

---

## Context Summary

We have extensive research (10+ Codex agents) and a skeleton framework (`src/polymarket_backtest/`, 11 files). The framework has a solid schema, replay engine, market simulator, strategies, and metrics — but lacks a data pipeline (ingestion from real sources), CLI, real Grok transport, and Metaculus validation.

The backtester must prove or disprove the hypothesis: "Grok + X/Twitter produces better-calibrated forecasts than Polymarket prices on political/societal markets."

Capital context: $1,000-5,000 deployment if validated. Backtest simulates $1,000.

---

## Requirements

### Functional
1. Download and ingest historical Polymarket data (trades, markets, resolutions)
2. Replay historical markets with date-bounded Grok forecasts (or deterministic fallback)
3. Simulate order fills with realistic book-walking, fees, and slippage
4. Test multiple strategies (carry, news-driven, deep research) with configurable parameters
5. Compute and report metrics: Brier score, markout, calibration, PnL, Sharpe, edge decay
6. Validate forecast quality on Metaculus before Polymarket
7. CLI interface for running backtests and viewing results

### Non-Functional
1. **Zero data leakage**: no information after timestamp T reaches the model
2. **Reproducibility**: pinned model versions, hashed contexts, deterministic replay
3. **Statistical rigor**: bootstrap CIs, clustered standard errors, min sample sizes
4. **Cost efficiency**: ~$30 per full backtest run with Grok
5. **Incremental**: works without Grok key (deterministic transport), adds Grok when available

---

## Technical Approach

### Architecture
```
CLI (click)
  → DataPipeline (download + ingest)
  → ReplayEngine (existing, enhanced)
    → GrokReplayClient (existing, add real xAI transport)
    → MarketSimulator (existing, enhance with paper-trader book-walking)
    → StrategyEngine (existing)
  → MetricsCalculator (existing, add scoringrules)
  → ReportGenerator (existing, enhance)
  → MetaculusValidator (new)
```

### Data Model
SQLite with existing 13-table schema. Key additions:
- Populate from warproxxx CSV / Goldsky GraphQL / Gamma REST
- Derive resolutions from `outcome_prices` + `closedTime`
- Build price snapshots from trade aggregation (no native OHLC)

### Key Decisions
1. **warproxxx first** (pre-processed trades with prices) — fastest to working backtest
2. **Goldsky fallback** for markets not in warproxxx
3. **Gamma API** for metadata + resolutions
4. **scoringrules** replaces custom Brier/scoring code
5. **No orderbook history** for now — simulate from trade-derived bid/ask spreads
6. **Start without Grok** — deterministic transport first, prove pipeline works

---

## Tasks

### Wave 1 — Foundation (parallel, no dependencies)

| ID | Task | Files | Agent | Validation |
|----|------|-------|-------|------------|
| 1 | **Project setup**: pyproject.toml, uv deps, ruff/ty config | `pyproject.toml` | coder-1 | `uv sync` passes, `uv run ruff check .` clean |
| 2 | **Data downloader: warproxxx** — download archive.tar.xz, extract, parse CSVs | `src/polymarket_backtest/downloaders/warproxxx.py` | coder-2 | Downloads sample, parses to dataframes |
| 3 | **Data downloader: Gamma API** — fetch resolved markets metadata | `src/polymarket_backtest/downloaders/gamma.py` | coder-3 | Fetches 10 resolved markets with rules |
| 4 | **Enhance existing db.py** — add bulk insert methods for real data | `src/polymarket_backtest/db.py` | coder-1 | Bulk insert 100 markets in <1s |

### Wave 2 — Data Pipeline (depends on Wave 1)

| ID | Task | Depends | Files | Agent | Validation |
|----|------|---------|-------|-------|------------|
| 5 | **Data ingester: warproxxx → SQLite** — map CSV columns to schema, derive snapshots from trades | 1, 2, 4 | `src/polymarket_backtest/ingest.py` | coder-2 | 100+ markets with snapshots in SQLite |
| 6 | **Data ingester: Gamma → SQLite** — market metadata, rules, derive resolutions | 1, 3, 4 | `src/polymarket_backtest/ingest.py` | coder-3 | Resolutions derived for closed markets |
| 7 | **Data downloader: Goldsky** — GraphQL client for orderFilledEvents with keyset pagination | 1 | `src/polymarket_backtest/downloaders/goldsky.py` | coder-1 | Fetches 1000 fills for a given market |
| 8 | **CLOB price history** — fetch `/prices-history` for recent markets | 1 | `src/polymarket_backtest/downloaders/clob.py` | coder-3 | Price series for 5 recent markets |

### Wave 3 — Engine Enhancements (depends on Wave 2)

| ID | Task | Depends | Files | Agent | Validation |
|----|------|---------|-------|-------|------------|
| 9 | **Enhance market_simulator.py** — port book-walking logic from paper-trader | 5 | `src/polymarket_backtest/market_simulator.py` | coder-1 | Fill simulation matches paper-trader on same input |
| 10 | **Enhance metrics.py** — integrate scoringrules, add log_score | 1 | `src/polymarket_backtest/metrics.py` | coder-2 | `sr.brier_score` produces same results as current impl |
| 11 | **Snapshot builder** — aggregate trades into market_snapshots (bid/ask/mid/volume) | 5, 6 | `src/polymarket_backtest/snapshot_builder.py` | coder-3 | 5-min snapshots with derived bid/ask from trades |
| 12 | **Enhance replay_engine.py** — handle real data edge cases, missing snapshots | 5, 6, 11 | `src/polymarket_backtest/replay_engine.py` | coder-1 | Runs on 20 real markets without crashes |

### Wave 4 — CLI & Grok (depends on Wave 3)

| ID | Task | Depends | Files | Agent | Validation |
|----|------|---------|-------|-------|------------|
| 13 | **CLI** — click-based: `download`, `ingest`, `backtest`, `report` commands | 9, 10, 11, 12 | `src/polymarket_backtest/cli.py` | coder-1 | `uv run polymarket-backtest download && uv run polymarket-backtest backtest` works end-to-end |
| 14 | **Grok xAI transport** — real API calls with x_search date-bounded, store=false | 12 | `src/polymarket_backtest/grok_replay.py` | coder-2 | Calls Grok API with date bounds, returns forecast JSON |
| 15 | **Metaculus validator** — call Grok on open Metaculus questions, compare to community | 1 | `src/polymarket_backtest/metaculus_validator.py` | coder-3 | Forecasts 10 Metaculus questions, computes Brier |

### Wave 5 — Integration & Testing (depends on Wave 4)

| ID | Task | Depends | Files | Agent | Validation |
|----|------|---------|-------|-------|------------|
| 16 | **End-to-end test** — full pipeline: download → ingest → backtest → report | 13 | `tests/test_e2e.py` | tester | E2E passes on 50+ real resolved markets |
| 17 | **Report generator** — markdown report with summary, per-strategy PnL, calibration, go/no-go | 13 | `src/polymarket_backtest/report.py` | coder-1 | Report generated with all metrics sections |
| 18 | **Grid search** — run multiple strategy configs, compare, rank | 13, 17 | `src/polymarket_backtest/grid_search.py` | coder-2 | 4+ strategy configs compared in one report |

---

## Agent Team & Parallelism

**Team name**: `trading-bot-backtester`

```
                           ┌─────────────┐
                           │  LEAD (me)  │
                           │ orchestrate │
                           └──────┬──────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
        ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
        │  CODER-1  │      │  CODER-2  │      │  CODER-3  │
        │ engine,   │      │ warproxxx,│      │ gamma,    │
        │ simulator,│      │ metrics,  │      │ goldsky,  │
        │ CLI, db   │      │ grok, grid│      │ clob,     │
        │           │      │           │      │ metaculus │
        └───────────┘      └───────────┘      └───────────┘

Wave 1:  [1,4]──coder-1    [2]──coder-2      [3]──coder-3      ← PARALLEL
Wave 2:  [7]───coder-1     [5]──coder-2      [6,8]──coder-3    ← PARALLEL
Wave 3:  [9,12]─coder-1    [10]─coder-2      [11]──coder-3     ← PARALLEL
Wave 4:  [13]──coder-1     [14]─coder-2      [15]──coder-3     ← PARALLEL
Wave 5:  [17]──coder-1     [18]─coder-2      [16]──tester      ← PARALLEL
```

### Executor Table

| Task | Executor | Sandbox | Rationale |
|------|----------|---------|-----------|
| 1 | Codex via relay | workspace-write | Project setup, isolated |
| 2-3 | Codex via relay | danger-full-access | Network access for downloads/APIs |
| 4 | Codex via relay | workspace-write | DB code, isolated |
| 5-8 | Codex via relay | danger-full-access | Network + file write |
| 9-12 | Codex via relay | workspace-write | Code enhancement, isolated |
| 13-15 | Codex via relay | danger-full-access | CLI/API integration |
| 16 | Codex via relay | danger-full-access | E2E needs network |
| 17-18 | Codex via relay | workspace-write | Report/grid, isolated |

---

## Git Workflow

- **Base branch**: main (or current HEAD)
- **Feature branch**: `feature/polymarket-backtester`
- **Worktree**: agents work in isolated worktrees
- **Merge**: squash-merge feature branch when all tasks complete
- **No commits until approved** — `/code` handles commits

---

## Skills & MCPs

| Agent | Skills | MCPs |
|-------|--------|------|
| All coders | mgrep, commit-work | Codex |
| coder-2, coder-3 | — | Codex, Perplexity (docs lookup) |
| tester | python-quality-gate | Codex |

---

## Testing Strategy

### Per-task validation
Each task has explicit validation criteria (see task table). All must pass `uv run ruff check` and `uv run ty` on modified files.

### Integration test
Task 16: full E2E pipeline on 50+ real resolved markets.

### GO/NO-GO criteria
After backtest runs:
- Brier(agent) < Brier(market) on 50+ markets?
- Bootstrap CI(PnL) > 0 on 200+ trades?
- No crashes or data leakage detected in replay_audit table?

---

## Out of Scope

- Live trading / paper trading on Polymarket WebSocket
- Web UI / dashboard
- Multi-agent orchestration (the bot itself)
- Orderbook reconstruction (no historical data available pre-Feb 2026)
- Deployment to cloud (GCP/AWS)
- Grok multi-agent (single forecaster for now)

---

## Open Questions

1. **Orderbook simulation fidelity**: Without historical orderbook, we derive bid/ask from trades. How much does this distort fill simulation? → Mitigated by also testing with real recent pmxt data (Feb-Mar 2026).

2. **Resolution derivation accuracy**: Deriving resolutions from `outcome_prices` may miss edge cases (disputes, 50/50). → Validate against a sample of known resolutions manually.

3. **x_search date reliability**: Does xAI's `x_search` reliably exclude posts after `to_date`? → Test empirically with known historical events before trusting results.
