# Research Brief — Polymarket Backtester

> Compiled from 10+ Codex research agents, March 12, 2026

---

## 1. Objective

Build a backtesting engine to test Grok-powered trading strategies on historical Polymarket data **before investing real capital**. The backtester must prevent data leakage and produce statistically meaningful metrics.

## 2. Data Sources (Validated Live)

### Primary: warproxxx/poly_data (~2 GiB)
- **Pre-processed trades** with prices computed: `timestamp, market_id, maker, taker, price, usd_amount, token_amount, direction`
- Markets metadata: `question, answer1, answer2, token1, token2, condition_id, volume, closedTime`
- Download: `wget https://polydata-archive.s3.us-east-1.amazonaws.com/archive.tar.xz`
- **Easiest to ingest**, trades already have computed prices

### Secondary: Goldsky Subgraph (free, from Nov 2022)
- 1.3M+ `orderFilledEvents` with `maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee, timestamp`
- Keyset pagination works: `where: { timestamp_gt: "..." }`
- No rate limiting observed (0.22-0.63s latency)
- **Fallback for markets not in warproxxx** or for deeper history

### Market Metadata: Gamma API
- All resolved markets with rules, `outcomePrices`, `closedTime`, `resolutionSource`
- 500 req/10s, pagination via `offset`
- **Resolution data derivable** from `closed=true` + `outcomePrices`

### Price History: CLOB API
- `/prices-history` gives second-level price series
- **CRITICAL: Returns empty for markets > few months old**
- Only useful for recently resolved markets

### Orderbook: pmxt Archive (limited)
- Only covers Feb 21 - Mar 12 2026 (~19 days)
- ~468 MB/hour, 16k markets per snapshot
- Schema: `timestamp, market_id, update_type, data` (JSON with bids/asks)
- **Too recent for historical backtest**, but useful for forward collection

### NOT Available
- No source has explicit resolution timestamps or outcomes — must derive
- No orderbook history before Feb 21, 2026
- OI subgraph has no timestamp — current state only
- Kaggle datasets: 404, no longer available

## 3. Grok API for Replay (Validated)

### What works for backtest
- `x_search` supports `from_date`/`to_date` — **usable for post-Nov 2024 markets**
- Client-side tools (function calling) for custom historical tools
- `store: false` prevents xAI from storing data
- `temperature: 0` for reproducibility
- System prompt temporal isolation

### What doesn't work
- `web_search` has **NO date filter** — cannot use in backtest
- No dated model IDs for `grok-4-fast-*` — only aliases
- Must log `system_fingerprint` for reproducibility

### Cost per evaluation
- ~$0.015/eval (tokens + x_search calls)
- Full 200-market backtest: ~$30
- Grid search (5-10 configs): ~$150-300

### Valid backtest window
- **Only markets resolved AFTER November 2024** (Grok knowledge cutoff)
- Pre-cutoff markets: test pipeline only, NOT forecast quality

## 4. Existing Code in Repo

### `src/polymarket_backtest/` (11 files, written by architecture agent)
- `schema.sql` — 13 SQLite tables
- `db.py` — temporal queries, seed data
- `grok_replay.py` — leakage prevention, replay client, context builder
- `market_simulator.py` — aggressive/passive fills, fees, impact
- `strategies.py` — carry, news-driven, deep research, grid config
- `replay_engine.py` — main loop, portfolio, resolution
- `metrics.py` — Brier, markout, calibration, Sharpe, edge decay, bootstrap
- `report.py` — markdown reports
- `demo.py` — end-to-end demo with seed data
- `types.py` — dataclasses (MarketState, ForecastOutput, etc.)

### Quality assessment
- Well-structured, typed, leakage-aware
- Uses deterministic transport for demo (no real Grok calls)
- XAI transport exists but untested
- Missing: **data pipeline** (ingestion from real sources), **report generation**, **CLI**

## 5. Repos Analyzed

### polymarket-paper-trader (agent-next, 6 stars)
- **Best book-walking simulator**: walks asks/bids level by level
- Fee formula: `(bps/10000) * min(price, 1-price) * size`
- 615 tests, 100% coverage
- **Limitations**: no historical data, backtest uses synthetic 3-level books, marks open inventory at $0
- **Extract**: book-walking fill logic, fee calculation

### polybot (ent0n29, 199 stars)
- Java/Kafka/ClickHouse — not directly usable
- **Valuable algorithms**: maker fill calibration (geometric model), replication scoring (L1 distance), execution quality metrics
- Complete-set arb: `edge = 1 - bid_up - bid_down`

### Jon-Becker/prediction-market-analysis (2.2k stars)
- 36 GiB Parquet dataset, blockchain OrderFilled events from 2020-2026
- No orderbook, no explicit resolutions
- Heavier than warproxxx for same data type

## 6. Python Packages to Use

| Package | Version | Purpose |
|---------|---------|---------|
| `scoringrules` | 0.9.0 | Brier, CRPS, log score, calibration |
| `forecasting-tools` | 0.2.85 | Metaculus validation before Polymarket |
| `py-clob-client` | 0.34.6 | Live execution (not backtest) |
| `polars` | latest | Parquet ingestion, fast transforms |
| `httpx` | latest | Async API calls |

## 7. Strategy Validation Ladder

```
1. Metaculus (free) — validate forecast quality
2. Backtest on historical Polymarket — validate strategy PnL
3. Paper trading on live Polymarket — validate execution
4. Micro-live ($100-250) — real money test
```

## 8. Key Metrics & Go/No-Go

| Metric | Minimum for Go |
|--------|---------------|
| Brier score | Agent < market price (50+ resolved markets) |
| Bootstrap CI(PnL) | Strictly > 0 (200+ trades) |
| Adverse selection | < 55% |
| Fill ratio | > 60% |
| Edge stability | Across domains, not just one market burst |

## 9. Statistical Significance
- Min 200-300 trades, 50 resolved markets for directional signal
- Min 1000 trades, 200 markets for strategy ranking
- Bootstrap CI, clustered standard errors, Diebold-Mariano test
