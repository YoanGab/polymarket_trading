# Polymarket Backtesting Architecture

This design assumes a binary-market Polymarket bot whose forecasting engine is
Grok, but whose replay context is built entirely from locally stored,
date-bounded data.

## 1. Architecture

```text
                         +----------------------------------+
                         | Immutable Snapshot Store         |
                         | SQLite/Postgres                  |
                         | markets, rules, books, news      |
                         +----------------+-----------------+
                                          |
                                   as_of <= replay clock
                                          |
                         +----------------v-----------------+
                         | Replay Context Builder           |
                         | local tools only                 |
                         | rules/news/book snapshots        |
                         +----------------+-----------------+
                                          |
                           audit every bounded local query
                                          |
         +------------------------------- v ------------------------------+
         | Replay Grok Client                                             |
         | pinned model release                                           |
         | no web_search / x_search / code tools                          |
         | temporal isolation system prompt                               |
         +-------------------------------+--------------------------------+
                                         |
                                         v
                         +---------------+----------------+
                         | Strategy Grid                   |
                         | carry, news, research, arb     |
                         | Kelly / thresholds / stops     |
                         +---------------+----------------+
                                         |
                                         v
                         +---------------+----------------+
                         | Market Simulator               |
                         | orderbook, fills, impact, fees |
                         | queue model, markouts          |
                         +---------------+----------------+
                                         |
                                         v
                         +---------------+----------------+
                         | Portfolio + Resolution Engine  |
                         | positions, cash, marks, PnL    |
                         +---------------+----------------+
                                         |
                                         v
                         +---------------+----------------+
                         | Metrics + Significance         |
                         | Brier, markout, calibration    |
                         | Sharpe-like, bootstrap tests   |
                         +---------------+----------------+
                                         |
                                         v
                         +---------------+----------------+
                         | Reports                         |
                         | markdown, csv, parquet, plots   |
                         +---------------------------------+
```

## 2. Replay Loop

The replay clock advances on the union of:

- market snapshot timestamps
- news/document `first_seen_ts`
- rule revision timestamps
- market lifecycle timestamps such as `close_ts` and `resolution_ts`

For each `(market_id, T)`:

1. Load the latest market state with `snapshot.ts <= T`.
2. Load the latest rule text and any clarifications with `effective_ts <= T`.
3. Load only documents with both:
   - `published_ts <= T`
   - `ingested_ts <= T`
4. Build a bounded context bundle.
5. Call Grok with the pinned model release and no server-side tools.
6. Convert forecast output into orders using the strategy under test.
7. Simulate fills against the historical or reconstructed book at `T`.
8. Mark inventory to the next available midpoint or bid/ask mark.
9. If the market reaches a lifecycle transition, process it:
   - close for trading
   - pending resolution
   - disputed
   - final resolution
10. Persist orders, fills, marks, model output, and audit records.

### What data the model can see at time `T`

Allowed:

- market title and category
- rules and clarifications effective at or before `T`
- orderbook snapshot and price history up to `T`
- volume, spread, and open interest up to `T`
- news/event documents published and ingested by `T`
- related-market states up to `T`

Forbidden:

- any document first seen after `T`
- any trade, price, or book level after `T`
- any final market resolution when `resolution_ts > T`
- any live web or live X search

### Recommended context bundle

```json
{
  "as_of": "2026-03-01T12:05:00+00:00",
  "market": {
    "market_id": "pm_dropout_june",
    "title": "Will Candidate X drop out by June 30?",
    "best_bid": 0.44,
    "best_ask": 0.48,
    "mid": 0.46,
    "tick_size": 0.01,
    "volume_1m": 950.0,
    "volume_24h": 188000.0
  },
  "rules": {
    "rules_text": "Resolves YES if candidate officially suspends...",
    "additional_context": ""
  },
  "recent_news": [
    {
      "document_id": "doc_123",
      "source": "official",
      "published_ts": "2026-03-01T12:03:00+00:00",
      "title": "Campaign statement announces suspension"
    }
  ],
  "related_markets": []
}
```

## 3. Leakage Prevention

### Grok access policy

Use Grok only as a text model over pre-built replay context.

- Do not expose xAI built-in tools.
- Do not pass `web_search`, `x_search`, `code_execution`, or file-search tools.
- Do not use floating aliases such as `<model>-latest`.
- Persist the exact `model_release`, prompt hash, and context hash for every
  inference.

As of March 12, 2026, xAI documents both built-in server-side tools and
date-pinned model releases. Their docs also state that `<model>-<date>` points
to a specific release and is not updated, while built-in tools such as
`web_search` and `x_search` are enabled only when explicitly configured.

### Pinned model recommendation

Good:

- `grok-4.20-beta-0309-reasoning`

Bad:

- `grok-4-fast-reasoning`
- `grok-4.20-beta-latest-non-reasoning`

The strict rule is:

- experiment config must contain `model_release`
- the runtime fails closed if `model_release` does not match a release-pinned
  regex such as `.*-[0-9]{4}$`

### Temporal isolation system prompt

```text
You are running inside a historical replay.

Current replay time: 2026-03-01T12:05:00+00:00
You must behave as if nothing after this timestamp exists.

Hard rules:
1. Use only the evidence included in the provided replay context.
2. Never assume access to live web, live X, future prices, future news, or
   final market outcomes.
3. If evidence is insufficient as of the replay time, say so explicitly.
4. Do not speculate using knowledge from outside the supplied context bundle.
5. Return calibrated probability_yes, confidence, key evidence, key risks, and
   a short thesis.
```

### Operational controls

- Local tool layer enforces `WHERE ts <= as_of`.
- Store both `published_ts` and `ingested_ts`; use both.
- Add optional source-specific lag:
  - Reuters/AP: 30-120 seconds
  - official site fetches: observed polling cadence
  - social feeds: measured collector latency
- Hash every model input bundle. If the same market/time is replayed, the hash
  must be identical.
- Keep inference deterministic:
  - `temperature = 0`
  - seeded tool ordering
  - stable serialization

## 4. Simulation Fidelity

### Orderbook simulation without tick-level history

Best case:

- full historical L2 or L3 snapshots plus trades

Realistic MVP:

- 1-minute or 5-minute top-of-book plus visible depth ladder
- trade price history
- volume by interval

If you do not have full tick history, simulate with two layers:

1. Visible liquidity layer
   - consume the stored ladder directly
   - marketable orders walk levels until size is exhausted
2. Latent impact layer
   - residual quantity beyond visible depth moves price by a calibrated impact
     function
   - use a square-root style penalty:
     `impact_bps = k * sqrt(residual_qty / max(visible_depth, 1)) * 10_000`

### Slippage model

For aggressive orders:

- fill visible levels exactly
- residual size incurs impact
- mark execution at VWAP over visible levels plus latent impact

For passive orders:

- estimate queue ahead from visible size at the order price
- estimate marketable flow from future bar volume and side move
- fill only if estimated flow clears the queue ahead
- partial fill if only part of the queue is consumed

### Time-to-fill model

Without tick data, model time-to-fill as a hazard:

- `P(fill in horizon) = min(1, aggressor_flow / queue_ahead)`
- `expected_time = horizon * queue_ahead / max(aggressor_flow, epsilon)`

Use separate horizons for:

- 1 minute for event-driven urgency
- 5 to 30 minutes for normal liquidity taking
- longer for passive carry trades

### Fee model

As of March 12, 2026, official Polymarket docs say:

- most markets are fee-free
- fee-enabled markets include crypto, NCAAB, and Serie A markets
- taker fee formula:
  `fee = C * p * feeRate * (p * (1 - p))^exponent`
- crypto fee curve peaks around 1.56% effective rate at 50c
- sports fee curve peaks around 0.44% at 50c
- makers may receive rebates in fee-enabled markets, but qualification depends
  on the maker program

Backtest default:

- apply taker fees exactly when `fees_enabled = true`
- assume `maker_rebate = 0` unless you have historical scoring and rebate
  eligibility data

### Resolution handling

Persist market state transitions explicitly:

- `active`
- `closed_for_trading`
- `pending_resolution`
- `proposed`
- `challenged`
- `disputed`
- `resolved`
- `cancelled`

As of March 12, 2026, Polymarket docs describe:

- a 2-hour challenge period after a proposal
- a second proposal round if disputed
- escalation to UMA DVM after a second dispute
- rare `unknown/50-50` outcomes
- onchain clarifications via bulletin board updates

Backtest rules:

- stop opening new positions once trading closes
- keep marking inventory to the last valid midpoint while unresolved
- capital remains tied up during disputes
- final payout is applied only at final `resolution_ts`
- if resolution is `50/50`, settle YES and NO at `0.5`
- if market is cancelled or voided, settle at collateral return rule from the
  historical record

## 5. Metrics

Track each metric by:

- strategy
- market domain
- agent or model role
- calendar bucket

Required metrics:

- Brier score per agent/domain
- markout at 1, 5, 30, 240 minutes
- adverse selection:
  - signed markout immediately after entry
- fill ratio:
  - `filled_qty / requested_qty`
- PnL pre-resolution:
  - exits before final resolution
- hold-to-resolution PnL:
  - open until final payout
- calibration curve:
  - decile bucket forecast vs realized frequency
- Sharpe-like ratio:
  - daily marked-to-mid return Sharpe or information ratio
- edge decay:
  - initial expected edge vs future realized edge across horizons

Useful formulas:

- Brier:
  `mean((p_yes - outcome_yes)^2)`
- Signed markout for YES buy:
  `future_mid - fill_price`
- Adverse selection rate:
  `share(markout_horizon < 0)`
- Edge retention:
  `realized_markout_h / initial_edge`

## 6. Strategy Grid

Treat strategy variants as parameterized policies, not separate engines.

Grid to run:

- carry only:
  - buy 95-99c and hold to resolution
- news-driven:
  - enter after event shocks
- deep research:
  - slower entry, longer hold
- cross-market logical arbitrage:
  - detect impossible joint prices
- Kelly fractions:
  - 0.05, 0.10, 0.15, 0.25
- edge thresholds:
  - 25, 50, 100, 200 bps
- holding periods:
  - 5, 30, 240, resolution
- position management:
  - none
  - thesis stops
  - time stops

Recommended experiment dimensions:

- `strategy_family`
- `kelly_fraction`
- `edge_threshold_bps`
- `liquidity_style`
- `max_holding_minutes`
- `thesis_stop_delta`
- `time_stop_minutes`

## 7. SQLite MVP Schema

Core tables:

- `markets`
- `market_rule_revisions`
- `market_snapshots`
- `orderbook_levels`
- `news_documents`
- `market_news_links`
- `model_outputs`
- `orders`
- `fills`
- `positions`
- `pnl_marks`
- `market_resolutions`
- `metric_results`
- `replay_audit`

Design choices:

- rules and clarifications are versioned, never overwritten
- orderbook levels are normalized, not just stored as JSON
- model output persists reasoning, evidence IDs, and prompt hashes
- audit table records every bounded local tool call

## 8. Python Structure

```text
src/polymarket_backtest/
  __init__.py
  db.py
  demo.py
  grok_replay.py
  market_simulator.py
  metrics.py
  replay_engine.py
  report.py
  schema.sql
  strategies.py
  types.py
```

Responsibilities:

- `db.py`: schema init, bounded queries, demo seed
- `grok_replay.py`: local context build, temporal system prompt, pinned model
  requests
- `market_simulator.py`: fee model, market impact, passive fill model
- `strategies.py`: parameterized policies
- `replay_engine.py`: main loop, portfolio state, resolution handling
- `metrics.py`: post-trade and forecast evaluation
- `report.py`: markdown summary

## 9. Statistical Significance

Use both trade-level and market-level analysis. Prediction-market samples are
clustered, so raw trade counts overstate confidence.

### Minimum useful sample sizes

For early directional reads:

- at least 200 to 300 trades
- at least 50 resolved markets

For stable strategy ranking:

- at least 1,000 trades
- at least 200 resolved markets
- at least 3 independent market regimes

For calibration analysis:

- target at least 100 outcomes per forecast decile

### Tests

Use:

- bootstrap confidence intervals for mean PnL and markout
- paired bootstrap when comparing two strategies on the same events
- Diebold-Mariano style test for forecast loss differences
- permutation test for edge-vs-no-edge grouping
- clustered standard errors by market or event family
- survival analysis for time-to-fill

Do not rely on:

- naive t-tests over raw fills without clustering
- Sharpe estimates from a single event regime

### Practical interpretation

Look for:

- confidence interval on mean edge strictly above 0 after fees
- stable Brier improvement across domains, not just one market burst
- markout and hold-to-resolution PnL agreeing in direction
- edge decay slower than your execution latency

## 10. Sources

Official references used for time-sensitive parts of this design:

- Polymarket fees: `https://docs.polymarket.com/trading/fees`
- Polymarket order lifecycle: `https://docs.polymarket.com/concepts/order-lifecycle`
- Polymarket resolution: `https://docs.polymarket.com/concepts/resolution`
- xAI models and aliases: `https://docs.x.ai/developers/models`
- xAI tool calling overview: `https://docs.x.ai/developers/tools/overview`
