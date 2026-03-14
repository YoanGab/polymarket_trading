# Autoresearch Prompt

Copy-paste this into a new Claude Code session to start the autoresearch.

---

## Prompt

I want you to run the autoresearch experiment loop on this Polymarket trading bot.

**IMPORTANT: Read `program.md` first.** It is the ground truth for everything — objectives, constraints, what you can edit, the experiment loop, and priorities.

**Context**: We just completed a comprehensive audit (see `AUDIT_REPORT.md`) and fixed critical infrastructure bugs:
1. **Sharpe calculation** was inflated 5-10x — now uses daily returns with position-only filtering
2. **Resolution proximity** features were broken at inference — now fixed
3. **Confidence formula** was meaningless — now uses tradable edge proxy
4. **Synthetic orderbook** was 1-unit on v2 — now uses volume_24h/1440 proxy
5. **Eval migrated** from v1 (459 markets) to v2 (181K markets) with stratified sampling
6. **Split flag added** — eval defaults to `--split val` for proper out-of-sample evaluation

**The old results (Sharpe 3.95, PnL $17K) are OBSOLETE.** You must establish a new corrected baseline first.

Start by reading these files:
- `program.md` — the ground truth (READ THIS FIRST)
- `AUDIT_REPORT.md` — what was found and fixed
- `src/polymarket_backtest/strategies.py` — strategy logic
- `src/polymarket_backtest/grid_search.py` — strategy configurations
- `scripts/train_model.py` — model training
- `src/polymarket_backtest/ml_transport.py` — how the model is used
- `src/polymarket_backtest/features.py` — feature extraction

Then follow `program.md` Phase 0 → Phase 1 → Phase 2 → Phase 3.

The eval command is:
```
uv run python scripts/eval_strategies.py --forecast-mode ml_model --max-markets 500 --split val > eval.log 2>&1
```

Go.
