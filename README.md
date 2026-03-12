# Leakage-Safe Polymarket Backtesting

This repository contains an MVP architecture for backtesting a Polymarket bot
that uses Grok as the forecasting engine without temporal leakage.

Artifacts:

- `docs/backtesting_architecture.md`: full architecture, assumptions, ASCII
  diagram, leakage controls, metrics, statistical tests, and SQLite schema.
- `src/polymarket_backtest/`: runnable stdlib-only reference implementation.

Run the demo replay:

```bash
uv run polymarket-backtest
```

The demo seeds a small SQLite database, runs two example strategies, and prints
an experiment report.
