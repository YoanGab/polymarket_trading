"""Benchmark Wave 1 optimizations."""

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.ERROR)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_backtest.grid_search import rank_strategies, run_grid_search
from polymarket_backtest.ml_transport import MLModelTransport

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"

max_markets = int(sys.argv[1]) if len(sys.argv) > 1 else 10
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 4

print(f"Benchmark: {max_markets} markets, stride={stride}")
start = time.monotonic()

results = run_grid_search(
    DB_PATH,
    max_markets=max_markets,
    transport_factory=lambda: MLModelTransport(),
    in_memory=True,
    split="val",
    eval_stride=stride,
)
ranked = rank_strategies(results)
elapsed = time.monotonic() - start

print(f"\nResults ({elapsed:.1f}s):")
total_trades = 0
for r in ranked:
    trades = int(r["n_trades"])
    total_trades += trades
    if trades > 0:
        print(f"  {r['strategy_name']:30s} PnL={r['pnl']:+8.2f} Sharpe={r['sharpe']:8.4f} trades={trades}")

print(f"\nTotal trades: {total_trades}")
print(f"Elapsed: {elapsed:.1f}s")
