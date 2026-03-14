"""Fast strategy evaluation for autoresearch loop.

Usage:
    uv run python scripts/eval_strategies.py [--max-markets 100]

Runs grid search on the existing DB, prints a TSV-parseable results line.
Uses --max-markets to limit data for fast iteration (default: 100 markets).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_backtest.grid_search import (
    build_grid_report,
    rank_strategies,
    run_grid_search,
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"
RESULTS_TSV = Path(__file__).resolve().parent.parent / "results.tsv"


def _make_transport_factory(mode: str):
    """Build a transport factory from a CLI mode string."""
    from polymarket_backtest.grok_replay import create_transport

    if mode == "ml_model":
        # ML model transport — loads trained model from models/
        from polymarket_backtest.ml_transport import MLModelTransport

        return lambda: MLModelTransport()

    def factory():
        return create_transport(mode=mode, model_release="grok-3")

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strategies via grid search")
    parser.add_argument("--max-markets", type=int, default=100, help="Max markets to use (default: 100)")
    parser.add_argument(
        "--forecast-mode",
        choices=["deterministic", "smart_rules", "ml_model"],
        default="smart_rules",
        help="Forecast transport to use (default: smart_rules)",
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=None,
        help="Exclude markets in this category (can repeat). E.g., --exclude-category Sports",
    )
    parser.add_argument(
        "--no-in-memory",
        action="store_true",
        default=False,
        help="Disable in-memory DB copy (use for very large DBs)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Chronological split to evaluate on (default: val). Use 'val' for tuning, 'test' for final eval only.",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print("ERROR: Database not found. Run download_data.py first.", file=sys.stderr)
        sys.exit(1)

    transport_factory = _make_transport_factory(args.forecast_mode)

    start = time.monotonic()
    results = run_grid_search(
        DB_PATH,
        max_markets=args.max_markets,
        transport_factory=transport_factory,
        exclude_categories=args.exclude_category,
        in_memory=not args.no_in_memory,
        split=args.split,
    )
    ranked = rank_strategies(results)
    elapsed = time.monotonic() - start

    if not ranked:
        print("ERROR: No strategies evaluated.", file=sys.stderr)
        sys.exit(1)

    best = ranked[0]

    # Print machine-readable summary line
    print(
        f"BEST_STRATEGY={best['strategy_name']}"
        f"\tCOMPOSITE={best['composite_score']:.6f}"
        f"\tPNL={best['pnl']:+.2f}"
        f"\tSHARPE={best['sharpe']:.4f}"
        f"\tBRIER_IMPROVEMENT={best['brier_improvement']:+.4f}"
        f"\tTRADES={best['n_trades']}"
        f"\tFILL_RATIO={best['fill_ratio']:.4f}"
        f"\tELAPSED={elapsed:.1f}s"
        f"\tMARKETS={args.max_markets}"
        f"\tFORECAST={args.forecast_mode}"
    )

    # Append to results.tsv
    header_needed = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a") as f:
        if header_needed:
            f.write("timestamp\tbest_strategy\tcomposite_score\tpnl\tsharpe\tbrier_improvement\tn_trades\telapsed_s\n")
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t"
            f"{best['strategy_name']}\t"
            f"{best['composite_score']:.6f}\t"
            f"{best['pnl']:+.2f}\t"
            f"{best['sharpe']:.4f}\t"
            f"{best['brier_improvement']:+.4f}\t"
            f"{best['n_trades']}\t"
            f"{elapsed:.1f}\n"
        )

    # Print full ranking for human review
    print(f"\nAll strategies ({elapsed:.1f}s, {args.max_markets} markets):")
    for r in ranked:
        print(
            f"  {r['strategy_name']:25s}  "
            f"PnL={r['pnl']:+8.2f}  "
            f"Sharpe={r['sharpe']:.4f}  "
            f"Brier={r['brier_improvement']:+.4f}  "
            f"trades={r['n_trades']:5d}  "
            f"composite={r['composite_score']:.4f}"
        )

    # Print grid report
    print("\n" + build_grid_report(ranked))

    sys.exit(0)


if __name__ == "__main__":
    main()
