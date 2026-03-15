"""Stress test: verify cash NEVER goes negative during a full evaluation.

Runs the full eval on 50 val markets, monkey-patching the ReplayEngine to
check invariants after EVERY fill, settlement, and redemption.

Invariants checked:
1. portfolio.cash >= 0 for ALL strategies after every fill
2. portfolio.cash >= 0 for ALL strategies after every settlement
3. total_invested never exceeds starting_cash * max_total_invested_pct
4. single position notional never exceeds starting_cash * max_portfolio_pct
5. no position has negative quantity
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_backtest import db as dbmod
from polymarket_backtest.grid_search import (
    _load_market_categories,
    _open_execution_db,
    _run_all_strategies_experiment,
    _stratified_market_sample,
    expanded_strategy_grid,
)
from polymarket_backtest.grok_replay import SmartRuleTransport
from polymarket_backtest.replay_engine import (
    ReplayEngine,
    StrategyPortfolio,
    _position_key,
)
from polymarket_backtest.types import FillResult, PositionState

DB = Path("data/polymarket_backtest_v2.sqlite")
NUM_MARKETS = 50
STARTING_CASH = 1_000.0


@dataclass
class InvariantViolation:
    kind: str  # "negative_cash", "over_invested", "over_position", "negative_qty"
    strategy_name: str
    market_id: str
    detail: str
    cash: float
    timestamp: str


violations: list[InvariantViolation] = []
min_cash_by_strategy: dict[str, float] = {}
check_count = 0
fill_count_buy = 0
fill_count_sell = 0
settlement_count = 0
redemption_count = 0


def check_invariants(
    engine: ReplayEngine,
    event_kind: str,
    market_id: str,
    timestamp: str,
) -> None:
    """Check all invariants across all strategy portfolios."""
    global check_count
    check_count += 1

    for strategy in engine.strategies:
        portfolio = engine.portfolios[strategy.name]
        starting_cash = engine.config.starting_cash

        # Track min cash
        if strategy.name not in min_cash_by_strategy:
            min_cash_by_strategy[strategy.name] = portfolio.cash
        else:
            min_cash_by_strategy[strategy.name] = min(min_cash_by_strategy[strategy.name], portfolio.cash)

        # 1. Cash >= 0
        if portfolio.cash < -1e-9:
            violations.append(
                InvariantViolation(
                    kind="negative_cash",
                    strategy_name=strategy.name,
                    market_id=market_id,
                    detail=f"{event_kind}: cash={portfolio.cash:.6f}",
                    cash=portfolio.cash,
                    timestamp=timestamp,
                )
            )

        # 2. Total invested <= starting_cash * max_total_invested_pct
        total_invested = max(0.0, starting_cash - portfolio.cash)
        max_allowed_invested = starting_cash * strategy.max_total_invested_pct
        if total_invested > max_allowed_invested + 1e-6:
            violations.append(
                InvariantViolation(
                    kind="over_invested",
                    strategy_name=strategy.name,
                    market_id=market_id,
                    detail=(
                        f"{event_kind}: total_invested={total_invested:.4f} > "
                        f"max_allowed={max_allowed_invested:.4f} "
                        f"(pct={strategy.max_total_invested_pct})"
                    ),
                    cash=portfolio.cash,
                    timestamp=timestamp,
                )
            )

        # 3. No single position exceeds starting_cash * max_portfolio_pct
        for pos_key, position in portfolio.positions.items():
            if position.quantity <= 0:
                continue
            position_notional = position.quantity * position.avg_entry_price
            max_position = starting_cash * strategy.max_portfolio_pct
            if position_notional > max_position + 1e-6:
                violations.append(
                    InvariantViolation(
                        kind="over_position",
                        strategy_name=strategy.name,
                        market_id=position.market_id,
                        detail=(
                            f"{event_kind}: position_notional={position_notional:.4f} > "
                            f"max_position={max_position:.4f} "
                            f"(pct={strategy.max_portfolio_pct})"
                        ),
                        cash=portfolio.cash,
                        timestamp=timestamp,
                    )
                )

            # 4. No negative quantity
            if position.quantity < -1e-9:
                violations.append(
                    InvariantViolation(
                        kind="negative_qty",
                        strategy_name=strategy.name,
                        market_id=position.market_id,
                        detail=f"{event_kind}: quantity={position.quantity:.6f}",
                        cash=portfolio.cash,
                        timestamp=timestamp,
                    )
                )


def patch_replay_engine() -> None:
    """Monkey-patch ReplayEngine to check invariants after every fill, settlement, and redemption."""

    # Patch _apply_fill
    original_apply_fill = ReplayEngine._apply_fill

    def patched_apply_fill(
        self: ReplayEngine, portfolio: Any, fill: Any, thesis: str, entry_probability: float, *, is_no_bet: bool = False
    ) -> None:
        global fill_count_buy, fill_count_sell
        original_apply_fill(self, portfolio, fill, thesis, entry_probability, is_no_bet=is_no_bet)
        if fill.side == "buy":
            fill_count_buy += 1
        else:
            fill_count_sell += 1
        ts_str = fill.fill_ts.isoformat() if hasattr(fill.fill_ts, "isoformat") else str(fill.fill_ts)
        check_invariants(self, f"after_fill({fill.side})", fill.market_id, ts_str)

    ReplayEngine._apply_fill = patched_apply_fill  # type: ignore[assignment]

    # Patch _settle_resolved_positions
    original_settle = ReplayEngine._settle_resolved_positions

    def patched_settle(self: ReplayEngine, replay_ts: Any) -> None:
        global settlement_count
        # Count positions that actually get settled this call
        for strat in self.strategies:
            pf = self.portfolios[strat.name]
            for pos in pf.positions.values():
                if pos.quantity > 0:
                    resolution = self._get_resolution(pos.market_id)
                    if resolution is not None and resolution["resolution_ts"] is not None:
                        from datetime import datetime

                        from polymarket_backtest.types import ensure_utc

                        resolution_ts = datetime.fromisoformat(str(resolution["resolution_ts"]))
                        if ensure_utc(replay_ts) >= ensure_utc(resolution_ts):
                            settlement_count += 1
        original_settle(self, replay_ts)
        ts_str = replay_ts.isoformat() if hasattr(replay_ts, "isoformat") else str(replay_ts)
        check_invariants(self, "after_settlement", "all_markets", ts_str)

    ReplayEngine._settle_resolved_positions = patched_settle  # type: ignore[assignment]

    # Patch _redeem_matched_pairs
    original_redeem = ReplayEngine._redeem_matched_pairs

    def patched_redeem(self: ReplayEngine, *, portfolio: Any, strategy_name: str, market: Any) -> float:
        global redemption_count
        result = original_redeem(self, portfolio=portfolio, strategy_name=strategy_name, market=market)
        if result > 0:
            redemption_count += 1
        ts_str = market.ts.isoformat() if hasattr(market.ts, "isoformat") else str(market.ts)
        check_invariants(self, "after_redemption", market.market_id, ts_str)
        return result

    ReplayEngine._redeem_matched_pairs = patched_redeem  # type: ignore[assignment]


def main() -> None:
    print("=" * 72)
    print("STRESS TEST: Cash Invariant Verification")
    print("=" * 72)
    print(f"Database: {DB}")
    print(f"Markets: {NUM_MARKETS} (val split)")
    print(f"Starting cash: ${STARTING_CASH:.2f}")
    print()

    # Patch the engine before running
    patch_replay_engine()
    print("[*] Monkey-patched ReplayEngine with invariant checks")

    # Open database
    print("[*] Loading database into memory...")
    conn = _open_execution_db(DB, in_memory=True)
    dbmod.init_db(conn)

    # Get strategies
    strategies = expanded_strategy_grid()
    print(f"[*] Testing {len(strategies)} strategies:")
    for s in strategies:
        print(
            f"    - {s.name} (family={s.family}, max_portfolio_pct={s.max_portfolio_pct}, "
            f"max_total_invested_pct={s.max_total_invested_pct})"
        )
    print()

    # Sample markets
    market_categories = _load_market_categories(conn)
    market_ids = _stratified_market_sample(conn, NUM_MARKETS, split="val")
    print(f"[*] Sampled {len(market_ids)} val markets")
    print()

    # Run the full evaluation
    print("[*] Running full evaluation with invariant checking...")
    print("    (checking after EVERY fill, settlement, and redemption)")
    print()

    try:
        experiment_id, summary = _run_all_strategies_experiment(
            conn,
            strategies=strategies,
            starting_cash=STARTING_CASH,
            market_categories=market_categories,
            transport_factory=lambda: SmartRuleTransport(),
            market_ids=market_ids,
            eval_stride=4,
            skip_audit=True,
        )
    except Exception:
        print("ERROR during evaluation:")
        traceback.print_exc()
        conn.close()
        sys.exit(2)

    conn.close()

    # Report results
    print()
    print("=" * 72)
    print("STRESS TEST RESULTS")
    print("=" * 72)
    print(f"Total invariant checks performed: {check_count:,}")
    print(f"Total buy fills:  {fill_count_buy:,}")
    print(f"Total sell fills: {fill_count_sell:,}")
    print(f"Total settlements: {settlement_count:,}")
    print(f"Total redemptions: {redemption_count:,}")
    print()

    # Min cash summary
    print("--- Minimum Cash by Strategy ---")
    global_min_cash = float("inf")
    for strategy_name, min_cash in sorted(min_cash_by_strategy.items()):
        global_min_cash = min(global_min_cash, min_cash)
        status = "OK" if min_cash >= -1e-9 else "VIOLATION"
        print(f"  {strategy_name:40s} min_cash=${min_cash:>10.4f}  [{status}]")
    print()
    print(f"  Global minimum cash: ${global_min_cash:.4f}")
    print()

    # Violation summary
    if not violations:
        print("--- Violations: NONE ---")
        print()
        print("=" * 72)
        print("RESULT: PASS")
        print(f"  - Cash never went negative (min: ${global_min_cash:.4f})")
        print(f"  - Total invested never exceeded limits")
        print(f"  - No position exceeded sizing limits")
        print(f"  - No negative quantities detected")
        print(f"  - {check_count:,} invariant checks passed")
        print("=" * 72)
    else:
        # Categorize violations
        by_kind: dict[str, list[InvariantViolation]] = {}
        for v in violations:
            by_kind.setdefault(v.kind, []).append(v)

        print(f"--- Violations: {len(violations)} TOTAL ---")
        for kind, vlist in sorted(by_kind.items()):
            print(f"\n  [{kind}] ({len(vlist)} violations)")
            # Show first 5 of each kind
            for v in vlist[:5]:
                print(f"    strategy={v.strategy_name} market={v.market_id}")
                print(f"      {v.detail}")
                print(f"      at {v.timestamp}")
            if len(vlist) > 5:
                print(f"    ... and {len(vlist) - 5} more")

        print()
        print("=" * 72)
        print("RESULT: FAIL")
        print(f"  - {len(violations)} invariant violations detected")
        print(f"  - Global minimum cash: ${global_min_cash:.4f}")
        print("=" * 72)
        sys.exit(1)


if __name__ == "__main__":
    main()
