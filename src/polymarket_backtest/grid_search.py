import json
import os
import random
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import db

TAGS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "market_tags.json"
from .grok_replay import (
    DeterministicReplayTransport,
    ForecastTransport,
    ReplayGrokClient,
)
from .metrics import build_metrics_summary, persist_metric_results
from .replay_engine import ReplayEngine
from .types import ReplayConfig, StrategyConfig, dc_asdict

DEFAULT_LOOKBACK_MINUTES = 240
DEFAULT_MODEL_RELEASE = os.environ.get("GROK_MODEL_RELEASE", "grok-3")


def expanded_strategy_grid() -> list[StrategyConfig]:
    """Deduplicated strategy grid: 12 materially distinct configs.

    Reduced from 29 configs by removing:
    - Time-horizon gap-fillers that took identical trades (nearterm_extended,
      resolution_medium, resolution_20d, core_medium, core_longterm,
      core_ultralong)
    - Conviction lowodds time variants (near/ext/med/long/ultra all shared
      the same trades as the base conviction_lowodds)
    - nearterm_core (subset of resolution_core)
    - lowodds time variants (longterm/medium/extended/40d — redundant)
    - highodds (superset of highodds_core)
    - conviction_resolution (subset of high_conviction)
    """
    return [
        # 1. Main wide-range control: 30-day window, 500bps edge
        StrategyConfig(
            name="resolution_convergence",
            family="resolution_convergence",
            kelly_fraction=0.40,
            edge_threshold_bps=500.0,
            max_position_notional=1500.0,
            max_holding_minutes=None,
            resolution_hours_max=720.0,
            min_confidence=0.65,
            extreme_low=0.25,
            extreme_high=0.80,
            use_thesis_stop=True,
            thesis_stop_delta=0.10,
        ),
        # 2. Short-term: 7-day window, 150bps edge
        StrategyConfig(
            name="resolution_nearterm",
            family="resolution_convergence",
            kelly_fraction=0.30,
            edge_threshold_bps=150.0,
            max_position_notional=1000.0,
            max_holding_minutes=None,
            resolution_hours_max=168.0,
            min_confidence=0.65,
            extreme_low=0.25,
            extreme_high=0.80,
            use_thesis_stop=True,
            thesis_stop_delta=0.09,
        ),
        # 3. Ultra-short: 3-day window, 300bps edge
        StrategyConfig(
            name="resolution_72h",
            family="resolution_convergence",
            kelly_fraction=0.30,
            edge_threshold_bps=300.0,
            max_position_notional=1000.0,
            max_holding_minutes=None,
            resolution_hours_max=72.0,
            min_confidence=0.65,
            extreme_low=0.25,
            extreme_high=0.80,
            use_thesis_stop=True,
            thesis_stop_delta=0.10,
        ),
        # 4. Long-term: 60-day window, 700bps edge
        StrategyConfig(
            name="resolution_longterm",
            family="resolution_convergence",
            kelly_fraction=0.40,
            edge_threshold_bps=700.0,
            max_position_notional=1500.0,
            max_holding_minutes=None,
            resolution_hours_max=1440.0,
            min_confidence=0.65,
            extreme_low=0.25,
            extreme_high=0.80,
            use_thesis_stop=True,
            thesis_stop_delta=0.10,
        ),
        # 5. Ultra-long: 90-day window, 1000bps edge
        StrategyConfig(
            name="resolution_ultralong",
            family="resolution_convergence",
            kelly_fraction=0.35,
            edge_threshold_bps=1000.0,
            max_position_notional=1500.0,
            max_holding_minutes=None,
            resolution_hours_max=2160.0,
            min_confidence=0.65,
            extreme_low=0.25,
            extreme_high=0.75,
            use_thesis_stop=True,
            thesis_stop_delta=0.12,
        ),
        # 6. Core mid-range: aggressive sizing on 0.30-0.70 markets
        StrategyConfig(
            name="resolution_core",
            family="resolution_convergence",
            kelly_fraction=1.00,
            edge_threshold_bps=20.0,
            max_position_notional=3500.0,
            max_holding_minutes=None,
            resolution_hours_max=720.0,
            min_confidence=0.65,
            extreme_low=0.30,
            extreme_high=0.70,
            use_thesis_stop=True,
            thesis_stop_delta=0.10,
        ),
        # 7. Core mega: mid-range at 6-month horizon, high edge
        StrategyConfig(
            name="core_mega",
            family="resolution_convergence",
            kelly_fraction=0.40,
            edge_threshold_bps=500.0,
            max_position_notional=1500.0,
            max_holding_minutes=None,
            resolution_hours_max=4320.0,
            min_confidence=0.65,
            extreme_low=0.30,
            extreme_high=0.70,
            use_thesis_stop=True,
            thesis_stop_delta=0.10,
        ),
        # 8. Low-odds underdogs: 0.15-0.80 with high edge
        StrategyConfig(
            name="resolution_lowodds",
            family="resolution_convergence",
            kelly_fraction=0.30,
            edge_threshold_bps=800.0,
            max_position_notional=1000.0,
            max_holding_minutes=None,
            resolution_hours_max=720.0,
            min_confidence=0.65,
            extreme_low=0.15,
            extreme_high=0.80,
            use_thesis_stop=True,
            thesis_stop_delta=0.10,
        ),
        # 9. Edge-based high conviction: high confidence + pure edge
        StrategyConfig(
            name="edge_conviction",
            family="edge_based",
            kelly_fraction=0.40,
            edge_threshold_bps=300.0,
            max_position_notional=1500.0,
            max_holding_minutes=10080,
            min_confidence=0.75,
            use_thesis_stop=True,
            thesis_stop_delta=0.12,
            aggressive_entry=True,
        ),
        # 10. High conviction: high confidence, wide range
        StrategyConfig(
            name="high_conviction",
            family="resolution_convergence",
            kelly_fraction=1.00,
            edge_threshold_bps=100.0,
            max_position_notional=3500.0,
            max_holding_minutes=None,
            resolution_hours_max=720.0,
            min_confidence=0.75,
            extreme_low=0.25,
            extreme_high=0.80,
            use_thesis_stop=True,
            thesis_stop_delta=0.12,
        ),
        # 11. Edge-based moderate: balanced edge + sizing
        StrategyConfig(
            name="edge_moderate",
            family="edge_based",
            kelly_fraction=0.20,
            edge_threshold_bps=700.0,
            max_position_notional=800.0,
            max_holding_minutes=10080,
            min_confidence=0.65,
            use_thesis_stop=True,
            thesis_stop_delta=0.10,
            aggressive_entry=True,
        ),
        # 12. Edge-based: pure forecast edge, no resolution context
        StrategyConfig(
            name="edge_based",
            family="edge_based",
            kelly_fraction=0.30,
            edge_threshold_bps=500.0,
            max_position_notional=1000.0,
            max_holding_minutes=10080,
            min_confidence=0.65,
            use_thesis_stop=True,
            thesis_stop_delta=0.10,
            aggressive_entry=True,
        ),
    ]


def _stratified_market_sample(
    conn: sqlite3.Connection,
    max_markets: int,
    *,
    min_snapshots: int = 20,
    max_snapshots: int = 2000,
    seed: int = 42,
    split: str | None = None,
) -> list[str]:
    """Sample markets with stratified random selection by duration bucket.

    Args:
        split: If set, restrict to markets in this chronological split:
            "train" (resolved < 2025-10-01), "val" (Oct-Dec 2025),
            "test" (resolved >= 2026-01-01), or None (all markets).
    """
    split_filter = ""
    if split == "train":
        split_filter = "AND m.resolution_ts < '2025-10-01'"
    elif split == "val":
        split_filter = "AND m.resolution_ts >= '2025-10-01' AND m.resolution_ts < '2026-01-01'"
    elif split == "test":
        split_filter = "AND m.resolution_ts >= '2026-01-01'"

    # Two-phase approach to avoid GROUP BY on 68M rows (63s → <1s):
    # Phase 1: Get candidate market_ids from small tables (markets + resolutions)
    candidates = conn.execute(
        f"""
        SELECT m.market_id
        FROM markets m
        JOIN market_resolutions mr ON mr.market_id = m.market_id
        {("WHERE 1=1 " + split_filter) if split_filter else ""}
        """,
    ).fetchall()
    candidate_ids = [str(r["market_id"]) for r in candidates]

    if not candidate_ids:
        return []

    # Phase 2: Get snapshot counts via indexed lookups (much faster than GROUP BY)
    rows = []
    for mid in candidate_ids:
        row = conn.execute(
            "SELECT COUNT(*) as cnt, MIN(ts) as first_ts, MAX(ts) as last_ts FROM market_snapshots WHERE market_id = ?",
            (mid,),
        ).fetchone()
        cnt = int(row["cnt"])
        if min_snapshots <= cnt <= max_snapshots:
            rows.append({"market_id": mid, "cnt": cnt, "first_ts": row["first_ts"], "last_ts": row["last_ts"]})

    if not rows:
        return []

    # Bucket by market duration
    buckets: dict[str, list[str]] = {
        "short": [],  # < 7 days
        "medium": [],  # 7-30 days
        "long": [],  # 30-90 days
        "very_long": [],  # 90+ days
    }
    for row in rows:
        market_id = str(row["market_id"])
        cnt = int(row["cnt"])
        # Approximate duration from snapshot count (hourly snapshots)
        duration_days = cnt / 24.0
        if duration_days < 7:
            buckets["short"].append(market_id)
        elif duration_days < 30:
            buckets["medium"].append(market_id)
        elif duration_days < 90:
            buckets["long"].append(market_id)
        else:
            buckets["very_long"].append(market_id)

    # Proportional allocation with minimum 1 per non-empty bucket
    non_empty = {k: v for k, v in buckets.items() if v}
    total_available = sum(len(v) for v in non_empty.values())
    if total_available == 0:
        return []

    rng = random.Random(seed)
    selected: list[str] = []
    remaining = max_markets

    for bucket_name, bucket_markets in non_empty.items():
        proportion = len(bucket_markets) / total_available
        n = max(1, int(proportion * max_markets))
        n = min(n, len(bucket_markets), remaining)
        if n <= 0:
            continue
        sample = rng.sample(bucket_markets, n)
        selected.extend(sample)
        remaining -= n

    # Fill remaining slots from largest bucket
    if remaining > 0:
        largest = max(non_empty.values(), key=len)
        available = [m for m in largest if m not in set(selected)]
        extra = rng.sample(available, min(remaining, len(available)))
        selected.extend(extra)

    rng.shuffle(selected)
    return selected


def _open_execution_db(db_path: Path, *, in_memory: bool) -> sqlite3.Connection:
    """Open a DB connection for grid search execution.

    When ``in_memory=True``, copies the on-disk DB into a ``:memory:``
    connection.  All reads AND writes happen in RAM — no disk I/O during
    the replay.  The in-memory copy is discarded when the connection closes.
    """
    if not in_memory:
        return db.connect(db_path)

    # Open source read-only, copy into memory
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    mem_conn = sqlite3.connect(":memory:")
    source.backup(mem_conn)
    source.close()

    # Apply the same pragmas as db.connect()
    mem_conn.row_factory = sqlite3.Row
    mem_conn.execute("PRAGMA foreign_keys = ON")
    return mem_conn


def run_grid_search(
    db_path: Path,
    *,
    strategies: list[StrategyConfig] | None = None,
    starting_cash: float = 1_000.0,
    transport_factory: Callable[[], ForecastTransport] | None = None,
    max_markets: int | None = None,
    in_memory: bool = True,
    exclude_categories: list[str] | None = None,
    split: str | None = None,
    eval_stride: int = 4,
    n_workers: int = 1,
    transport_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Run a grid search across strategies.

    Args:
        db_path: Path to the SQLite database.
        strategies: Override the default strategy grid.
        starting_cash: Starting cash per strategy.
        transport_factory: Optional callable returning a ``ForecastTransport``.
            Defaults to ``DeterministicReplayTransport`` for backward compatibility.
        max_markets: Limit the number of markets used (for faster iteration).
        in_memory: Copy the DB into RAM before replaying.  Eliminates all
            disk I/O during the replay.  Requires ~2-3 GB RAM.  Defaults to True.
        n_workers: Number of parallel worker processes. 1 = sequential (default),
            0 = auto (cpu_count). Values > 1 enable multiprocessing.
        transport_mode: Transport mode string for parallel workers (e.g. "smart_rules").
            Required when n_workers != 1. Ignored in sequential mode.
    """
    selected_strategies = expanded_strategy_grid() if strategies is None else strategies
    if not selected_strategies:
        return []

    # For parallel mode, we need a disk-based DB (workers read from disk via WAL)
    use_parallel = n_workers != 1
    if use_parallel:
        in_memory = False  # Workers read from disk

    conn = _open_execution_db(db_path, in_memory=in_memory)
    try:
        db.init_db(conn)

        # Determine which market IDs to replay (stratified random sampling)
        market_ids: list[str] | None = None
        if max_markets is not None:
            market_ids = _stratified_market_sample(conn, max_markets, split=split)
            if not market_ids:
                raise ValueError(f"No market data found in {db_path}")

        # Filter by category if requested
        if exclude_categories and market_ids is not None and TAGS_PATH.exists():
            with open(TAGS_PATH) as f:
                market_tags = json.load(f)
            exclude_set = set(exclude_categories)
            before = len(market_ids)
            market_ids = [
                mid
                for mid in market_ids
                if (market_tags.get(mid, ["Unknown"])[0] if market_tags.get(mid) else "Unknown") not in exclude_set
            ]
            print(f"  Category filter: {before} → {len(market_ids)} markets (excluded {exclude_set})")
        elif exclude_categories and market_ids is None:
            # Need to get all market IDs first to filter
            all_ids = [
                str(row["market_id"])
                for row in conn.execute("SELECT DISTINCT market_id FROM market_snapshots").fetchall()
            ]
            if TAGS_PATH.exists():
                with open(TAGS_PATH) as f:
                    market_tags = json.load(f)
                exclude_set = set(exclude_categories)
                market_ids = [
                    mid
                    for mid in all_ids
                    if (market_tags.get(mid, ["Unknown"])[0] if market_tags.get(mid) else "Unknown") not in exclude_set
                ]
                print(f"  Category filter: {len(all_ids)} → {len(market_ids)} markets (excluded {exclude_set})")

        # Use parallel path when requested and we have market_ids
        if use_parallel and market_ids is not None and len(market_ids) > 1:
            from .parallel_eval import run_parallel_grid_search

            resolved_mode = transport_mode or "smart_rules"
            experiment_id, summary = run_parallel_grid_search(
                db_path,
                conn,
                strategies=selected_strategies,
                starting_cash=starting_cash,
                transport_mode=resolved_mode,
                market_ids=market_ids,
                eval_stride=eval_stride,
                n_workers=n_workers,
            )
        else:
            # Single-pass: run ALL strategies in one ReplayEngine pass.
            # The engine processes each market event once and applies all strategies,
            # avoiding 13x redundant forecast computation and DB reads.
            experiment_id, summary = _run_all_strategies_experiment(
                conn,
                strategies=selected_strategies,
                starting_cash=starting_cash,
                transport_factory=transport_factory,
                market_ids=market_ids,
                eval_stride=eval_stride,
                skip_audit=True,
            )
        results: list[dict[str, Any]] = []
        for strategy in selected_strategies:
            results.append(_extract_strategy_result(conn, experiment_id, strategy.name, summary))
        return results
    finally:
        conn.close()


def rank_strategies(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []

    pnl_scores = _normalize_metric(results, "pnl")
    sharpe_scores = _normalize_metric(results, "sharpe")
    brier_scores = _normalize_metric(results, "brier_improvement")

    ranked: list[dict[str, Any]] = []
    for result in results:
        strategy_name = str(result["strategy_name"])
        composite_score = (
            0.4 * pnl_scores[strategy_name] + 0.3 * sharpe_scores[strategy_name] + 0.3 * brier_scores[strategy_name]
        )
        ranked.append(
            {
                **result,
                "composite_score": round(composite_score, 6),
                "is_profitable": float(result["pnl"]) > 0,
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item["composite_score"]),
            -float(item["pnl"]),
            -float(item["sharpe"]),
            -float(item["brier_improvement"]),
            str(item["strategy_name"]),
        )
    )
    for index, result in enumerate(ranked, start=1):
        result["rank"] = index
    return ranked


def build_grid_report(ranked_results: list[dict[str, Any]]) -> str:
    if not ranked_results:
        return "# Grid Search Report\n\nNo strategies were evaluated."

    best = ranked_results[0]
    best_sharpe = max(ranked_results, key=lambda item: (float(item["sharpe"]), float(item["pnl"])))
    best_brier = max(ranked_results, key=lambda item: (float(item["brier_improvement"]), float(item["pnl"])))
    most_active = max(ranked_results, key=lambda item: (int(item["n_trades"]), float(item["fill_ratio"])))

    lines = [
        "# Grid Search Report",
        "",
        "## Summary",
        (
            f"Best overall strategy: **{best['strategy_name']}** "
            f"(score={best['composite_score']:.4f}, pnl={best['pnl']:+.2f}, "
            f"sharpe={best['sharpe']:.4f}, brier_improvement={best['brier_improvement']:+.4f})"
        ),
        "",
        "## Strategy Comparison",
        "| Rank | Strategy | PnL (Hold) | PnL (MtM) | Sharpe | Brier Improvement | Trades | Fill Ratio | Composite |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in ranked_results:
        strategy_name = str(result["strategy_name"])
        strategy_label = f"**{strategy_name}**" if int(result["rank"]) == 1 else strategy_name
        pnl_mtm = float(result.get("pnl_mark_to_market", 0.0))
        lines.append(
            "| "
            f"{result['rank']} | "
            f"{strategy_label} | "
            f"{float(result['pnl']):+.2f} | "
            f"{pnl_mtm:+.2f} | "
            f"{float(result['sharpe']):.4f} | "
            f"{float(result['brier_improvement']):+.4f} | "
            f"{int(result['n_trades'])} | "
            f"{float(result['fill_ratio']):.2%} | "
            f"{float(result['composite_score']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            (
                f"- Use **{best['strategy_name']}** as the default candidate for follow-up analysis. "
                "It leads on the weighted composite of PnL, Sharpe-like return, and Brier improvement."
            ),
            (
                f"- Monitor **{best_sharpe['strategy_name']}** if risk-adjusted stability matters more than raw PnL. "
                f"It has the strongest Sharpe-like score at {best_sharpe['sharpe']:.4f}."
            ),
            (
                f"- Treat **{best_brier['strategy_name']}** as the forecast-quality leader. "
                f"It delivers the strongest average Brier improvement at {best_brier['brier_improvement']:+.4f}."
            ),
            (
                f"- Check execution assumptions on **{most_active['strategy_name']}**, "
                f"which generated the most filled trades ({most_active['n_trades']}) with "
                f"a {most_active['fill_ratio']:.2%} fill ratio."
            ),
        ]
    )

    profitable = [r for r in ranked_results if float(r["pnl"]) > 0]
    if not profitable:
        lines.append("")
        lines.append("## WARNING: NO PROFITABLE STRATEGY")
        lines.append("All strategies lost money in this backtest. Do NOT deploy any strategy.")
        lines.append("")

    return "\n".join(lines)


def run_full_grid_search(db_path: Path) -> str:
    results = run_grid_search(db_path)
    ranked_results = rank_strategies(results)
    return build_grid_report(ranked_results)


def _build_experiment_name(strategy_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"grid_{strategy_name}_{timestamp}"


def _extract_strategy_result(
    conn: sqlite3.Connection,
    experiment_id: int,
    strategy_name: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    pnl_hold_by_strategy = {item["strategy_name"]: float(item["pnl_hold_to_resolution"]) for item in summary["pnl"]}
    # Mark-to-market PnL from trade-level details
    pnl_mtm_by_strategy: dict[str, float] = {}
    for item in summary.get("trade_pnl", []):
        name = str(item["strategy_name"])
        pnl_mtm_by_strategy[name] = pnl_mtm_by_strategy.get(name, 0.0) + float(item.get("pnl", 0.0))
    sharpe_by_strategy = {item["strategy_name"]: float(item["sharpe_like"]) for item in summary["sharpe_like"]}
    trade_row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM orders
        WHERE experiment_id = ? AND strategy_name = ? AND filled_quantity > 0
        """,
        (experiment_id, strategy_name),
    ).fetchone()
    brier_improvements = [float(item["brier_improvement"]) for item in summary["brier_comparison"]]
    return {
        "strategy_name": strategy_name,
        "pnl": round(pnl_hold_by_strategy.get(strategy_name, 0.0), 4),
        "pnl_mark_to_market": round(pnl_mtm_by_strategy.get(strategy_name, 0.0), 4),
        "sharpe": round(sharpe_by_strategy.get(strategy_name, 0.0), 6),
        "brier_improvement": round(_mean(brier_improvements), 6),
        "n_trades": int(trade_row["n"] or 0),
        "fill_ratio": round(float(summary["fill_ratio"]["fill_ratio"]), 6),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize_metric(results: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(result[key]) for result in results]
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return {str(result["strategy_name"]): 0.5 for result in results}
    return {str(result["strategy_name"]): (float(result[key]) - minimum) / (maximum - minimum) for result in results}


def _run_all_strategies_experiment(
    conn: sqlite3.Connection,
    *,
    eval_stride: int = 1,
    skip_audit: bool = False,
    strategies: list[StrategyConfig],
    starting_cash: float,
    transport_factory: Callable[[], ForecastTransport] | None = None,
    market_ids: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run ALL strategies in a single ReplayEngine pass.

    This is ~13x faster than running each strategy separately because:
    - Forecasts are computed once per (market_id, timestamp), not per strategy
    - Market state DB reads happen once, not per strategy
    - Context building (news, related markets) happens once
    """
    config = ReplayConfig(
        experiment_name=_build_experiment_name("grid_all"),
        starting_cash=starting_cash,
        lookback_minutes=DEFAULT_LOOKBACK_MINUTES,
        eval_stride=eval_stride,
    )
    transport = transport_factory() if transport_factory is not None else DeterministicReplayTransport(model_id="grok")
    model_id = getattr(transport, "model_id", "grok")
    # Need a timestamp for prompt_hash — use a sentinel
    prompt_hash_ts = datetime.now(UTC)
    grok = ReplayGrokClient(
        conn=conn,
        experiment_id=None,
        model_id=model_id,
        model_release=DEFAULT_MODEL_RELEASE,
        transport=transport,
        lookback_minutes=config.lookback_minutes,
    )
    experiment_id = db.create_experiment(
        conn,
        name=config.experiment_name,
        model_id=grok.model_id,
        model_release=grok.model_release,
        system_prompt_hash=grok.prompt_hash(prompt_hash_ts),
        config={
            "starting_cash": config.starting_cash,
            "lookback_minutes": config.lookback_minutes,
            "markout_horizons_min": config.markout_horizons_min,
            "eval_stride": config.eval_stride,
            "grid_search": True,
            "strategies": [dc_asdict(s) for s in strategies],
        },
    )
    grok.experiment_id = experiment_id
    grok.context_builder.experiment_id = experiment_id
    grok.context_builder.skip_audit = skip_audit
    grok.context_builder.skip_related_markets = skip_audit  # skip related markets when skipping audit (eval mode)

    engine = ReplayEngine(
        conn=conn,
        config=config,
        grok=grok,
        strategies=strategies,
    )
    if market_ids is not None:
        engine.run_markets(market_ids)
    else:
        engine.run()
    persist_metric_results(conn, experiment_id, config.markout_horizons_min)
    return experiment_id, build_metrics_summary(conn, experiment_id, config.markout_horizons_min)


def _run_strategy_experiment(
    conn: sqlite3.Connection,
    *,
    strategy: StrategyConfig,
    replay_timestamps: list[datetime],
    starting_cash: float,
    transport_factory: Callable[[], ForecastTransport] | None = None,
) -> tuple[int, dict[str, Any]]:
    config = ReplayConfig(
        experiment_name=_build_experiment_name(strategy.name),
        starting_cash=starting_cash,
        lookback_minutes=DEFAULT_LOOKBACK_MINUTES,
    )
    transport = transport_factory() if transport_factory is not None else DeterministicReplayTransport(model_id="grok")
    model_id = getattr(transport, "model_id", "grok")
    grok = ReplayGrokClient(
        conn=conn,
        experiment_id=None,
        model_id=model_id,
        model_release=DEFAULT_MODEL_RELEASE,
        transport=transport,
        lookback_minutes=config.lookback_minutes,
    )
    experiment_id = db.create_experiment(
        conn,
        name=config.experiment_name,
        model_id=grok.model_id,
        model_release=grok.model_release,
        system_prompt_hash=grok.prompt_hash(replay_timestamps[0]),
        config={
            "starting_cash": config.starting_cash,
            "lookback_minutes": config.lookback_minutes,
            "markout_horizons_min": config.markout_horizons_min,
            "grid_search": True,
            "strategy": dc_asdict(strategy),
        },
    )
    grok.experiment_id = experiment_id
    grok.context_builder.experiment_id = experiment_id

    engine = ReplayEngine(
        conn=conn,
        config=config,
        grok=grok,
        strategies=[strategy],
    )
    engine.run()
    persist_metric_results(conn, experiment_id, config.markout_horizons_min)
    return experiment_id, build_metrics_summary(conn, experiment_id, config.markout_horizons_min)
