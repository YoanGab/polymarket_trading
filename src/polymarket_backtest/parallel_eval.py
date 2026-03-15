"""Multiprocessing support for parallel market evaluation.

Splits market_ids into N groups and runs each group in a separate process.
Each worker opens its own disk-based DB connection.  WAL mode (already enabled
by ``db.connect``) allows concurrent readers AND writers without locking
contention -- readers never block writers, writers only block other writers for
the brief moment of page commit.

Workers all share the same experiment_id (created by the main process) and
write disjoint market_ids, so there is no row-level conflict.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import db
from .grok_replay import (
    DeterministicReplayTransport,
    ForecastTransport,
    ReplayGrokClient,
)
from .metrics import build_metrics_summary, persist_metric_results
from .replay_engine import ReplayEngine
from .types import ReplayConfig, StrategyConfig, dc_asdict

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_MINUTES = 240
DEFAULT_MODEL_RELEASE = os.environ.get("GROK_MODEL_RELEASE", "grok-3")


def _split_market_ids(market_ids: list[str], n_groups: int) -> list[list[str]]:
    """Split market_ids into n_groups roughly equal batches."""
    if n_groups <= 0:
        n_groups = 1
    n_groups = min(n_groups, len(market_ids))
    if n_groups <= 0:
        return []
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    for i, mid in enumerate(market_ids):
        groups[i % n_groups].append(mid)
    return [g for g in groups if g]


def _worker_run_markets(args: tuple[Any, ...]) -> dict[str, Any]:
    """Worker function for parallel market processing.

    Runs in a child process.  Opens its own disk-based DB connection (WAL
    mode enables concurrent reads and writes).  All output rows are written
    directly to the on-disk DB under the shared experiment_id.

    Args is a tuple of:
        (db_path, market_ids, strategy_dicts, starting_cash, transport_mode,
         eval_stride, experiment_id, worker_index, market_categories)
    """
    (
        db_path,
        market_ids,
        strategy_dicts,
        starting_cash,
        transport_mode,
        eval_stride,
        experiment_id,
        worker_index,
        market_categories,
    ) = args

    worker_name = f"worker-{worker_index}"
    print(f"[{worker_name}] Starting with {len(market_ids)} markets")

    # Reconstruct strategies from dicts (StrategyConfig is frozen dataclass).
    # ty cannot verify **dict unpacking statically, but dc_asdict() guarantees
    # all required fields are present.
    strategies = [StrategyConfig(**d) for d in strategy_dicts]  # type: ignore[missing-argument]

    # Open disk-based connection (WAL allows concurrent reads + writes)
    conn = db.connect(db_path)

    # Create the transport
    transport: ForecastTransport
    if transport_mode == "ml_model":
        from .ml_transport import MLModelTransport

        transport = MLModelTransport()
    elif transport_mode == "smart_rules":
        from .grok_replay import SmartRuleTransport

        transport = SmartRuleTransport()
    else:
        transport = DeterministicReplayTransport(model_id="grok")

    model_id = getattr(transport, "model_id", "grok")

    config = ReplayConfig(
        experiment_name=f"grid_parallel_{worker_name}",
        starting_cash=starting_cash,
        lookback_minutes=DEFAULT_LOOKBACK_MINUTES,
        eval_stride=eval_stride,
    )

    grok = ReplayGrokClient(
        conn=conn,
        experiment_id=experiment_id,
        model_id=model_id,
        model_release=DEFAULT_MODEL_RELEASE,
        transport=transport,
        lookback_minutes=config.lookback_minutes,
    )
    grok.context_builder.skip_audit = True
    grok.context_builder.skip_related_markets = True

    engine = ReplayEngine(
        conn=conn,
        config=config,
        grok=grok,
        strategies=strategies,
        market_categories=market_categories,
    )
    engine.run_markets(market_ids)

    conn.close()

    print(f"[{worker_name}] Finished processing {len(market_ids)} markets")

    return {
        "n_markets": len(market_ids),
    }


def run_parallel_grid_search(
    db_path: Path,
    conn: sqlite3.Connection,
    *,
    strategies: list[StrategyConfig],
    starting_cash: float,
    transport_mode: str,
    market_ids: list[str],
    market_categories: dict[str, list[str]],
    eval_stride: int = 4,
    n_workers: int = 0,
) -> tuple[int, dict[str, Any]]:
    """Run grid search with multiprocessing.

    Args:
        db_path: Path to the on-disk SQLite database.
        conn: Main process connection (for creating experiment and computing
            metrics after workers finish).  Must be a disk-based connection
            (not in-memory).
        strategies: Strategy configs to evaluate.
        starting_cash: Starting cash per strategy.
        transport_mode: Transport mode string (e.g. "smart_rules").
        market_ids: List of market IDs to process.
        eval_stride: Evaluate every Nth snapshot.
        n_workers: Number of worker processes. 0 = auto (cpu_count).

    Returns:
        (experiment_id, metrics_summary)
    """
    if n_workers <= 0:
        n_workers = os.cpu_count() or 8

    # Cap workers to number of markets
    n_workers = min(n_workers, len(market_ids))

    # Create experiment in main process
    from .grok_replay import build_temporal_system_prompt, stable_hash

    transport: ForecastTransport
    if transport_mode == "ml_model":
        from .ml_transport import MLModelTransport

        transport = MLModelTransport()
    elif transport_mode == "smart_rules":
        from .grok_replay import SmartRuleTransport

        transport = SmartRuleTransport()
    else:
        transport = DeterministicReplayTransport(model_id="grok")

    model_id = getattr(transport, "model_id", "grok")
    prompt_hash_ts = datetime.now(UTC)
    prompt_hash = stable_hash(build_temporal_system_prompt(prompt_hash_ts.isoformat()))

    config = ReplayConfig(
        experiment_name=f"grid_parallel_all_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        starting_cash=starting_cash,
        lookback_minutes=DEFAULT_LOOKBACK_MINUTES,
        eval_stride=eval_stride,
    )

    experiment_id = db.create_experiment(
        conn,
        name=config.experiment_name,
        model_id=model_id,
        model_release=DEFAULT_MODEL_RELEASE,
        system_prompt_hash=prompt_hash,
        config={
            "starting_cash": config.starting_cash,
            "lookback_minutes": config.lookback_minutes,
            "markout_horizons_min": config.markout_horizons_min,
            "eval_stride": config.eval_stride,
            "grid_search": True,
            "parallel_workers": n_workers,
            "strategies": [dc_asdict(s) for s in strategies],
        },
    )

    # Serialize strategy configs as dicts
    strategy_dicts = [dc_asdict(s) for s in strategies]

    # Split markets into groups
    groups = _split_market_ids(market_ids, n_workers)
    actual_workers = len(groups)

    print(f"[Parallel] Spawning {actual_workers} workers for {len(market_ids)} markets")

    # Build args for each worker
    worker_args = [
        (
            str(db_path),
            group,
            strategy_dicts,
            starting_cash,
            transport_mode,
            eval_stride,
            experiment_id,
            i,
            {mid: market_categories.get(mid, []) for mid in group},
        )
        for i, group in enumerate(groups)
    ]

    # Ensure experiment row is committed so workers can see it
    conn.commit()

    # Run workers (each opens its own connection)
    ctx = mp.get_context("spawn")  # spawn is safest for SQLite
    with ctx.Pool(processes=actual_workers) as pool:
        results = pool.map(_worker_run_markets, worker_args)

    total_markets = sum(r["n_markets"] for r in results)
    print(f"[Parallel] Workers done. Processed {total_markets} markets across {actual_workers} workers")

    # Workers wrote to the on-disk DB.  The main conn (also on-disk, WAL mode)
    # needs to start a fresh read transaction to see the workers' committed
    # writes.  Executing any statement implicitly starts a new transaction.
    # Force a checkpoint to merge WAL into main DB file first.
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    # Compute metrics on merged data
    persist_metric_results(conn, experiment_id, config.markout_horizons_min)
    summary = build_metrics_summary(conn, experiment_id, config.markout_horizons_min)

    print(f"[Parallel] Done. Metrics computed for experiment {experiment_id}")

    return experiment_id, summary
