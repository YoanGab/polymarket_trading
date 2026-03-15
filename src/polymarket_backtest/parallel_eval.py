"""Multiprocessing support for parallel market evaluation.

Splits market_ids into N groups and runs each group in a separate process.
Each worker opens its own disk-based read-only DB connection (WAL mode allows
concurrent readers) and writes output to its own in-memory DB.  The main
process merges worker results into the primary connection.

This avoids the 8x20GB RAM explosion that would occur if each worker copied
the full DB into memory.
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

# Output tables that workers write to and that need merging.
_OUTPUT_TABLES = [
    "model_outputs",
    "orders",
    "fills",
    "positions",
    "pnl_marks",
]


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

    Runs in a child process.  Opens its own disk-based DB connection for reads
    and writes output rows to an in-memory DB.  Returns the output rows as
    lists of tuples for merging back into the main DB.

    Args is a tuple of:
        (db_path, market_ids, strategy_dicts, starting_cash, transport_mode,
         eval_stride, experiment_id, worker_index)
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
    ) = args

    worker_name = f"worker-{worker_index}"
    logger.info("[%s] Starting with %d markets", worker_name, len(market_ids))

    # Reconstruct strategies from dicts (StrategyConfig is frozen dataclass).
    # ty cannot verify **dict unpacking statically, but dc_asdict() guarantees
    # all required fields are present.
    strategies = [StrategyConfig(**d) for d in strategy_dicts]  # type: ignore[missing-argument]

    # Open disk-based read connection (WAL allows concurrent readers)
    read_conn = db.connect(db_path)

    # Create an in-memory DB for writes
    write_conn = sqlite3.connect(":memory:")
    write_conn.row_factory = sqlite3.Row
    write_conn.execute("PRAGMA foreign_keys = OFF")  # No FK checks in worker
    db.init_db(write_conn)

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

    # Build a "split" connection that reads from disk but writes to memory.
    # The ReplayEngine needs a single conn for both reads and writes.
    # We use the disk connection and intercept writes via a custom wrapper.
    # Actually — simpler approach: attach the disk DB to the in-memory DB
    # so the worker has one connection that reads source data from the
    # attached DB and writes output to the in-memory tables.

    # Attach the disk DB as a read source
    write_conn.execute(f"ATTACH DATABASE 'file:{db_path}?mode=ro' AS source")

    # Create views in main (in-memory) that proxy read-only tables from source.
    # This way the ReplayEngine's reads hit the disk DB, while writes go to
    # the in-memory tables.
    _SOURCE_TABLES = [
        "markets",
        "market_snapshots",
        "market_rule_revisions",
        "orderbook_levels",
        "news_documents",
        "market_news_links",
        "market_resolutions",
    ]
    for table in _SOURCE_TABLES:
        # Drop the local table created by init_db and replace with a view
        write_conn.execute(f"DROP TABLE IF EXISTS main.{table}")
        write_conn.execute(f"CREATE VIEW main.{table} AS SELECT * FROM source.{table}")

    # We also need the experiments table to exist with the experiment_id row.
    # Insert a stub row so FK-free writes referencing experiment_id work.
    write_conn.execute(
        """
        INSERT INTO experiments (id, name, model_id, model_release,
                                 system_prompt_hash, config_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (experiment_id, f"parallel_{worker_name}", model_id, DEFAULT_MODEL_RELEASE, "", "{}"),
    )

    # Build grok client using the write_conn (which reads from attached source)
    grok = ReplayGrokClient(
        conn=write_conn,
        experiment_id=experiment_id,
        model_id=model_id,
        model_release=DEFAULT_MODEL_RELEASE,
        transport=transport,
        lookback_minutes=config.lookback_minutes,
    )
    grok.context_builder.skip_audit = True
    grok.context_builder.skip_related_markets = True

    engine = ReplayEngine(
        conn=write_conn,
        config=config,
        grok=grok,
        strategies=strategies,
    )
    engine.run_markets(market_ids)

    # Extract output rows from in-memory tables
    output_data: dict[str, list[tuple[Any, ...]]] = {}
    for table in _OUTPUT_TABLES:
        rows = write_conn.execute(f"SELECT * FROM main.{table}").fetchall()
        if rows:
            output_data[table] = [tuple(row) for row in rows]
        else:
            output_data[table] = []

    # Also extract column names for each table (needed for INSERT)
    column_info: dict[str, list[str]] = {}
    for table in _OUTPUT_TABLES:
        pragma_rows = write_conn.execute(f"PRAGMA table_info({table})").fetchall()
        column_info[table] = [str(r[1]) for r in pragma_rows]

    read_conn.close()
    write_conn.close()

    logger.info("[%s] Finished processing %d markets", worker_name, len(market_ids))

    return {
        "output_data": output_data,
        "column_info": column_info,
        "n_markets": len(market_ids),
    }


def _merge_worker_results(
    conn: sqlite3.Connection,
    worker_results: list[dict[str, Any]],
) -> None:
    """Merge output rows from all workers into the main connection."""
    for result in worker_results:
        output_data = result["output_data"]
        column_info = result["column_info"]
        for table in _OUTPUT_TABLES:
            rows = output_data.get(table, [])
            if not rows:
                continue
            columns = column_info[table]
            # Skip auto-increment 'id' column for tables that have it
            if table in ("model_outputs", "pnl_marks") and columns and columns[0] == "id":
                columns = columns[1:]
                rows = [row[1:] for row in rows]
            col_list = ", ".join(columns)
            placeholders = ", ".join("?" for _ in columns)
            conn.executemany(
                f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
                rows,
            )
    conn.commit()


def run_parallel_grid_search(
    db_path: Path,
    conn: sqlite3.Connection,
    *,
    strategies: list[StrategyConfig],
    starting_cash: float,
    transport_mode: str,
    market_ids: list[str],
    eval_stride: int = 4,
    n_workers: int = 0,
) -> tuple[int, dict[str, Any]]:
    """Run grid search with multiprocessing.

    Args:
        db_path: Path to the on-disk SQLite database.
        conn: Main process connection (for creating experiment and merging).
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
        )
        for i, group in enumerate(groups)
    ]

    # Run workers
    ctx = mp.get_context("spawn")  # spawn is safest for SQLite
    with ctx.Pool(processes=actual_workers) as pool:
        results = pool.map(_worker_run_markets, worker_args)

    # Merge all worker output into main connection
    print(f"[Parallel] Merging results from {actual_workers} workers")
    _merge_worker_results(conn, results)

    # Compute metrics on merged data
    persist_metric_results(conn, experiment_id, config.markout_horizons_min)
    summary = build_metrics_summary(conn, experiment_id, config.markout_horizons_min)

    total_markets = sum(r["n_markets"] for r in results)
    print(f"[Parallel] Done. Processed {total_markets} markets across {actual_workers} workers")

    return experiment_id, summary
