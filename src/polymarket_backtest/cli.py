from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from . import db
from .demo import build_demo_database, run_demo
from .downloaders.gamma import fetch_resolved_markets
from .downloaders.warproxxx import download_archive, extract_archive
from .grok_replay import ReplayGrokClient, create_transport
from .ingest import ingest_gamma, ingest_warproxxx
from .metrics import (
    build_metrics_summary,
    compute_calibration_curve,
    compute_edge_decay,
    compute_pnl_summary,
    compute_sharpe_like,
    compute_trade_pnl_details,
    persist_metric_results,
)
from .replay_engine import ReplayEngine
from .report import ReportGenerator
from .snapshot_builder import build_all_snapshots
from .strategies import default_strategy_grid
from .types import ReplayConfig

DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_PATH = Path("backtest.sqlite")
DEFAULT_DEMO_DB_PATH = Path("demo_replay.sqlite")
DEFAULT_GAMMA_CACHE_FILENAME = "gamma_resolved_markets.json"
DEFAULT_MODEL_RELEASE = "grok-3"
FORECAST_MODE_CHOICES = ("deterministic", "xai", "xai_search")
DEFAULT_LOOKBACK_MINUTES = 240
DEFAULT_MARKOUT_HORIZONS = (1, 5, 30, 240)
SOURCE_CHOICES = ("warproxxx", "gamma", "all")
STRATEGY_CHOICES = ("carry", "news", "all")


@click.group()
def main() -> None:
    """Run Polymarket backtest workflows."""


@main.command()
@click.option(
    "--dest-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DATA_DIR,
    show_default=True,
)
@click.option(
    "--source",
    type=click.Choice(SOURCE_CHOICES, case_sensitive=False),
    default="all",
    show_default=True,
)
def download(dest_dir: Path, source: str) -> None:
    """Download raw backtest inputs into the local cache directory."""

    dest_dir.mkdir(parents=True, exist_ok=True)

    for selected_source in _selected_sources(source):
        if selected_source == "warproxxx":
            click.echo(f"[download] warproxxx archive -> {dest_dir}")
            archive_path = download_archive(dest_dir)
            extracted_dir = extract_archive(archive_path, dest_dir)
            click.echo(f"[download] warproxxx ready at {extracted_dir}")
            continue

        click.echo("[download] gamma resolved markets -> cache file")
        markets = fetch_resolved_markets()
        cache_path = dest_dir / DEFAULT_GAMMA_CACHE_FILENAME
        cache_payload = {
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "count": len(markets),
            "markets": markets,
        }
        cache_path.write_text(json.dumps(cache_payload, indent=2, sort_keys=True), encoding="utf-8")
        click.echo(f"[download] gamma cached {len(markets)} markets at {cache_path}")


@main.command()
@click.option(
    "--db-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_DB_PATH,
    show_default=True,
)
@click.option(
    "--source",
    type=click.Choice(SOURCE_CHOICES, case_sensitive=False),
    default="all",
    show_default=True,
)
def ingest(db_path: Path, source: str) -> None:
    """Ingest downloaded data into SQLite and build regularized snapshots."""

    _ensure_parent_dir(db_path)
    source_counts: dict[str, dict[str, int]] = {}

    if source in {"warproxxx", "all"}:
        click.echo(f"[ingest] warproxxx -> {db_path} from {DEFAULT_DATA_DIR}")
        source_counts["warproxxx"] = ingest_warproxxx(db_path, DEFAULT_DATA_DIR)

    conn = db.connect(db_path)
    try:
        db.init_db(conn)

        if source in {"gamma", "all"}:
            click.echo(f"[ingest] gamma -> {db_path}")
            source_counts["gamma"] = ingest_gamma(conn)

        click.echo("[ingest] regularizing market snapshots")
        regularized_counts = build_all_snapshots(conn)
        totals = _table_counts(conn)
    finally:
        conn.close()

    for label, counts in source_counts.items():
        click.echo(f"[ingest] {label}: {_format_counts(counts)}")
    click.echo(
        f"[ingest] snapshots_regularized={sum(regularized_counts.values())} across {len(regularized_counts)} markets"
    )
    click.echo(f"[ingest] totals: {_format_counts(totals)}")


@main.command()
@click.option(
    "--db-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_DB_PATH,
    show_default=True,
)
@click.option(
    "--strategy",
    type=click.Choice(STRATEGY_CHOICES, case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option(
    "--starting-cash",
    type=click.FloatRange(min=0.0, min_open=True),
    default=1000.0,
    show_default=True,
)
@click.option("--market-ids", type=str, default=None)
@click.option(
    "--forecast-mode",
    type=click.Choice(FORECAST_MODE_CHOICES, case_sensitive=False),
    default="deterministic",
    show_default=True,
    help="Forecast transport: deterministic (fake), xai (live API), or xai_search (live + X search).",
)
def backtest(
    db_path: Path,
    strategy: str,
    starting_cash: float,
    market_ids: str | None,
    forecast_mode: str,
) -> None:
    """Run a replay backtest and print the markdown report."""

    _ensure_parent_dir(db_path)
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        replay_timestamps = db.get_all_timestamps(conn)
        if not replay_timestamps:
            raise click.ClickException(f"No replay data found in {db_path}. Run `polymarket-backtest ingest` first.")

        selected_market_ids = _parse_market_ids(market_ids)
        strategies = _select_strategies(strategy)
        config = ReplayConfig(
            experiment_name=_build_experiment_name(strategy),
            starting_cash=starting_cash,
            lookback_minutes=DEFAULT_LOOKBACK_MINUTES,
        )
        transport = create_transport(mode=forecast_mode, model_release=DEFAULT_MODEL_RELEASE)
        click.echo(f"[backtest] Forecast transport: {type(transport).__name__}")
        if not getattr(transport, "is_live_safe", True):
            click.echo("[backtest] WARNING: Using FAKE forecaster. Results are for testing only.")
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
                "strategy_selection": strategy,
                "market_ids": selected_market_ids or [],
            },
        )
        grok.experiment_id = experiment_id
        grok.context_builder.experiment_id = experiment_id
        engine = ReplayEngine(
            conn=conn,
            config=config,
            grok=grok,
            strategies=strategies,
        )

        click.echo(
            "[backtest] "
            f"experiment_id={experiment_id} strategy={strategy} "
            f"markets={len(selected_market_ids) if selected_market_ids else 'all'}"
        )
        if selected_market_ids is None:
            engine.run()
        elif len(selected_market_ids) == 1:
            engine.run_single_market(selected_market_ids[0])
        else:
            engine.run_markets(selected_market_ids)

        click.echo("[backtest] persisting metrics")
        persist_metric_results(conn, experiment_id, config.markout_horizons_min)
        summary = build_metrics_summary(conn, experiment_id, config.markout_horizons_min)
        report_output = ReportGenerator().build_markdown(
            summary,
            _load_experiment_report_config(conn, experiment_id),
        )
    finally:
        conn.close()

    click.echo(f"Experiment ID: {experiment_id}")
    click.echo(report_output)


@main.command()
@click.option(
    "--db-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_DB_PATH,
    show_default=True,
)
@click.option("--experiment-id", type=click.IntRange(min=1), required=True)
def report(db_path: Path, experiment_id: int) -> None:
    """Build the markdown report for an existing experiment."""

    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        horizons = _load_markout_horizons(conn, experiment_id)
        summary = _load_metrics_summary(conn, experiment_id, horizons)
        report_config = _load_experiment_report_config(conn, experiment_id)
    finally:
        conn.close()

    click.echo(ReportGenerator().build_markdown(summary, report_config))


@main.command()
@click.option(
    "--db-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_DEMO_DB_PATH,
    show_default=True,
)
def demo(db_path: Path) -> None:
    """Build and run the existing deterministic demo."""

    _ensure_parent_dir(db_path)
    build_demo_database(db_path)
    click.echo(run_demo(db_path))


def _selected_sources(source: str) -> tuple[str, ...]:
    normalized = source.lower()
    if normalized == "all":
        return ("warproxxx", "gamma")
    return (normalized,)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_market_ids(raw_market_ids: str | None) -> list[str] | None:
    if raw_market_ids is None:
        return None
    parsed = [market_id.strip() for market_id in raw_market_ids.split(",") if market_id.strip()]
    if not parsed:
        return None
    return list(dict.fromkeys(parsed))


def _select_strategies(selection: str) -> list[Any]:
    strategies = default_strategy_grid()
    normalized = selection.lower()
    if normalized == "all":
        return strategies
    if normalized == "carry":
        selected = [strategy for strategy in strategies if strategy.family == "carry_only"]
    else:
        selected = [strategy for strategy in strategies if strategy.family != "carry_only"]
    if not selected:
        raise click.ClickException(f"No strategies matched selection: {selection}")
    return selected


def _build_experiment_name(selection: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"replay_{selection.lower()}_{timestamp}"


def _table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table_name in ("markets", "market_snapshots", "market_resolutions", "market_rule_revisions"):
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}").fetchone()
        counts[table_name] = int(row["n"])
    return counts


def _format_counts(counts: Mapping[str, int | float]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _load_experiment_report_config(conn: sqlite3.Connection, experiment_id: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT name, created_ts, config_json
        FROM experiments
        WHERE id = ?
        """,
        (experiment_id,),
    ).fetchone()
    if row is None:
        raise click.ClickException(f"Experiment {experiment_id} was not found in the database.")

    config = json.loads(str(row["config_json"]))
    starting_cash = float(config.get("starting_cash", 0.0))
    strategy_rows = conn.execute(
        """
        SELECT strategy_name
        FROM (
            SELECT DISTINCT strategy_name FROM orders WHERE experiment_id = ?
            UNION
            SELECT DISTINCT strategy_name FROM fills WHERE experiment_id = ?
            UNION
            SELECT DISTINCT strategy_name FROM positions WHERE experiment_id = ?
            UNION
            SELECT DISTINCT strategy_name FROM pnl_marks WHERE experiment_id = ?
        )
        ORDER BY strategy_name
        """,
        (experiment_id, experiment_id, experiment_id, experiment_id),
    ).fetchall()
    strategy_count = len(strategy_rows)
    market_row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM (
            SELECT market_id FROM model_outputs WHERE experiment_id = ?
            UNION
            SELECT market_id FROM orders WHERE experiment_id = ?
            UNION
            SELECT market_id FROM fills WHERE experiment_id = ?
            UNION
            SELECT market_id FROM positions WHERE experiment_id = ?
        )
        """,
        (experiment_id, experiment_id, experiment_id, experiment_id),
    ).fetchone()
    trade_row = conn.execute(
        "SELECT COUNT(*) AS n FROM positions WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    range_row = conn.execute(
        """
        SELECT MIN(ts) AS start_ts, MAX(ts) AS end_ts
        FROM (
            SELECT ts FROM model_outputs WHERE experiment_id = ?
            UNION ALL
            SELECT ts FROM orders WHERE experiment_id = ?
            UNION ALL
            SELECT fill_ts AS ts FROM fills WHERE experiment_id = ?
            UNION ALL
            SELECT ts FROM pnl_marks WHERE experiment_id = ?
        )
        """,
        (experiment_id, experiment_id, experiment_id, experiment_id),
    ).fetchone()
    return {
        "experiment_name": str(row["name"]),
        "created_ts": str(row["created_ts"]),
        "date_start": str(range_row["start_ts"]) if range_row and range_row["start_ts"] is not None else None,
        "date_end": str(range_row["end_ts"]) if range_row and range_row["end_ts"] is not None else None,
        "market_count": int(market_row["n"]) if market_row is not None else 0,
        "trade_count": int(trade_row["n"]) if trade_row is not None else 0,
        "starting_cash": starting_cash,
        "strategy_count": strategy_count,
        "starting_cash_total": round(starting_cash * strategy_count, 4) if strategy_count else starting_cash,
    }


def _load_markout_horizons(conn: sqlite3.Connection, experiment_id: int) -> tuple[int, ...]:
    row = conn.execute(
        "SELECT config_json FROM experiments WHERE id = ?",
        (experiment_id,),
    ).fetchone()
    if row is None:
        raise click.ClickException(f"Experiment {experiment_id} was not found in the database.")

    config = json.loads(str(row["config_json"]))
    raw_horizons = config.get("markout_horizons_min")
    if not isinstance(raw_horizons, list | tuple) or not raw_horizons:
        return DEFAULT_MARKOUT_HORIZONS
    return tuple(int(horizon) for horizon in raw_horizons)


def _load_metrics_summary(
    conn: sqlite3.Connection,
    experiment_id: int,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT scope_type, scope_name, metric_name, horizon_min, metric_value, sample_size, extra_json
        FROM metric_results
        WHERE experiment_id = ?
        ORDER BY metric_name, scope_name, horizon_min
        """,
        (experiment_id,),
    ).fetchall()
    if not rows:
        return build_metrics_summary(conn, experiment_id, horizons)

    brier: list[dict[str, Any]] = []
    log_score: list[dict[str, Any]] = []
    brier_comparison: list[dict[str, Any]] = []
    markouts: dict[int, float] = {}
    adverse_selection: dict[int, float] = {}
    fill_ratio = {
        "filled_quantity": 0.0,
        "requested_quantity": 0.0,
        "fill_ratio": 0.0,
    }

    for row in rows:
        metric_name = str(row["metric_name"])
        scope_name = str(row["scope_name"])
        horizon_min = int(row["horizon_min"]) if row["horizon_min"] is not None else None
        metric_value = float(row["metric_value"])
        sample_size = int(row["sample_size"])
        extra = json.loads(str(row["extra_json"]))

        if metric_name == "brier_score":
            agent_name, domain = scope_name.split(":", maxsplit=1)
            brier.append(
                {
                    "agent_name": agent_name,
                    "domain": domain,
                    "brier_score": metric_value,
                    "n": sample_size,
                }
            )
        elif metric_name == "log_score":
            agent_name, domain = scope_name.split(":", maxsplit=1)
            log_score.append(
                {
                    "agent_name": agent_name,
                    "domain": domain,
                    "log_score": metric_value,
                    "n": sample_size,
                }
            )
        elif metric_name == "brier_comparison":
            agent_name, market_id, event_ts = scope_name.split(":", maxsplit=2)
            brier_comparison.append(
                {
                    "agent_name": agent_name,
                    "domain": str(extra.get("domain", "")),
                    "market_id": market_id,
                    "market_title": str(extra.get("market_title", market_id)),
                    "ts": str(extra.get("ts", event_ts)),
                    "agent_probability": float(extra.get("agent_probability", 0.0)),
                    "market_mid": float(extra.get("market_mid", 0.0)),
                    "resolved_outcome": float(extra.get("resolved_outcome", 0.0)),
                    "agent_brier": float(extra.get("agent_brier", 0.0)),
                    "market_brier": float(extra.get("market_brier", 0.0)),
                    "brier_improvement": metric_value,
                    "agent_better": bool(extra.get("agent_better", False)),
                }
            )
        elif metric_name == "markout" and horizon_min is not None:
            markouts[horizon_min] = metric_value
        elif metric_name == "adverse_selection_rate" and horizon_min is not None:
            adverse_selection[horizon_min] = metric_value
        elif metric_name == "fill_ratio":
            fill_ratio = {
                "filled_quantity": float(extra.get("filled_quantity", 0.0)),
                "requested_quantity": float(extra.get("requested_quantity", 0.0)),
                "fill_ratio": float(extra.get("fill_ratio", metric_value)),
            }

    if not brier and not markouts:
        return build_metrics_summary(conn, experiment_id, horizons)

    return {
        "brier": brier,
        "log_score": log_score,
        "brier_comparison": brier_comparison,
        "markouts": dict(sorted(markouts.items())),
        "adverse_selection": dict(sorted(adverse_selection.items())),
        "fill_ratio": fill_ratio,
        "pnl": compute_pnl_summary(conn, experiment_id),
        "trade_pnl": compute_trade_pnl_details(conn, experiment_id),
        "calibration": compute_calibration_curve(conn, experiment_id),
        "sharpe_like": compute_sharpe_like(conn, experiment_id),
        "edge_decay": compute_edge_decay(conn, experiment_id, horizons),
    }


if __name__ == "__main__":
    main()
