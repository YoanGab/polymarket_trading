from __future__ import annotations

from pathlib import Path

from . import db
from .grok_replay import DeterministicReplayTransport, ReplayGrokClient
from .metrics import build_metrics_summary, persist_metric_results
from .replay_engine import ReplayEngine
from .report import ReportGenerator
from .strategies import default_strategy_grid
from .types import ReplayConfig


def build_demo_database(path: Path) -> None:
    if path.exists():
        path.unlink()
    conn = db.connect(path)
    db.init_db(conn)
    db.seed_demo_data(conn)
    conn.close()


def run_demo(path: Path) -> str:
    conn = db.connect(path)
    config = ReplayConfig(
        experiment_name="demo_replay",
        starting_cash=1_000.0,
        lookback_minutes=240,
    )
    transport = DeterministicReplayTransport(model_id="grok")
    grok = ReplayGrokClient(
        conn=conn,
        experiment_id=None,
        model_id="grok",
        model_release="grok-4.20-beta-0309-reasoning",
        transport=transport,
        lookback_minutes=config.lookback_minutes,
    )
    experiment_id = db.create_experiment(
        conn,
        name=config.experiment_name,
        model_id=grok.model_id,
        model_release=grok.model_release,
        system_prompt_hash=grok.prompt_hash(db.get_all_timestamps(conn)[0]),
        config={
            "starting_cash": config.starting_cash,
            "lookback_minutes": config.lookback_minutes,
            "markout_horizons_min": config.markout_horizons_min,
        },
    )
    grok.experiment_id = experiment_id
    grok.context_builder.experiment_id = experiment_id
    engine = ReplayEngine(
        conn=conn,
        config=config,
        grok=grok,
        strategies=default_strategy_grid(),
    )
    engine.run()
    persist_metric_results(conn, experiment_id, config.markout_horizons_min)
    summary = build_metrics_summary(conn, experiment_id, config.markout_horizons_min)
    report = ReportGenerator().build_markdown(summary)
    conn.close()
    return report


def main() -> None:
    db_path = Path("demo_replay.sqlite")
    build_demo_database(db_path)
    print(run_demo(db_path))


if __name__ == "__main__":
    main()
