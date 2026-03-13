from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from polymarket_backtest import db
from polymarket_backtest.demo import build_demo_database, run_demo
from polymarket_backtest.grok_replay import DeterministicReplayTransport, ReplayGrokClient
from polymarket_backtest.metrics import build_metrics_summary, persist_metric_results
from polymarket_backtest.replay_engine import ReplayEngine
from polymarket_backtest.report import ReportGenerator
from polymarket_backtest.strategies import default_strategy_grid
from polymarket_backtest.types import ReplayConfig

MODEL_RELEASE = "grok-4.20-beta-0309-reasoning"
REPORT_SECTIONS = ("Forecast Quality", "Execution", "PnL", "Calibration")
STRATEGY_NAMES = {strategy.name for strategy in default_strategy_grid()}


@dataclass(frozen=True)
class ReplayRun:
    experiment_id: int
    config: ReplayConfig
    report: str
    summary: dict[str, Any]


def _assert_report_sections(report: str) -> None:
    assert isinstance(report, str)
    assert report.strip()
    for section in REPORT_SECTIONS:
        assert section in report


def _market_index(market_id: str) -> int:
    digits = "".join(ch for ch in market_id if ch.isdigit())
    if digits:
        return int(digits)
    return sum(ord(ch) for ch in market_id)


def _scalar(conn: Any, query: str, params: tuple[Any, ...] = ()) -> Any:
    return conn.execute(query, params).fetchone()[0]


def _build_replay_run(
    conn: Any,
    *,
    experiment_name: str,
    market_ids: list[str] | None = None,
    single_market_id: str | None = None,
) -> ReplayRun:
    timestamps = db.get_all_timestamps(conn)
    assert timestamps, "Synthetic test database must have replay timestamps"

    config = ReplayConfig(
        experiment_name=experiment_name,
        starting_cash=1_000.0,
        lookback_minutes=240,
    )
    transport = DeterministicReplayTransport(model_id="grok")
    grok = ReplayGrokClient(
        conn=conn,
        experiment_id=None,
        model_id="grok",
        model_release=MODEL_RELEASE,
        transport=transport,
        lookback_minutes=config.lookback_minutes,
    )
    experiment_id = db.create_experiment(
        conn,
        name=experiment_name,
        model_id=grok.model_id,
        model_release=grok.model_release,
        system_prompt_hash=grok.prompt_hash(timestamps[0]),
        config={
            "starting_cash": config.starting_cash,
            "lookback_minutes": config.lookback_minutes,
            "markout_horizons_min": config.markout_horizons_min,
            "market_ids": market_ids or ([] if single_market_id else []),
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

    if single_market_id is not None:
        returned_experiment_id = engine.run_single_market(single_market_id)
    elif market_ids is not None:
        returned_experiment_id = engine.run_markets(market_ids)
    else:
        returned_experiment_id = engine.run()

    assert returned_experiment_id == experiment_id

    persist_metric_results(conn, experiment_id, config.markout_horizons_min)
    summary = build_metrics_summary(conn, experiment_id, config.markout_horizons_min)
    report = ReportGenerator().build_markdown(summary)
    return ReplayRun(
        experiment_id=experiment_id,
        config=config,
        report=report,
        summary=summary,
    )


def create_synthetic_market(
    conn: Any,
    market_id: str,
    *,
    resolved_yes: bool = True,
    n_snapshots: int = 20,
) -> None:
    market_index = _market_index(market_id)
    base_ts = datetime(2026, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=market_index * 7)
    resolution_ts = base_ts + timedelta(hours=24)
    domain = f"synthetic_domain_{market_index % 3}"

    db.add_market(
        conn,
        market_id=market_id,
        title=f"Synthetic market {market_id}",
        domain=domain,
        market_type="binary",
        open_ts=base_ts - timedelta(hours=12),
        close_ts=resolution_ts - timedelta(minutes=5),
        resolution_ts=resolution_ts,
        status="active",
    )
    db.add_rule_revision(
        conn,
        market_id=market_id,
        effective_ts=base_ts - timedelta(hours=12),
        rules_text="Resolves YES if the synthetic event occurs within the replay window.",
        additional_context="Synthetic E2E fixture market.",
    )
    db.add_news(
        conn,
        document_id=f"news_{market_id}",
        source="synthetic",
        url=f"https://example.test/{market_id}",
        title="Policy note says no change expected",
        published_ts=base_ts + timedelta(minutes=10),
        first_seen_ts=base_ts + timedelta(minutes=12),
        ingested_ts=base_ts + timedelta(minutes=13),
        content="Analysts expect conditions to remain unchanged during the replay window.",
        metadata={"impact": 0.0, "fixture": "e2e"},
        market_ids=[market_id],
    )

    snapshot_spacing_seconds = ((24 * 60 * 60) - 60) // max(1, n_snapshots - 1)
    total_drift = 0.008 if resolved_yes else -0.01
    base_mid = 0.955 + ((market_index % 4) * 0.002)

    for snapshot_index in range(n_snapshots):
        ts = base_ts + timedelta(seconds=snapshot_spacing_seconds * snapshot_index)
        spread = 0.01 if (market_index + snapshot_index) % 2 == 0 else 0.02
        drift = (snapshot_index / max(1, n_snapshots - 1)) * total_drift
        mid = min(0.985, max(0.93, base_mid + drift))
        best_bid = round(mid - (spread / 2.0), 3)
        best_ask = round(best_bid + spread, 3)
        status = "active" if snapshot_index < n_snapshots - 1 else "pending_resolution"

        db.add_snapshot(
            conn,
            market_id=market_id,
            ts=ts,
            status=status,
            best_bid=best_bid,
            best_ask=best_ask,
            last_trade=round((best_bid + best_ask) / 2.0, 3),
            volume_1m=120.0 + (snapshot_index * 5.0) + market_index,
            volume_24h=15_000.0 + (snapshot_index * 75.0) + (market_index * 20.0),
            open_interest=8_000.0 + (snapshot_index * 40.0) + (market_index * 15.0),
            tick_size=0.001,
            orderbook=[
                ("bid", 1, best_bid, 300.0 + market_index),
                ("bid", 2, round(max(0.001, best_bid - 0.002), 3), 500.0 + market_index),
                ("bid", 3, round(max(0.001, best_bid - 0.004), 3), 700.0 + market_index),
                ("ask", 1, best_ask, 300.0 + market_index),
                ("ask", 2, round(min(0.999, best_ask + 0.002), 3), 500.0 + market_index),
                ("ask", 3, round(min(0.999, best_ask + 0.004), 3), 700.0 + market_index),
            ],
        )

    db.add_resolution(
        conn,
        market_id=market_id,
        resolution_ts=resolution_ts,
        resolved_outcome=1.0 if resolved_yes else 0.0,
        status="resolved",
    )
    conn.commit()


def _build_bulk_rows(
    *,
    n_markets: int,
    n_snapshots: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    markets: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []

    for market_index in range(n_markets):
        market_id = f"bulk_market_{market_index:03d}"
        base_ts = datetime(2026, 2, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=market_index)
        resolution_ts = base_ts + timedelta(hours=12)

        markets.append(
            {
                "market_id": market_id,
                "title": f"Bulk synthetic market {market_index}",
                "domain": f"bulk_domain_{market_index % 5}",
                "market_type": "binary",
                "open_ts": base_ts - timedelta(hours=6),
                "close_ts": resolution_ts - timedelta(minutes=5),
                "resolution_ts": resolution_ts,
                "status": "active",
            }
        )
        resolutions.append(
            {
                "market_id": market_id,
                "resolution_ts": resolution_ts,
                "resolved_outcome": 1.0 if market_index % 2 == 0 else 0.0,
                "status": "resolved",
            }
        )

        for snapshot_index in range(n_snapshots):
            ts = base_ts + timedelta(minutes=90 * snapshot_index)
            spread = 0.01 if (market_index + snapshot_index) % 2 == 0 else 0.02
            mid = min(0.79, 0.45 + ((market_index % 7) * 0.02) + (snapshot_index * 0.005))
            best_bid = round(max(0.05, mid - (spread / 2.0)), 3)
            best_ask = round(min(0.95, best_bid + spread), 3)

            snapshots.append(
                {
                    "market_id": market_id,
                    "ts": ts,
                    "status": "active" if snapshot_index < n_snapshots - 1 else "pending_resolution",
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "last_trade": round((best_bid + best_ask) / 2.0, 3),
                    "volume_1m": 50.0 + snapshot_index + market_index,
                    "volume_24h": 5_000.0 + (snapshot_index * 40.0) + (market_index * 10.0),
                    "open_interest": 2_500.0 + (snapshot_index * 20.0) + (market_index * 3.0),
                    "tick_size": 0.001,
                }
            )

    return markets, snapshots, resolutions


def test_demo_e2e(tmp_path: Path) -> None:
    db_path = tmp_path / "demo_replay.sqlite"
    build_demo_database(db_path)

    report = run_demo(db_path)

    _assert_report_sections(report)


def test_synthetic_multi_market_e2e(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_multi.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        for market_index in range(10):
            create_synthetic_market(
                conn,
                f"synthetic_market_{market_index:02d}",
                resolved_yes=market_index < 5,
                n_snapshots=20,
            )

        replay = _build_replay_run(conn, experiment_name="synthetic_multi_market")
        order_count = _scalar(
            conn,
            "SELECT COUNT(*) FROM orders WHERE experiment_id = ?",
            (replay.experiment_id,),
        )
        fill_count = _scalar(
            conn,
            "SELECT COUNT(*) FROM fills WHERE experiment_id = ?",
            (replay.experiment_id,),
        )

        assert isinstance(replay.experiment_id, int)
        assert replay.experiment_id > 0
        assert order_count > 0
        assert fill_count > 0
        assert replay.summary["brier"]
        assert all(0.0 <= item["brier_score"] <= 1.0 for item in replay.summary["brier"])
        assert {item["strategy_name"] for item in replay.summary["pnl"]} == STRATEGY_NAMES
        _assert_report_sections(replay.report)


def test_bulk_ingest_and_replay(tmp_path: Path) -> None:
    db_path = tmp_path / "bulk_ingest.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        markets, snapshots, resolutions = _build_bulk_rows(n_markets=50)

        assert db.bulk_add_markets(conn, markets) == 50
        assert db.bulk_add_snapshots(conn, snapshots) == 400
        assert db.bulk_add_resolutions(conn, resolutions) == 50

        replay = _build_replay_run(conn, experiment_name="bulk_ingest_replay")
        model_output_count = _scalar(
            conn,
            "SELECT COUNT(*) FROM model_outputs WHERE experiment_id = ?",
            (replay.experiment_id,),
        )

        assert isinstance(replay.experiment_id, int)
        assert replay.experiment_id > 0
        assert model_output_count > 0
        assert replay.summary["brier"]
        assert replay.summary["calibration"]


def test_single_market_replay(tmp_path: Path) -> None:
    db_path = tmp_path / "single_market.sqlite"
    market_id = "single_market_001"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        create_synthetic_market(conn, market_id, resolved_yes=True, n_snapshots=20)

        replay = _build_replay_run(
            conn,
            experiment_name="single_market_replay",
            single_market_id=market_id,
        )
        model_output_count = _scalar(
            conn,
            "SELECT COUNT(*) FROM model_outputs WHERE experiment_id = ?",
            (replay.experiment_id,),
        )
        order_count = _scalar(
            conn,
            "SELECT COUNT(*) FROM orders WHERE experiment_id = ?",
            (replay.experiment_id,),
        )
        fill_count = _scalar(
            conn,
            "SELECT COUNT(*) FROM fills WHERE experiment_id = ?",
            (replay.experiment_id,),
        )
        position_count = _scalar(
            conn,
            "SELECT COUNT(*) FROM positions WHERE experiment_id = ?",
            (replay.experiment_id,),
        )
        only_market_id = conn.execute(
            """
            SELECT MIN(market_id)
            FROM model_outputs
            WHERE experiment_id = ?
            """,
            (replay.experiment_id,),
        ).fetchone()[0]
        distinct_market_count = _scalar(
            conn,
            """
            SELECT COUNT(DISTINCT market_id)
            FROM model_outputs
            WHERE experiment_id = ?
            """,
            (replay.experiment_id,),
        )

        assert isinstance(replay.experiment_id, int)
        assert replay.experiment_id > 0
        assert model_output_count == 20
        assert order_count > 0
        assert fill_count > 0
        assert position_count == len(STRATEGY_NAMES)
        assert distinct_market_count == 1
        assert only_market_id == market_id
        assert {item["strategy_name"] for item in replay.summary["pnl"]} == STRATEGY_NAMES
        _assert_report_sections(replay.report)
