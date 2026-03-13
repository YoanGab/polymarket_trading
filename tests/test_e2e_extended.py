from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from click.testing import CliRunner
from test_e2e import _build_replay_run, create_synthetic_market

from polymarket_backtest import db
from polymarket_backtest.cli import main as cli_main
from polymarket_backtest.grid_search import (
    build_grid_report,
    expanded_strategy_grid,
    rank_strategies,
)
from polymarket_backtest.metrics import bootstrap_mean_confidence_interval
from polymarket_backtest.replay_engine import ReplayEngine, StrategyPortfolio
from polymarket_backtest.report import ReportGenerator
from polymarket_backtest.strategies import StrategyEngine, kelly_fraction_for_yes
from polymarket_backtest.types import (
    ForecastOutput,
    MarketState,
    OrderIntent,
    OrderLevel,
    PositionState,
    ReplayConfig,
    StrategyConfig,
)

REPORT_SECTIONS = ("Forecast Quality", "Execution", "PnL", "Calibration")


def _scalar(conn: Any, query: str, params: tuple[Any, ...] = ()) -> Any:
    return conn.execute(query, params).fetchone()[0]


def _make_replay_engine(conn: Any, *, starting_cash: float = 1_000.0) -> ReplayEngine:
    config = ReplayConfig(
        experiment_name="unit_test",
        starting_cash=starting_cash,
        lookback_minutes=60,
    )
    experiment_id = db.create_experiment(
        conn,
        name="unit_test",
        model_id="grok",
        model_release="test",
        system_prompt_hash="unit-test",
        config={},
    )
    return ReplayEngine(
        conn=conn,
        config=config,
        grok=SimpleNamespace(experiment_id=experiment_id),
        strategies=[],
    )


def _make_market_state(
    market_id: str = "test_market",
    *,
    best_bid: float = 0.95,
    best_ask: float = 0.97,
    status: str = "active",
) -> MarketState:
    ts = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    return MarketState(
        market_id=market_id,
        title=f"Test market {market_id}",
        domain="test_domain",
        market_type="binary",
        ts=ts,
        status=status,
        best_bid=best_bid,
        best_ask=best_ask,
        mid=round((best_bid + best_ask) / 2.0, 4),
        last_trade=round((best_bid + best_ask) / 2.0, 4),
        volume_1m=100.0,
        volume_24h=10_000.0,
        open_interest=5_000.0,
        tick_size=0.001,
        rules_text="Test rules",
        additional_context="",
        resolution_ts=ts + timedelta(hours=24),
        fees_enabled=True,
        fee_rate=0.02,
        fee_exponent=1.0,
        maker_rebate_rate=0.01,
        orderbook=[
            OrderLevel(side="bid", price=best_bid, quantity=500.0, level_no=1),
            OrderLevel(side="ask", price=best_ask, quantity=500.0, level_no=1),
        ],
    )


def _make_forecast(
    market_id: str = "test_market",
    probability_yes: float = 0.98,
    confidence: float = 0.75,
) -> ForecastOutput:
    return ForecastOutput(
        agent_name="grok",
        model_id="grok",
        model_release="test",
        as_of=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        market_id=market_id,
        domain="test_domain",
        probability_yes=probability_yes,
        confidence=confidence,
        expected_edge_bps=100.0,
        thesis="Test thesis",
        reasoning="Test reasoning",
        evidence=[],
        raw_response={},
    )


# -- Strategy engine unit tests --


def test_kelly_fraction_boundaries() -> None:
    assert kelly_fraction_for_yes(1.0, 0.99) == 0.0
    assert kelly_fraction_for_yes(0.5, 0.5) == 0.0
    assert kelly_fraction_for_yes(0.5, 0.8) > 0.0
    assert kelly_fraction_for_yes(0.0, 0.0) == 0.0
    assert kelly_fraction_for_yes(0.01, 0.99) > 0.0


def test_carry_strategy_filters_price_range() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="carry_test",
        family="carry_only",
        kelly_fraction=0.05,
        edge_threshold_bps=10.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
        carry_price_min=0.95,
        carry_price_max=0.99,
    )
    forecast = _make_forecast(probability_yes=0.99)

    in_range = _make_market_state(best_ask=0.97)
    orders = engine.decide(config=config, market=in_range, forecast=forecast, position=None, available_cash=1_000.0)
    assert len(orders) > 0

    below_range = _make_market_state(best_ask=0.50)
    orders = engine.decide(
        config=config,
        market=below_range,
        forecast=forecast,
        position=None,
        available_cash=1_000.0,
    )
    assert len(orders) == 0

    above_range = _make_market_state(best_ask=0.995)
    orders = engine.decide(
        config=config,
        market=above_range,
        forecast=forecast,
        position=None,
        available_cash=1_000.0,
    )
    assert len(orders) == 0


def test_edge_based_respects_confidence_threshold() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="news_test",
        family="news_driven",
        kelly_fraction=0.15,
        edge_threshold_bps=50.0,
        max_position_notional=400.0,
        max_holding_minutes=240,
        min_confidence=0.70,
    )
    market = _make_market_state(best_ask=0.60, best_bid=0.58)
    forecast_high = _make_forecast(probability_yes=0.75, confidence=0.80)
    forecast_low = _make_forecast(probability_yes=0.75, confidence=0.50)

    orders_high = engine.decide(
        config=config,
        market=market,
        forecast=forecast_high,
        position=None,
        available_cash=1_000.0,
    )
    orders_low = engine.decide(
        config=config,
        market=market,
        forecast=forecast_low,
        position=None,
        available_cash=1_000.0,
    )

    assert len(orders_high) > 0
    assert len(orders_low) == 0


def test_should_exit_time_stop() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="time_stop_test",
        family="news_driven",
        kelly_fraction=0.10,
        edge_threshold_bps=50.0,
        max_position_notional=400.0,
        max_holding_minutes=120,
        use_time_stop=True,
        use_thesis_stop=False,
    )
    opened_ts = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    position = PositionState(
        strategy_name="time_stop_test",
        market_id="test_market",
        quantity=10.0,
        avg_entry_price=0.96,
        opened_ts=opened_ts,
        entry_probability=0.98,
    )
    forecast = _make_forecast(probability_yes=0.98)

    recent_market = _make_market_state()
    recent_market_early = MarketState(
        **{
            **recent_market.__dict__,
            "ts": opened_ts + timedelta(minutes=60),
        }
    )
    assert not engine.should_exit(config=config, market=recent_market_early, forecast=forecast, position=position)

    late_market = MarketState(
        **{
            **recent_market.__dict__,
            "ts": opened_ts + timedelta(minutes=180),
        }
    )
    assert engine.should_exit(config=config, market=late_market, forecast=forecast, position=position)


def test_should_exit_thesis_stop() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="thesis_stop_test",
        family="carry_only",
        kelly_fraction=0.05,
        edge_threshold_bps=25.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
        use_thesis_stop=True,
        thesis_stop_delta=0.08,
    )
    position = PositionState(
        strategy_name="thesis_stop_test",
        market_id="test_market",
        quantity=10.0,
        avg_entry_price=0.96,
        opened_ts=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        entry_probability=0.98,
    )
    market = _make_market_state()

    no_exit_forecast = _make_forecast(probability_yes=0.95)
    assert not engine.should_exit(config=config, market=market, forecast=no_exit_forecast, position=position)

    exit_forecast = _make_forecast(probability_yes=0.85)
    assert engine.should_exit(config=config, market=market, forecast=exit_forecast, position=position)


def test_no_double_entry_when_position_exists() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="no_double",
        family="carry_only",
        kelly_fraction=0.05,
        edge_threshold_bps=10.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
    )
    position = PositionState(
        strategy_name="no_double",
        market_id="test_market",
        quantity=5.0,
    )
    market = _make_market_state(best_ask=0.97)
    forecast = _make_forecast(probability_yes=0.99)

    orders = engine.decide(
        config=config,
        market=market,
        forecast=forecast,
        position=position,
        available_cash=1_000.0,
    )
    assert len(orders) == 0


def test_decide_uses_available_cash_for_kelly_sizing() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="carry_cash",
        family="carry_only",
        kelly_fraction=0.20,
        edge_threshold_bps=10.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
    )
    market = _make_market_state(best_ask=0.96, best_bid=0.94)
    forecast = _make_forecast(probability_yes=0.99)

    orders = engine.decide(
        config=config,
        market=market,
        forecast=forecast,
        position=None,
        available_cash=100.0,
    )

    assert len(orders) == 1
    expected_notional = min(
        config.max_position_notional,
        100.0 * config.kelly_fraction * kelly_fraction_for_yes(market.best_ask, forecast.probability_yes),
    )
    assert abs((orders[0].requested_quantity * market.best_ask) - expected_notional) < 1e-9


def test_edge_based_generates_sell_order_for_overpriced_yes() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="sell_test",
        family="news_driven",
        kelly_fraction=0.15,
        edge_threshold_bps=50.0,
        max_position_notional=400.0,
        max_holding_minutes=240,
        min_confidence=0.70,
    )
    market = _make_market_state(best_bid=0.62, best_ask=0.64)
    forecast = _make_forecast(probability_yes=0.55, confidence=0.80)
    position = PositionState(
        strategy_name="sell_test",
        market_id=market.market_id,
        quantity=12.0,
        avg_entry_price=0.58,
    )

    orders = engine.decide(
        config=config,
        market=market,
        forecast=forecast,
        position=position,
        available_cash=0.0,
    )

    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].requested_quantity == position.quantity
    assert orders[0].limit_price == market.best_bid


def test_carry_strategy_generates_sell_order_when_mid_near_zero() -> None:
    engine = StrategyEngine()
    config = StrategyConfig(
        name="carry_sell",
        family="carry_only",
        kelly_fraction=0.05,
        edge_threshold_bps=10.0,
        max_position_notional=250.0,
        max_holding_minutes=None,
    )
    market = _make_market_state(best_bid=0.03, best_ask=0.04)
    forecast = _make_forecast(probability_yes=0.02)
    position = PositionState(
        strategy_name="carry_sell",
        market_id=market.market_id,
        quantity=8.0,
        avg_entry_price=0.97,
    )

    orders = engine.decide(
        config=config,
        market=market,
        forecast=forecast,
        position=position,
        available_cash=0.0,
    )

    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].requested_quantity == position.quantity
    assert orders[0].limit_price == market.best_bid


def test_execute_order_skips_buy_when_estimated_notional_exceeds_cash(tmp_path: Path) -> None:
    db_path = tmp_path / "cash_guard.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        engine = _make_replay_engine(conn)
        portfolio = StrategyPortfolio(cash=5.0)
        market = _make_market_state(best_bid=0.95, best_ask=0.97)
        order = OrderIntent(
            strategy_name="cash_guard",
            market_id=market.market_id,
            ts=market.ts,
            side="buy",
            liquidity_intent="aggressive",
            limit_price=market.best_ask,
            requested_quantity=20.0,
            kelly_fraction=0.10,
            edge_bps=100.0,
            holding_period_minutes=None,
            thesis="cash guard",
        )

        engine._execute_order(
            portfolio=portfolio,
            market=market,
            next_market=market,
            order=order,
            entry_probability=0.99,
        )

        assert portfolio.cash == 5.0
        assert _scalar(conn, "SELECT COUNT(*) FROM orders WHERE experiment_id = ?", (engine.experiment_id,)) == 0


def test_mark_portfolio_uses_portfolio_wide_inventory_for_equity(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio_marks.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        engine = _make_replay_engine(conn)
        portfolio = StrategyPortfolio(
            cash=100.0,
            positions={
                "market_a": PositionState(
                    strategy_name="portfolio",
                    market_id="market_a",
                    quantity=2.0,
                    avg_entry_price=0.50,
                ),
                "market_b": PositionState(
                    strategy_name="portfolio",
                    market_id="market_b",
                    quantity=3.0,
                    avg_entry_price=0.40,
                ),
            },
            last_known_mids={"market_b": 0.80},
        )
        market = _make_market_state(market_id="market_a", best_bid=0.64, best_ask=0.66)

        engine._mark_portfolio_from_market(
            portfolio=portfolio,
            strategy_name="portfolio",
            market=market,
        )

        row = conn.execute(
            "SELECT equity, inventory_value FROM pnl_marks WHERE experiment_id = ?",
            (engine.experiment_id,),
        ).fetchone()
        assert row is not None
        expected_equity = 100.0 + (2.0 * market.mid) + (3.0 * 0.80)
        assert row["inventory_value"] == round(2.0 * market.mid, 4)
        assert row["equity"] == round(expected_equity, 4)
        assert portfolio.last_known_mids["market_a"] == market.mid


# -- Grid search pipeline E2E --


def test_grid_search_e2e(tmp_path: Path) -> None:
    db_path = tmp_path / "grid_search.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        for i in range(15):
            create_synthetic_market(
                conn,
                f"grid_market_{i:02d}",
                resolved_yes=i < 8,
                n_snapshots=15,
            )

        strategies = expanded_strategy_grid()[:3]
        timestamps = db.get_all_timestamps(conn)
        assert len(timestamps) > 0

        from polymarket_backtest.grid_search import _run_strategy_experiment

        results = []
        for strategy in strategies:
            experiment_id, summary = _run_strategy_experiment(
                conn,
                strategy=strategy,
                replay_timestamps=timestamps,
                starting_cash=1_000.0,
            )
            result = {
                "strategy_name": strategy.name,
                "pnl": sum(float(item.get("pnl_hold_to_resolution", 0.0)) for item in summary["pnl"]),
                "sharpe": max(
                    (float(item.get("sharpe_like", 0.0)) for item in summary["sharpe_like"]),
                    default=0.0,
                ),
                "brier_improvement": sum(
                    float(item.get("brier_improvement", 0.0)) for item in summary["brier_comparison"]
                )
                / max(1, len(summary["brier_comparison"])),
                "n_trades": _scalar(
                    conn,
                    "SELECT COUNT(*) FROM orders WHERE experiment_id = ? AND filled_quantity > 0",
                    (experiment_id,),
                ),
                "fill_ratio": float(summary["fill_ratio"]["fill_ratio"]),
            }
            results.append(result)

        assert len(results) == 3
        ranked = rank_strategies(results)
        assert len(ranked) == 3
        assert ranked[0]["rank"] == 1
        assert ranked[-1]["rank"] == 3
        assert all("composite_score" in r for r in ranked)

        report = build_grid_report(ranked)
        assert "Grid Search Report" in report
        assert "Strategy Comparison" in report
        assert "Recommendations" in report
        for r in ranked:
            assert r["strategy_name"] in report


# -- Diverse market patterns --


def test_mixed_outcome_markets(tmp_path: Path) -> None:
    db_path = tmp_path / "mixed_outcomes.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        for i in range(20):
            create_synthetic_market(
                conn,
                f"mixed_{i:02d}",
                resolved_yes=(i % 3 != 0),
                n_snapshots=25,
            )

        replay = _build_replay_run(conn, experiment_name="mixed_outcome_test")

        brier_items = replay.summary["brier"]
        assert len(brier_items) > 0
        assert all(0.0 <= item["brier_score"] <= 1.0 for item in brier_items)

        calibration = replay.summary["calibration"]
        assert len(calibration) > 0

        for section in REPORT_SECTIONS:
            assert section in replay.report


def test_selective_market_replay(tmp_path: Path) -> None:
    db_path = tmp_path / "selective.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        all_ids = []
        for i in range(10):
            market_id = f"sel_{i:02d}"
            all_ids.append(market_id)
            create_synthetic_market(conn, market_id, resolved_yes=i < 6, n_snapshots=20)

        selected = all_ids[:3]
        replay = _build_replay_run(conn, experiment_name="selective_replay", market_ids=selected)

        distinct_markets = _scalar(
            conn,
            "SELECT COUNT(DISTINCT market_id) FROM model_outputs WHERE experiment_id = ?",
            (replay.experiment_id,),
        )
        assert distinct_markets == 3

        market_ids_in_outputs = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT market_id FROM model_outputs WHERE experiment_id = ?",
                (replay.experiment_id,),
            ).fetchall()
        }
        assert market_ids_in_outputs == set(selected)


# -- Metrics unit tests --


def test_bootstrap_ci_edge_cases() -> None:
    assert bootstrap_mean_confidence_interval([]) == (0.0, 0.0)

    lower, upper = bootstrap_mean_confidence_interval([42.0])
    assert lower == 42.0
    assert upper == 42.0

    lower, upper = bootstrap_mean_confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lower <= 3.0
    assert upper >= 3.0
    assert lower <= upper

    lower, upper = bootstrap_mean_confidence_interval([1.0, 10.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0])
    assert lower <= upper


def test_metrics_persistence_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics_roundtrip.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        create_synthetic_market(conn, "roundtrip_001", resolved_yes=True, n_snapshots=20)

        replay = _build_replay_run(conn, experiment_name="roundtrip_test", single_market_id="roundtrip_001")

        metric_count = _scalar(
            conn,
            "SELECT COUNT(*) FROM metric_results WHERE experiment_id = ?",
            (replay.experiment_id,),
        )
        assert metric_count > 0

        brier_rows = conn.execute(
            "SELECT * FROM metric_results WHERE experiment_id = ? AND metric_name = 'brier_score'",
            (replay.experiment_id,),
        ).fetchall()
        assert len(brier_rows) > 0

        fill_rows = conn.execute(
            "SELECT * FROM metric_results WHERE experiment_id = ? AND metric_name = 'fill_ratio'",
            (replay.experiment_id,),
        ).fetchall()
        assert len(fill_rows) == 1


# -- Report generator tests --


def test_report_go_verdict(tmp_path: Path) -> None:
    db_path = tmp_path / "report_verdict.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        for i in range(5):
            create_synthetic_market(conn, f"verdict_{i}", resolved_yes=True, n_snapshots=20)

        replay = _build_replay_run(conn, experiment_name="verdict_test")

        assert "GO / NO-GO Assessment" in replay.report
        assert any(
            verdict in replay.report
            for verdict in (
                "Overall verdict: **GO**",
                "Overall verdict: **NO-GO**",
                "Overall verdict: **INSUFFICIENT DATA**",
            )
        )


def test_report_with_experiment_config(tmp_path: Path) -> None:
    db_path = tmp_path / "report_config.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        create_synthetic_market(conn, "config_001", resolved_yes=True, n_snapshots=20)

        replay = _build_replay_run(conn, experiment_name="config_test", single_market_id="config_001")

        generator = ReportGenerator()
        config_report = generator.build_markdown(
            replay.summary,
            {
                "experiment_name": "config_test",
                "starting_cash": 1_000.0,
                "strategy_count": 2,
                "date_start": "2026-01-01",
                "date_end": "2026-01-02",
            },
        )
        assert "config_test" in config_report
        assert "Date range" in config_report
        assert "Starting cash" in config_report
        for section in REPORT_SECTIONS:
            assert section in config_report


# -- CLI tests --


def test_cli_demo_command(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "cli_demo.sqlite"
    result = runner.invoke(cli_main, ["demo", "--db-path", str(db_path)])
    assert result.exit_code == 0
    assert "Replay Report" in result.output
    assert "GO / NO-GO Assessment" in result.output


def test_cli_backtest_command(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "cli_backtest.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        for i in range(5):
            create_synthetic_market(conn, f"cli_bt_{i}", resolved_yes=i < 3, n_snapshots=20)

    result = runner.invoke(cli_main, ["backtest", "--db-path", str(db_path), "--strategy", "carry"])
    assert result.exit_code == 0
    assert "Replay Report" in result.output


def test_cli_backtest_specific_markets(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "cli_specific.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        for i in range(5):
            create_synthetic_market(conn, f"cli_sp_{i}", resolved_yes=True, n_snapshots=20)

    result = runner.invoke(
        cli_main,
        ["backtest", "--db-path", str(db_path), "--market-ids", "cli_sp_0,cli_sp_1"],
    )
    assert result.exit_code == 0
    assert "Replay Report" in result.output


def test_cli_report_command(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "cli_report.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        create_synthetic_market(conn, "cli_rpt_001", resolved_yes=True, n_snapshots=20)

    bt_result = runner.invoke(cli_main, ["backtest", "--db-path", str(db_path)])
    assert bt_result.exit_code == 0

    report_result = runner.invoke(cli_main, ["report", "--db-path", str(db_path), "--experiment-id", "1"])
    assert report_result.exit_code == 0
    assert "Replay Report" in report_result.output


def test_cli_backtest_no_data(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "cli_empty.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)

    result = runner.invoke(cli_main, ["backtest", "--db-path", str(db_path)])
    assert result.exit_code != 0
    assert "No replay data" in result.output


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--help"])
    assert result.exit_code == 0
    assert "Run Polymarket backtest workflows" in result.output

    for command in ("download", "ingest", "backtest", "report", "demo"):
        cmd_result = runner.invoke(cli_main, [command, "--help"])
        assert cmd_result.exit_code == 0


# -- Edge cases --


def test_minimal_snapshots(tmp_path: Path) -> None:
    db_path = tmp_path / "minimal.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        create_synthetic_market(conn, "minimal_001", resolved_yes=True, n_snapshots=3)

        replay = _build_replay_run(conn, experiment_name="minimal_test", single_market_id="minimal_001")
        assert isinstance(replay.experiment_id, int)
        assert replay.experiment_id > 0
        for section in REPORT_SECTIONS:
            assert section in replay.report


def test_all_markets_resolve_no(tmp_path: Path) -> None:
    db_path = tmp_path / "all_no.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        for i in range(5):
            create_synthetic_market(conn, f"no_{i}", resolved_yes=False, n_snapshots=20)

        replay = _build_replay_run(conn, experiment_name="all_no_test")
        assert isinstance(replay.experiment_id, int)
        brier_items = replay.summary["brier"]
        assert len(brier_items) > 0
        for section in REPORT_SECTIONS:
            assert section in replay.report


def test_many_markets_stress(tmp_path: Path) -> None:
    db_path = tmp_path / "stress.sqlite"
    with closing(db.connect(db_path)) as conn:
        db.init_db(conn)
        for i in range(100):
            create_synthetic_market(
                conn,
                f"stress_{i:03d}",
                resolved_yes=i % 2 == 0,
                n_snapshots=10,
            )

        replay = _build_replay_run(conn, experiment_name="stress_test")

        model_count = _scalar(
            conn,
            "SELECT COUNT(*) FROM model_outputs WHERE experiment_id = ?",
            (replay.experiment_id,),
        )
        assert model_count >= 100

        for section in REPORT_SECTIONS:
            assert section in replay.report
