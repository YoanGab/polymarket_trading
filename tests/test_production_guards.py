from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path

from polymarket_backtest.production_guards import TradingGuards


def test_check_staleness_rejects_old_data() -> None:
    guards = TradingGuards(max_stale_minutes=30)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    ok, message = guards.check_staleness(now - timedelta(minutes=31), now)

    assert not ok
    assert message == "Data is 31min old (max 30)"


def test_check_drawdown_uses_starting_cash_as_initial_peak() -> None:
    guards = TradingGuards(max_drawdown_pct=0.20)

    ok, message = guards.check_drawdown(current_equity=75.0, starting_cash=100.0)

    assert not ok
    assert message == "Drawdown 25.0% exceeds 20.0%"


def test_check_feature_distribution_flags_large_z_scores() -> None:
    guards = TradingGuards(feature_shift_threshold=3.0)
    guards._training_feature_stats = {"mid": (0.50, 0.05), "spread": (0.02, 0.01)}

    ok, message = guards.check_feature_distribution({"mid": 0.75, "spread": 0.02})

    assert not ok
    assert message == "Feature shift: mid: z=5.0"


def test_check_all_aggregates_guard_failures() -> None:
    guards = TradingGuards(max_stale_minutes=30, max_drawdown_pct=0.20, feature_shift_threshold=3.0)
    guards._training_feature_stats = {"mid": (0.50, 0.05)}
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    ok, warnings = guards.check_all(
        last_snapshot_ts=now - timedelta(minutes=45),
        now=now,
        current_equity=70.0,
        starting_cash=100.0,
        features={"mid": 0.75},
    )

    assert not ok
    assert warnings == [
        "Data is 45min old (max 30)",
        "Drawdown 30.0% exceeds 20.0%",
        "Feature shift: mid: z=5.0",
    ]


def test_load_training_stats_reads_stats_from_pickle(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(
            {
                "model": {"kind": "test"},
                "feature_names": ["mid", "spread"],
                "training_feature_stats": {
                    "mid": {"mean": 0.51, "std": 0.04},
                    "spread": [0.02, 0.01],
                },
            },
            f,
        )

    guards = TradingGuards()
    stats = guards.load_training_stats(model_path)

    assert stats == {"mid": (0.51, 0.04), "spread": (0.02, 0.01)}


def test_load_training_stats_reads_stats_from_sidecar_json(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"model": {"kind": "test"}, "feature_names": ["mid"]}, f)

    stats_path = tmp_path / "model_stats.json"
    with stats_path.open("w") as f:
        json.dump({"training_feature_stats": {"mid": {"mean": 0.5, "std": 0.1}}}, f)

    guards = TradingGuards()
    stats = guards.load_training_stats(model_path)

    assert stats == {"mid": (0.5, 0.1)}
