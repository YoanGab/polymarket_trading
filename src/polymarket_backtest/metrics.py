from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Any

from . import db
from .types import ensure_utc, isoformat


def _rows(conn: sqlite3.Connection, query: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    return list(conn.execute(query, args).fetchall())


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def compute_brier_scores(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    rows = _rows(
        conn,
        """
        SELECT o.agent_name, o.domain, o.probability_yes, r.resolved_outcome
        FROM model_outputs o
        JOIN market_resolutions r ON r.market_id = o.market_id
        WHERE o.experiment_id = ?
        """,
        (experiment_id,),
    )
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["agent_name"]), str(row["domain"]))
        error = (float(row["probability_yes"]) - float(row["resolved_outcome"])) ** 2
        buckets[key].append(error)
    results = []
    for (agent_name, domain), values in sorted(buckets.items()):
        results.append(
            {
                "agent_name": agent_name,
                "domain": domain,
                "brier_score": round(_mean(values), 6),
                "n": len(values),
            }
        )
    return results


def compute_markouts(
    conn: sqlite3.Connection,
    experiment_id: int,
    horizons: tuple[int, ...],
) -> dict[int, list[float]]:
    rows = _rows(
        conn,
        """
        SELECT market_id, fill_ts, side, price
        FROM fills
        WHERE experiment_id = ?
        """,
        (experiment_id,),
    )
    results: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        fill_ts = datetime.fromisoformat(str(row["fill_ts"]))
        for horizon in horizons:
            target_ts = fill_ts + timedelta(minutes=horizon)
            future = db.get_market_state_as_of(conn, str(row["market_id"]), target_ts)
            if future is None:
                continue
            signed = future.mid - float(row["price"])
            if str(row["side"]) == "sell":
                signed *= -1.0
            results[horizon].append(round(signed, 6))
    return results


def compute_fill_ratio(conn: sqlite3.Connection, experiment_id: int) -> dict[str, float]:
    row = conn.execute(
        """
        SELECT SUM(filled_quantity) AS filled, SUM(requested_quantity) AS requested
        FROM orders
        WHERE experiment_id = ?
        """,
        (experiment_id,),
    ).fetchone()
    filled = float(row["filled"] or 0.0)
    requested = float(row["requested"] or 0.0)
    return {
        "filled_quantity": round(filled, 4),
        "requested_quantity": round(requested, 4),
        "fill_ratio": round(filled / requested, 6) if requested else 0.0,
    }


def compute_adverse_selection(markouts: dict[int, list[float]]) -> dict[int, float]:
    return {
        horizon: round(sum(1 for value in values if value < 0) / len(values), 6)
        for horizon, values in markouts.items()
        if values
    }


def compute_calibration_curve(conn: sqlite3.Connection, experiment_id: int, bins: int = 10) -> list[dict[str, Any]]:
    rows = _rows(
        conn,
        """
        SELECT probability_yes, market_id
        FROM model_outputs
        WHERE experiment_id = ?
        """,
        (experiment_id,),
    )
    if not rows:
        return []
    outcomes = {
        str(row["market_id"]): float(row["resolved_outcome"])
        for row in _rows(
            conn,
            "SELECT market_id, resolved_outcome FROM market_resolutions",
        )
    }
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        probability = float(row["probability_yes"])
        bucket = min(bins - 1, int(probability * bins))
        buckets[bucket].append((probability, outcomes[str(row["market_id"])]))
    curve = []
    for bucket in range(bins):
        values = buckets.get(bucket, [])
        if not values:
            continue
        curve.append(
            {
                "bucket": bucket,
                "forecast_mean": round(_mean([p for p, _ in values]), 4),
                "realized_rate": round(_mean([y for _, y in values]), 4),
                "n": len(values),
            }
        )
    return curve


def compute_pnl_summary(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    rows = _rows(
        conn,
        """
        SELECT strategy_name, realized_pnl_pre_resolution, hold_to_resolution_pnl
        FROM positions
        WHERE experiment_id = ?
        """,
        (experiment_id,),
    )
    by_strategy: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strategy = str(row["strategy_name"])
        by_strategy[strategy]["pre"].append(float(row["realized_pnl_pre_resolution"]))
        by_strategy[strategy]["hold"].append(float(row["hold_to_resolution_pnl"]))
    results = []
    for strategy, values in sorted(by_strategy.items()):
        results.append(
            {
                "strategy_name": strategy,
                "pnl_pre_resolution": round(sum(values["pre"]), 4),
                "pnl_hold_to_resolution": round(sum(values["hold"]), 4),
            }
        )
    return results


def compute_sharpe_like(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    rows = _rows(
        conn,
        """
        SELECT strategy_name, ts, equity
        FROM pnl_marks
        WHERE experiment_id = ?
        ORDER BY strategy_name, ts
        """,
        (experiment_id,),
    )
    equity_by_strategy: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        equity_by_strategy[str(row["strategy_name"])].append(float(row["equity"]))
    results = []
    for strategy, equity_values in sorted(equity_by_strategy.items()):
        returns = []
        for previous, current in zip(equity_values, equity_values[1:]):
            if previous > 0:
                returns.append((current - previous) / previous)
        if not returns:
            sharpe = 0.0
        else:
            sigma = pstdev(returns) if len(returns) > 1 else 0.0
            sharpe = (_mean(returns) / sigma) if sigma > 0 else 0.0
        results.append({"strategy_name": strategy, "sharpe_like": round(sharpe, 6)})
    return results


def compute_edge_decay(conn: sqlite3.Connection, experiment_id: int, horizons: tuple[int, ...]) -> dict[int, float]:
    markouts = compute_markouts(conn, experiment_id, horizons)
    edges = [
        float(row["expected_edge_bps"])
        for row in _rows(
            conn,
            "SELECT expected_edge_bps FROM model_outputs WHERE experiment_id = ?",
            (experiment_id,),
        )
    ]
    initial_edge = _mean(edges) / 10_000.0 if edges else 0.0
    decay = {}
    for horizon, values in markouts.items():
        realized = _mean(values)
        decay[horizon] = round(realized / initial_edge, 6) if initial_edge else 0.0
    return decay


def bootstrap_mean_confidence_interval(values: list[float], repeats: int = 500) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    samples: list[float] = []
    count = len(values)
    for repeat in range(repeats):
        sample = [values[(repeat + index * 7) % count] for index in range(count)]
        samples.append(_mean(sample))
    samples.sort()
    lower = samples[int(0.025 * len(samples))]
    upper = samples[int(0.975 * len(samples))]
    return (round(lower, 6), round(upper, 6))


def persist_metric_results(conn: sqlite3.Connection, experiment_id: int, replay_config_horizons: tuple[int, ...]) -> None:
    for item in compute_brier_scores(conn, experiment_id):
        conn.execute(
            """
            INSERT INTO metric_results (
                experiment_id, scope_type, scope_name, metric_name, metric_value,
                sample_size, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                "agent_domain",
                f"{item['agent_name']}:{item['domain']}",
                "brier_score",
                item["brier_score"],
                item["n"],
                "{}",
            ),
        )
    markouts = compute_markouts(conn, experiment_id, replay_config_horizons)
    adverse = compute_adverse_selection(markouts)
    for horizon, values in markouts.items():
        interval = bootstrap_mean_confidence_interval(values)
        conn.execute(
            """
            INSERT INTO metric_results (
                experiment_id, scope_type, scope_name, metric_name, horizon_min,
                metric_value, sample_size, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                "portfolio",
                "all",
                "markout",
                horizon,
                round(_mean(values), 6),
                len(values),
                json.dumps({"bootstrap_ci": interval}),
            ),
        )
        conn.execute(
            """
            INSERT INTO metric_results (
                experiment_id, scope_type, scope_name, metric_name, horizon_min,
                metric_value, sample_size, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                "portfolio",
                "all",
                "adverse_selection_rate",
                horizon,
                adverse.get(horizon, 0.0),
                len(values),
                "{}",
            ),
        )
    fill_ratio = compute_fill_ratio(conn, experiment_id)
    conn.execute(
        """
        INSERT INTO metric_results (
            experiment_id, scope_type, scope_name, metric_name, metric_value,
            sample_size, extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            experiment_id,
            "portfolio",
            "all",
            "fill_ratio",
            fill_ratio["fill_ratio"],
            1,
            json.dumps(fill_ratio),
        ),
    )
    conn.commit()


def build_metrics_summary(conn: sqlite3.Connection, experiment_id: int, horizons: tuple[int, ...]) -> dict[str, Any]:
    markouts = compute_markouts(conn, experiment_id, horizons)
    return {
        "brier": compute_brier_scores(conn, experiment_id),
        "markouts": {horizon: round(_mean(values), 6) for horizon, values in markouts.items()},
        "adverse_selection": compute_adverse_selection(markouts),
        "fill_ratio": compute_fill_ratio(conn, experiment_id),
        "pnl": compute_pnl_summary(conn, experiment_id),
        "calibration": compute_calibration_curve(conn, experiment_id),
        "sharpe_like": compute_sharpe_like(conn, experiment_id),
        "edge_decay": compute_edge_decay(conn, experiment_id, horizons),
    }
