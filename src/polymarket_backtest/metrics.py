from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Any

import scoringrules as sr

from . import db

logger = logging.getLogger(__name__)


def _rows(conn: sqlite3.Connection, query: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    return list(conn.execute(query, args).fetchall())


def _mean(values: list[int | float]) -> float:
    return mean(values) if values else 0.0


def _clip_probability(probability: float) -> float:
    return min(0.999, max(0.001, probability))


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


def compute_log_score(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
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
    if not rows:
        return []
    observations = [float(row["resolved_outcome"]) for row in rows]
    forecasts = [_clip_probability(float(row["probability_yes"])) for row in rows]
    scores = [-float(score) for score in sr.log_score(observations, forecasts)]
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=False):
        key = (str(row["agent_name"]), str(row["domain"]))
        buckets[key].append(score)
    results = []
    for (agent_name, domain), values in sorted(buckets.items()):
        results.append(
            {
                "agent_name": agent_name,
                "domain": domain,
                "log_score": round(_mean(values), 6),
                "n": len(values),
            }
        )
    return results


def compute_market_brier_comparison(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    rows = _rows(
        conn,
        """
        SELECT
            o.agent_name,
            o.domain,
            o.market_id,
            m.title AS market_title,
            o.ts,
            o.probability_yes,
            r.resolved_outcome,
            (
                -- Use snapshot from 1 hour BEFORE the forecast time to mitigate
                -- timing bias: the agent may have incorporated news that the
                -- market price had not yet reflected at forecast time. Lagging
                -- the market snapshot by 1 hour gives the market time to
                -- incorporate the same information the agent used.
                SELECT s.mid
                FROM market_snapshots s
                WHERE s.market_id = o.market_id
                  AND s.ts <= datetime(o.ts, '-1 hour')
                ORDER BY s.ts DESC
                LIMIT 1
            ) AS market_mid
        FROM model_outputs o
        JOIN markets m ON m.market_id = o.market_id
        JOIN market_resolutions r ON r.market_id = o.market_id
        WHERE o.experiment_id = ?
        ORDER BY o.agent_name, o.domain, o.market_id, o.ts
        """,
        (experiment_id,),
    )
    comparable_rows = [row for row in rows if row["market_mid"] is not None]
    if not comparable_rows:
        return []
    observations = [float(row["resolved_outcome"]) for row in comparable_rows]
    agent_forecasts = [float(row["probability_yes"]) for row in comparable_rows]
    market_forecasts = [float(row["market_mid"]) for row in comparable_rows]
    agent_scores = sr.brier_score(observations, agent_forecasts)
    market_scores = sr.brier_score(observations, market_forecasts)
    results = []
    for row, agent_score, market_score in zip(
        comparable_rows,
        agent_scores,
        market_scores,
        strict=False,
    ):
        agent_brier = round(float(agent_score), 6)
        market_brier = round(float(market_score), 6)
        brier_improvement = round(market_brier - agent_brier, 6)
        results.append(
            {
                "agent_name": str(row["agent_name"]),
                "domain": str(row["domain"]),
                "market_id": str(row["market_id"]),
                "market_title": str(row["market_title"]),
                "ts": str(row["ts"]),
                "agent_probability": round(float(row["probability_yes"]), 6),
                "market_mid": round(float(row["market_mid"]), 6),
                "resolved_outcome": float(row["resolved_outcome"]),
                "agent_brier": agent_brier,
                "market_brier": market_brier,
                "brier_improvement": brier_improvement,
                "agent_better": agent_brier < market_brier,
                "timing_note": (
                    "Market snapshot is lagged 1 hour before forecast time to "
                    "reduce look-ahead bias. The agent may still have an "
                    "information advantage if news propagates slowly to the market."
                ),
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
        market_id = str(row["market_id"])
        if market_id not in outcomes:
            continue
        probability = float(row["probability_yes"])
        bucket = min(bins - 1, int(probability * bins))
        buckets[bucket].append((probability, outcomes[market_id]))
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


def _load_experiment_end_ts(conn: sqlite3.Connection, experiment_id: int) -> datetime | None:
    row = conn.execute(
        """
        SELECT MAX(ts) AS end_ts
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
    if row is None or row["end_ts"] is None:
        return None
    return datetime.fromisoformat(str(row["end_ts"]))


def compute_trade_pnl_details(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    experiment_end_ts = _load_experiment_end_ts(conn, experiment_id)
    if experiment_end_ts is None:
        return []

    rows = _rows(
        conn,
        """
        SELECT
            f.strategy_name,
            f.market_id,
            m.title AS market_title,
            m.domain,
            SUM(CASE WHEN f.side = 'buy' THEN f.quantity ELSE 0 END) AS buy_quantity,
            SUM(CASE WHEN f.side = 'sell' THEN f.quantity ELSE 0 END) AS sell_quantity,
            SUM(CASE WHEN f.side = 'buy' THEN f.price * f.quantity ELSE 0 END) AS buy_notional,
            SUM(CASE WHEN f.side = 'sell' THEN f.price * f.quantity ELSE 0 END) AS sell_notional,
            SUM(f.fee_usdc) AS fees_paid,
            SUM(f.rebate_usdc) AS rebates_earned
        FROM fills f
        JOIN markets m ON m.market_id = f.market_id
        WHERE f.experiment_id = ?
        GROUP BY f.strategy_name, f.market_id, m.title, m.domain
        ORDER BY f.strategy_name, f.market_id
        """,
        (experiment_id,),
    )
    results = []
    for row in rows:
        market_id = str(row["market_id"])
        resolution = db.get_resolution(conn, market_id)
        terminal_source = "mark"
        terminal_price = 0.0
        if resolution is not None and resolution["resolution_ts"] is not None:
            resolution_ts = datetime.fromisoformat(str(resolution["resolution_ts"]))
            if resolution_ts <= experiment_end_ts:
                terminal_price = float(resolution["resolved_outcome"])
                terminal_source = "resolution"
        if terminal_source == "mark":
            market_state = db.get_market_state_as_of(conn, market_id, experiment_end_ts)
            if market_state is not None:
                terminal_price = float(market_state.mid)
            else:
                snapshot_row = conn.execute(
                    """
                    SELECT mid
                    FROM market_snapshots
                    WHERE market_id = ?
                    ORDER BY ts DESC
                    LIMIT 1
                    """,
                    (market_id,),
                ).fetchone()
                if snapshot_row is None:
                    terminal_source = "missing_mark"
                    # Fallback: use the last known fill price for this market/strategy
                    last_fill_row = conn.execute(
                        """
                        SELECT price
                        FROM fills
                        WHERE experiment_id = ? AND market_id = ? AND strategy_name = ?
                        ORDER BY fill_ts DESC
                        LIMIT 1
                        """,
                        (experiment_id, market_id, str(row["strategy_name"])),
                    ).fetchone()
                    if last_fill_row is not None:
                        terminal_price = float(last_fill_row["price"])
                        logger.warning(
                            "Missing mark for market %s (strategy %s): using last fill price %.6f",
                            market_id,
                            str(row["strategy_name"]),
                            terminal_price,
                        )
                    else:
                        logger.warning(
                            "Missing mark for market %s (strategy %s): no fill price available, using 0.0",
                            market_id,
                            str(row["strategy_name"]),
                        )
                else:
                    terminal_price = float(snapshot_row["mid"])

        has_missing_marks = terminal_source == "missing_mark"
        buy_quantity = float(row["buy_quantity"] or 0.0)
        sell_quantity = float(row["sell_quantity"] or 0.0)
        buy_notional = float(row["buy_notional"] or 0.0)
        sell_notional = float(row["sell_notional"] or 0.0)
        fees_paid = float(row["fees_paid"] or 0.0)
        rebates_earned = float(row["rebates_earned"] or 0.0)
        net_quantity = buy_quantity - sell_quantity
        cash_pnl = sell_notional - buy_notional - fees_paid + rebates_earned
        terminal_value = net_quantity * terminal_price
        results.append(
            {
                "strategy_name": str(row["strategy_name"]),
                "market_id": market_id,
                "market_title": str(row["market_title"]),
                "domain": str(row["domain"]),
                "buy_quantity": round(buy_quantity, 4),
                "sell_quantity": round(sell_quantity, 4),
                "net_quantity": round(net_quantity, 4),
                "terminal_price": round(terminal_price, 6),
                "terminal_source": terminal_source,
                "pnl": round(cash_pnl + terminal_value, 4),
                "has_missing_marks": has_missing_marks,
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
    equity_by_strategy: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in rows:
        equity_by_strategy[str(row["strategy_name"])].append(
            (datetime.fromisoformat(str(row["ts"])), float(row["equity"]))
        )
    results = []
    for strategy, ts_equity_values in sorted(equity_by_strategy.items()):
        timestamps = [ts for ts, _ in ts_equity_values]
        equity_values = [eq for _, eq in ts_equity_values]
        returns = []
        for previous, current in zip(equity_values, equity_values[1:], strict=False):
            if previous > 0:
                returns.append((current - previous) / previous)
        # Require at least 10 data points for a meaningful Sharpe ratio
        if len(returns) < 10:
            sharpe = 0.0
        else:
            mean_return = _mean(returns)
            std_return = stdev(returns)
            if std_return > 0 and len(timestamps) >= 2:
                # Approximate annualization based on average snapshot interval
                total_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
                avg_interval_seconds = total_seconds / (len(timestamps) - 1)
                periods_per_year = 365.25 * 24 * 3600 / max(avg_interval_seconds, 1)
                sharpe = (mean_return / std_return) * math.sqrt(periods_per_year)
            else:
                sharpe = 0.0
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


def bootstrap_mean_confidence_interval(values: list[float], repeats: int = 2000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    samples: list[float] = []
    count = len(values)
    rng = random.Random(42)
    for _ in range(repeats):
        sample = rng.choices(values, k=count)
        samples.append(mean(sample) if sample else 0.0)
    samples.sort()
    lower = samples[int(0.025 * len(samples))]
    upper = samples[int(0.975 * len(samples))]
    return (round(lower, 6), round(upper, 6))


def persist_metric_results(
    conn: sqlite3.Connection, experiment_id: int, replay_config_horizons: tuple[int, ...]
) -> None:
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
    for item in compute_log_score(conn, experiment_id):
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
                "log_score",
                item["log_score"],
                item["n"],
                json.dumps({"clip_min": 0.001, "clip_max": 0.999}),
            ),
        )
    for item in compute_market_brier_comparison(conn, experiment_id):
        conn.execute(
            """
            INSERT INTO metric_results (
                experiment_id, scope_type, scope_name, metric_name, metric_value,
                sample_size, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                "agent_market",
                f"{item['agent_name']}:{item['market_id']}:{item['ts']}",
                "brier_comparison",
                item["brier_improvement"],
                1,
                json.dumps(
                    {
                        "domain": item["domain"],
                        "market_id": item["market_id"],
                        "market_title": item["market_title"],
                        "ts": item["ts"],
                        "agent_probability": item["agent_probability"],
                        "market_mid": item["market_mid"],
                        "resolved_outcome": item["resolved_outcome"],
                        "agent_brier": item["agent_brier"],
                        "market_brier": item["market_brier"],
                        "agent_better": item["agent_better"],
                    }
                ),
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
        "log_score": compute_log_score(conn, experiment_id),
        "brier_comparison": compute_market_brier_comparison(conn, experiment_id),
        "markouts": {horizon: round(_mean(values), 6) for horizon, values in markouts.items()},
        "adverse_selection": compute_adverse_selection(markouts),
        "fill_ratio": compute_fill_ratio(conn, experiment_id),
        "pnl": compute_pnl_summary(conn, experiment_id),
        "trade_pnl": compute_trade_pnl_details(conn, experiment_id),
        "calibration": compute_calibration_curve(conn, experiment_id),
        "sharpe_like": compute_sharpe_like(conn, experiment_id),
        "edge_decay": compute_edge_decay(conn, experiment_id, horizons),
    }
