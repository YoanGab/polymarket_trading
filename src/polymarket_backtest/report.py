from __future__ import annotations

import random
from collections import defaultdict
from math import log
from typing import Any


class ReportGenerator:
    def build_markdown(
        self,
        summary: dict[str, Any],
        experiment_config: dict[str, Any] | None = None,
    ) -> str:
        config = experiment_config or {}
        strategy_performance = self._strategy_performance(summary)
        brier_markets = self._aggregate_market_scores(
            summary.get("brier_comparison", []),
            agent_key="agent_brier",
            market_key="market_brier",
            improvement_key="brier_improvement",
        )
        log_markets = self._aggregate_market_log_scores(summary.get("brier_comparison", []))
        go_no_go_checks, verdict = self._go_no_go_checks(summary, brier_markets)
        calibration_error = self._calibration_error(summary.get("calibration", []))
        trade_pnls = [float(item.get("pnl", 0.0)) for item in summary.get("trade_pnl", [])]
        brier_ci = self._bootstrap_mean_confidence_interval([item["improvement"] for item in brier_markets])
        pnl_ci = self._bootstrap_total_confidence_interval(trade_pnls)

        lines = ["# Replay Report", "", "## Summary"]
        lines.extend(self._summary_lines(summary, config, strategy_performance))

        lines.extend(["", "## GO / NO-GO Assessment"])
        for check in go_no_go_checks:
            lines.append(f"- {check['label']}: {check['status']} ({check['detail']})")
        lines.append(f"- Overall verdict: **{verdict}**")

        lines.extend(["", "## Forecast Quality", "", "### Agent vs Market Brier"])
        lines.extend(self._brier_section(summary, brier_markets))

        lines.extend(["", "### Log Score"])
        lines.extend(self._log_score_section(summary, log_markets))

        lines.extend(["", "## Execution"])
        lines.extend(self._execution_section(summary))

        lines.extend(["", "## PnL"])
        lines.extend(self._pnl_section(strategy_performance))

        lines.extend(["", "## Calibration"])
        lines.extend(self._calibration_section(summary.get("calibration", []), calibration_error))

        lines.extend(["", "## Bootstrap Confidence Intervals"])
        lines.extend(self._bootstrap_section(pnl_ci, len(trade_pnls), brier_ci, len(brier_markets)))

        lines.extend(["", "## Risk-Adjusted"])
        lines.extend(self._risk_adjusted_section(summary.get("sharpe_like", [])))

        lines.extend(["", "## Edge Decay"])
        lines.extend(self._edge_decay_section(summary.get("edge_decay", {})))

        return "\n".join(lines)

    def _summary_lines(
        self,
        summary: dict[str, Any],
        config: dict[str, Any],
        strategy_performance: list[dict[str, Any]],
    ) -> list[str]:
        lines: list[str] = []
        experiment_name = config.get("experiment_name")
        if experiment_name:
            lines.append(f"- Experiment: {experiment_name}")

        date_range = self._date_range_text(config.get("date_start"), config.get("date_end"))
        if date_range:
            lines.append(f"- Date range: {date_range}")

        market_count = config.get("market_count")
        if market_count in {None, 0}:
            market_count = len(
                {
                    str(item.get("market_id"))
                    for item in [*summary.get("brier_comparison", []), *summary.get("trade_pnl", [])]
                    if item.get("market_id") is not None
                }
            )

        trade_count = config.get("trade_count")
        if trade_count in {None, 0}:
            trade_count = len(summary.get("trade_pnl", []))

        lines.append(f"- Markets: {market_count or 0}")
        lines.append(f"- Trades: {trade_count or 0}")

        per_strategy_start = self._float_or_none(config.get("starting_cash"))
        total_start = self._starting_cash_total(config, strategy_performance)
        if total_start is not None:
            if per_strategy_start is not None and int(config.get("strategy_count") or 0) > 1:
                lines.append(
                    "- Starting cash: "
                    f"{self._currency(total_start)} total "
                    f"({self._currency(per_strategy_start)} per strategy)"
                )
            else:
                lines.append(f"- Starting cash: {self._currency(total_start)}")

        if strategy_performance:
            total_pnl = sum(item["pnl"] for item in strategy_performance)
        elif summary.get("pnl"):
            total_pnl = sum(float(item.get("pnl_pre_resolution", 0.0)) for item in summary["pnl"])
        else:
            total_pnl = 0.0

        ending_equity = total_start + total_pnl if total_start is not None else None
        if ending_equity is not None:
            lines.append(f"- Ending equity: {self._currency(ending_equity)}")
        lines.append(f"- Total PnL: {self._currency(total_pnl, signed=True)}")
        if total_start not in {None, 0.0}:
            lines.append(f"- Total return: {self._percent(total_pnl / total_start, signed=True)}")

        return lines

    def _go_no_go_checks(
        self,
        summary: dict[str, Any],
        brier_markets: list[dict[str, Any]],
    ) -> tuple[list[dict[str, str]], str]:
        checks: list[dict[str, str]] = []

        if brier_markets:
            brier_improvements = [item["improvement"] for item in brier_markets]
            brier_better = sum(1 for value in brier_improvements if value > 0)
            brier_avg = self._safe_mean(brier_improvements)
            brier_status = "PASS" if brier_avg > 0 else "FAIL"
            checks.append(
                {
                    "label": "Brier(agent) < Brier(market) on resolved markets",
                    "status": brier_status,
                    "detail": f"{brier_better}/{len(brier_markets)} markets better, avg {self._bps(brier_avg)}",
                }
            )
        else:
            checks.append(
                {
                    "label": "Brier(agent) < Brier(market) on resolved markets",
                    "status": "INSUFFICIENT",
                    "detail": "No resolved market comparisons available",
                }
            )

        trade_pnls = [float(item.get("pnl", 0.0)) for item in summary.get("trade_pnl", [])]
        pnl_ci = self._bootstrap_total_confidence_interval(trade_pnls)
        if pnl_ci is None:
            checks.append(
                {
                    "label": "Bootstrap CI(PnL) > 0",
                    "status": "INSUFFICIENT",
                    "detail": "No executed trade PnL samples",
                }
            )
        else:
            pnl_status = "PASS" if pnl_ci[0] > 0 else "FAIL"
            checks.append(
                {
                    "label": "Bootstrap CI(PnL) > 0",
                    "status": pnl_status,
                    "detail": (
                        f"95% CI {self._currency(pnl_ci[0], signed=True)} to {self._currency(pnl_ci[1], signed=True)}"
                    ),
                }
            )

        adverse = summary.get("adverse_selection", {})
        if adverse:
            first_horizon = min(adverse)
            adverse_value = float(adverse[first_horizon])
            adverse_status = "PASS" if adverse_value < 0.55 else "FAIL"
            checks.append(
                {
                    "label": "Adverse selection < 55%",
                    "status": adverse_status,
                    "detail": f"{first_horizon}m adverse selection {self._percent(adverse_value)}",
                }
            )
        else:
            checks.append(
                {
                    "label": "Adverse selection < 55%",
                    "status": "INSUFFICIENT",
                    "detail": "No markout horizons available",
                }
            )

        fill_ratio = self._float_or_none(summary.get("fill_ratio", {}).get("fill_ratio"))
        if fill_ratio is None:
            checks.append(
                {
                    "label": "Fill ratio > 60%",
                    "status": "INSUFFICIENT",
                    "detail": "No fill ratio data",
                }
            )
        else:
            fill_status = "PASS" if fill_ratio > 0.60 else "FAIL"
            checks.append(
                {
                    "label": "Fill ratio > 60%",
                    "status": fill_status,
                    "detail": self._percent(fill_ratio),
                }
            )

        domain_stats = self._domain_improvement_stats(brier_markets)
        if len(domain_stats) < 2:
            checks.append(
                {
                    "label": "Edge stability across domains",
                    "status": "INSUFFICIENT",
                    "detail": "Need at least two domains with resolved comparisons",
                }
            )
        else:
            stable = all(item["avg_improvement"] > 0 for item in domain_stats)
            checks.append(
                {
                    "label": "Edge stability across domains",
                    "status": "PASS" if stable else "FAIL",
                    "detail": ", ".join(
                        f"{item['domain']} {self._bps(item['avg_improvement'])}" for item in domain_stats
                    ),
                }
            )

        statuses = [check["status"] for check in checks]
        if "INSUFFICIENT" in statuses:
            verdict = "INSUFFICIENT DATA"
        elif all(status == "PASS" for status in statuses):
            verdict = "GO"
        else:
            verdict = "NO-GO"
        return checks, verdict

    def _brier_section(
        self,
        summary: dict[str, Any],
        brier_markets: list[dict[str, Any]],
    ) -> list[str]:
        lines: list[str] = []
        brier_rows = summary.get("brier", [])
        if brier_rows:
            table_rows = [
                [
                    str(item.get("agent_name", "-")),
                    str(item.get("domain", "-")),
                    self._decimal(float(item.get("brier_score", 0.0)), digits=4),
                    str(item.get("n", 0)),
                ]
                for item in brier_rows
            ]
            lines.extend(self._markdown_table(["Agent", "Domain", "Brier", "n"], table_rows))
        else:
            lines.append("_No Brier score data available._")

        if not brier_markets:
            lines.extend(["", "_No agent-vs-market Brier comparison data available._"])
            return lines

        average_improvement = self._safe_mean([item["improvement"] for item in brier_markets])
        agent_better = sum(1 for item in brier_markets if item["improvement"] > 0)
        lines.extend(
            [
                "",
                f"- Agent better in {agent_better}/{len(brier_markets)} markets",
                f"- Average improvement: {self._bps(average_improvement)}",
                "",
                "Best markets by improvement:",
            ]
        )
        lines.extend(self._market_table(brier_markets[:3], score_label="Brier"))
        lines.extend(["", "Worst markets by improvement:"])
        lines.extend(self._market_table(list(reversed(brier_markets[-3:])), score_label="Brier"))
        return lines

    def _log_score_section(
        self,
        summary: dict[str, Any],
        log_markets: list[dict[str, Any]],
    ) -> list[str]:
        lines: list[str] = []
        log_rows = summary.get("log_score", [])
        if log_rows:
            table_rows = [
                [
                    str(item.get("agent_name", "-")),
                    str(item.get("domain", "-")),
                    self._decimal(float(item.get("log_score", 0.0)), digits=4),
                    str(item.get("n", 0)),
                ]
                for item in log_rows
            ]
            lines.extend(self._markdown_table(["Agent", "Domain", "Log Score", "n"], table_rows))
        else:
            lines.append("_No log score data available._")

        if not log_markets:
            lines.extend(["", "_No agent-vs-market log score comparison data available._"])
            return lines

        average_improvement = self._safe_mean([item["improvement"] for item in log_markets])
        agent_better = sum(1 for item in log_markets if item["improvement"] > 0)
        lines.extend(
            [
                "",
                f"- Agent better in {agent_better}/{len(log_markets)} markets",
                f"- Average improvement: {self._decimal(average_improvement, digits=4, signed=True)}",
                "",
                "Best markets by improvement:",
            ]
        )
        lines.extend(self._market_table(log_markets[:3], score_label="Log"))
        lines.extend(["", "Worst markets by improvement:"])
        lines.extend(self._market_table(list(reversed(log_markets[-3:])), score_label="Log"))
        return lines

    def _execution_section(self, summary: dict[str, Any]) -> list[str]:
        markouts = summary.get("markouts", {})
        adverse = summary.get("adverse_selection", {})
        fill_ratio = summary.get("fill_ratio", {})
        lines: list[str] = []

        if markouts:
            rows = []
            for horizon, value in sorted(markouts.items()):
                adverse_value = adverse.get(horizon)
                rows.append(
                    [
                        f"{horizon}m",
                        self._decimal(float(value), digits=4, signed=True),
                        self._percent(float(adverse_value)) if adverse_value is not None else "-",
                    ]
                )
            lines.extend(self._markdown_table(["Horizon", "Avg Markout", "Adverse Selection"], rows))
        else:
            lines.append("_No execution markout data available._")

        filled = self._float_or_none(fill_ratio.get("filled_quantity")) or 0.0
        requested = self._float_or_none(fill_ratio.get("requested_quantity")) or 0.0
        fill_value = self._float_or_none(fill_ratio.get("fill_ratio"))
        lines.extend(
            [
                "",
                f"- Fill ratio: {self._percent(fill_value or 0.0)} ({filled:.2f}/{requested:.2f})",
            ]
        )
        return lines

    def _pnl_section(self, strategy_performance: list[dict[str, Any]]) -> list[str]:
        if not strategy_performance:
            return ["_No strategy PnL data available._"]

        rows = []
        for rank, item in enumerate(strategy_performance, start=1):
            rows.append(
                [
                    str(rank),
                    item["strategy_name"],
                    self._currency(item["pnl"], signed=True),
                    self._decimal(item["sharpe"], digits=4, signed=True),
                    str(item["trades"]),
                    self._percent(item["win_rate"]),
                ]
            )
        return self._markdown_table(["Rank", "Strategy", "PnL", "Sharpe", "Trades", "Win Rate"], rows)

    def _calibration_section(
        self,
        calibration: list[dict[str, Any]],
        calibration_error: float | None,
    ) -> list[str]:
        if not calibration:
            return ["_No calibration data available._"]

        lines = []
        if calibration_error is not None:
            lines.append(f"- Calibration error (mean absolute bucket gap): {self._percent(calibration_error)}")
            lines.append("")

        rows = []
        for bucket in calibration:
            lower = float(bucket["bucket"]) / 10.0
            upper = lower + 0.1
            gap = float(bucket["realized_rate"]) - float(bucket["forecast_mean"])
            rows.append(
                [
                    f"{lower:.1f}-{upper:.1f}",
                    self._percent(float(bucket["forecast_mean"])),
                    self._percent(float(bucket["realized_rate"])),
                    self._percent(abs(gap)),
                    str(bucket["n"]),
                ]
            )
        lines.extend(self._markdown_table(["Bucket", "Forecast Mean", "Realized Rate", "Abs Gap", "n"], rows))
        return lines

    def _bootstrap_section(
        self,
        pnl_ci: tuple[float, float] | None,
        pnl_n: int,
        brier_ci: tuple[float, float] | None,
        brier_n: int,
    ) -> list[str]:
        lines: list[str] = []
        if pnl_ci is None:
            lines.append("- PnL CI: insufficient data")
        else:
            lines.append(
                "- PnL CI (95%): "
                f"{self._currency(pnl_ci[0], signed=True)} to {self._currency(pnl_ci[1], signed=True)} "
                f"across {pnl_n} trades"
            )

        if brier_ci is None:
            lines.append("- Brier improvement CI: insufficient data")
        else:
            lines.append(
                "- Brier improvement CI (95%): "
                f"{self._bps(brier_ci[0])} to {self._bps(brier_ci[1])} "
                f"across {brier_n} markets"
            )
        return lines

    def _risk_adjusted_section(self, sharpe_like: list[dict[str, Any]]) -> list[str]:
        if not sharpe_like:
            return ["_No Sharpe-like data available._"]
        rows = [
            [
                str(item.get("strategy_name", "-")),
                self._decimal(float(item.get("sharpe_like", 0.0)), digits=4, signed=True),
            ]
            for item in sharpe_like
        ]
        return self._markdown_table(["Strategy", "Sharpe-Like"], rows)

    def _edge_decay_section(self, edge_decay: dict[int, Any]) -> list[str]:
        if not edge_decay:
            return ["_No edge decay data available._"]
        rows = [[f"{horizon}m", self._decimal(float(value), digits=4)] for horizon, value in sorted(edge_decay.items())]
        return self._markdown_table(["Horizon", "Retention"], rows)

    def _strategy_performance(
        self,
        summary: dict[str, Any],
    ) -> list[dict[str, Any]]:
        trade_pnl = summary.get("trade_pnl", [])
        sharpe_by_strategy = {
            str(item.get("strategy_name")): float(item.get("sharpe_like", 0.0))
            for item in summary.get("sharpe_like", [])
        }
        grouped: dict[str, list[float]] = defaultdict(list)
        for item in trade_pnl:
            grouped[str(item.get("strategy_name"))].append(float(item.get("pnl", 0.0)))

        results = []
        for strategy_name in sorted(set(grouped) | set(sharpe_by_strategy)):
            trade_values = grouped.get(strategy_name, [])
            if trade_values:
                pnl_value = sum(trade_values)
                trades = len(trade_values)
                win_rate = sum(1 for value in trade_values if value > 0) / trades
            else:
                pnl_value = 0.0
                trades = 0
                win_rate = 0.0
            results.append(
                {
                    "strategy_name": strategy_name,
                    "pnl": round(pnl_value, 4),
                    "sharpe": sharpe_by_strategy.get(strategy_name, 0.0),
                    "trades": trades,
                    "win_rate": win_rate,
                }
            )

        if not results and summary.get("pnl"):
            for item in summary["pnl"]:
                strategy_name = str(item.get("strategy_name"))
                results.append(
                    {
                        "strategy_name": strategy_name,
                        "pnl": round(float(item.get("pnl_pre_resolution", 0.0)), 4),
                        "sharpe": sharpe_by_strategy.get(strategy_name, 0.0),
                        "trades": 0,
                        "win_rate": 0.0,
                    }
                )

        return sorted(results, key=lambda item: (-item["pnl"], -item["sharpe"], item["strategy_name"]))

    def _aggregate_market_log_scores(self, comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for item in comparisons:
            outcome = float(item.get("resolved_outcome", 0.0))
            agent_probability = self._clip_probability(float(item.get("agent_probability", 0.0)))
            market_probability = self._clip_probability(float(item.get("market_mid", 0.0)))
            agent_log_score = self._log_score(agent_probability, outcome)
            market_log_score = self._log_score(market_probability, outcome)
            rows.append(
                {
                    "agent_name": str(item.get("agent_name", "-")),
                    "domain": str(item.get("domain", "-")),
                    "market_id": str(item.get("market_id", "-")),
                    "market_title": str(item.get("market_title", item.get("market_id", "-"))),
                    "agent_log_score": agent_log_score,
                    "market_log_score": market_log_score,
                    "log_score_improvement": market_log_score - agent_log_score,
                }
            )
        return self._aggregate_market_scores(
            rows,
            agent_key="agent_log_score",
            market_key="market_log_score",
            improvement_key="log_score_improvement",
        )

    def _aggregate_market_scores(
        self,
        comparisons: list[dict[str, Any]],
        *,
        agent_key: str,
        market_key: str,
        improvement_key: str,
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in comparisons:
            key = (str(item.get("agent_name", "-")), str(item.get("market_id", "-")))
            bucket = grouped.setdefault(
                key,
                {
                    "agent_name": key[0],
                    "market_id": key[1],
                    "market_title": str(item.get("market_title", item.get("market_id", "-"))),
                    "domain": str(item.get("domain", "-")),
                    "agent_scores": [],
                    "market_scores": [],
                    "improvements": [],
                },
            )
            bucket["agent_scores"].append(float(item.get(agent_key, 0.0)))
            bucket["market_scores"].append(float(item.get(market_key, 0.0)))
            bucket["improvements"].append(float(item.get(improvement_key, 0.0)))

        aggregated = []
        for bucket in grouped.values():
            improvement = self._safe_mean(bucket["improvements"])
            aggregated.append(
                {
                    "agent_name": bucket["agent_name"],
                    "market_id": bucket["market_id"],
                    "market_title": bucket["market_title"],
                    "domain": bucket["domain"],
                    "agent_score": self._safe_mean(bucket["agent_scores"]),
                    "market_score": self._safe_mean(bucket["market_scores"]),
                    "improvement": improvement,
                    "agent_better": improvement > 0,
                }
            )
        return sorted(aggregated, key=lambda item: item["improvement"], reverse=True)

    def _market_table(self, markets: list[dict[str, Any]], *, score_label: str) -> list[str]:
        if not markets:
            return ["_No market-level comparison data available._"]
        rows = [
            [
                self._market_label(item),
                item["domain"],
                self._improvement_text(item["improvement"], score_label=score_label),
                self._decimal(item["agent_score"], digits=4),
                self._decimal(item["market_score"], digits=4),
            ]
            for item in markets
        ]
        return self._markdown_table(
            ["Market", "Domain", "Improvement", f"Agent {score_label}", f"Market {score_label}"],
            rows,
        )

    def _domain_improvement_stats(self, brier_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_domain: dict[str, list[float]] = defaultdict(list)
        for item in brier_markets:
            by_domain[item["domain"]].append(item["improvement"])
        stats = [
            {"domain": domain, "avg_improvement": self._safe_mean(values), "n": len(values)}
            for domain, values in sorted(by_domain.items())
        ]
        return stats

    def _calibration_error(self, calibration: list[dict[str, Any]]) -> float | None:
        if not calibration:
            return None
        gaps = [
            abs(float(bucket.get("forecast_mean", 0.0)) - float(bucket.get("realized_rate", 0.0)))
            for bucket in calibration
        ]
        return self._safe_mean(gaps)

    def _starting_cash_total(
        self,
        config: dict[str, Any],
        strategy_performance: list[dict[str, Any]],
    ) -> float | None:
        total_start = self._float_or_none(config.get("starting_cash_total"))
        if total_start is not None:
            return total_start
        per_strategy = self._float_or_none(config.get("starting_cash"))
        if per_strategy is None:
            return None
        strategy_count = int(config.get("strategy_count") or 0) or len(strategy_performance) or 1
        return per_strategy * strategy_count

    def _bootstrap_total_confidence_interval(self, values: list[float]) -> tuple[float, float] | None:
        mean_ci = self._bootstrap_mean_confidence_interval(values)
        if mean_ci is None:
            return None
        return (
            round(mean_ci[0] * len(values), 4),
            round(mean_ci[1] * len(values), 4),
        )

    def _bootstrap_mean_confidence_interval(
        self, values: list[float], repeats: int = 2000
    ) -> tuple[float, float] | None:
        if not values:
            return None
        if len(values) == 1:
            only_value = round(values[0], 6)
            return (only_value, only_value)
        samples: list[float] = []
        count = len(values)
        rng = random.Random(42)
        for _ in range(repeats):
            sample = rng.choices(values, k=count)
            samples.append(self._safe_mean(sample))
        samples.sort()
        lower = samples[int(0.025 * len(samples))]
        upper = samples[int(0.975 * len(samples))]
        return (round(lower, 6), round(upper, 6))

    def _markdown_table(self, headers: list[str], rows: list[list[str]]) -> list[str]:
        if not rows:
            return ["_No data available._"]
        lines = [
            f"| {' | '.join(headers)} |",
            f"| {' | '.join('---' for _ in headers)} |",
        ]
        for row in rows:
            lines.append(f"| {' | '.join(row)} |")
        return lines

    def _date_range_text(self, start: Any, end: Any) -> str | None:
        if start and end:
            return f"{start} to {end}"
        if start:
            return str(start)
        if end:
            return str(end)
        return None

    def _market_label(self, item: dict[str, Any]) -> str:
        title = str(item.get("market_title") or "").strip()
        market_id = str(item.get("market_id", "-"))
        if title and title != market_id:
            return f"{title} ({market_id})"
        return market_id

    def _improvement_text(self, value: float, *, score_label: str) -> str:
        if score_label == "Brier":
            return self._bps(value)
        return self._decimal(value, digits=4, signed=True)

    def _clip_probability(self, probability: float) -> float:
        return min(0.999, max(0.001, probability))

    def _log_score(self, probability: float, outcome: float) -> float:
        clipped_probability = self._clip_probability(probability)
        return -(outcome * log(clipped_probability) + (1.0 - outcome) * log(1.0 - clipped_probability))

    def _currency(self, value: float, *, signed: bool = False) -> str:
        return f"{value:+.2f}" if signed else f"{value:.2f}"

    def _percent(self, value: float, *, signed: bool = False) -> str:
        return f"{value:+.2%}" if signed else f"{value:.2%}"

    def _bps(self, value: float) -> str:
        return f"{value * 10_000:+.1f} bps"

    def _decimal(self, value: float, *, digits: int = 4, signed: bool = False) -> str:
        sign = "+" if signed else ""
        return f"{value:{sign}.{digits}f}"

    def _safe_mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _float_or_none(self, value: Any) -> float | None:
        if value is None:
            return None
        return float(value)
