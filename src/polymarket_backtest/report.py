from __future__ import annotations

from typing import Any


class ReportGenerator:
    def build_markdown(self, summary: dict[str, Any]) -> str:
        lines = [
            "# Replay Report",
            "",
            "## Forecast Quality",
        ]
        for item in summary["brier"]:
            lines.append(
                f"- {item['agent_name']} / {item['domain']}: "
                f"Brier={item['brier_score']:.4f} on n={item['n']}"
            )
        lines.extend(["", "## Execution"])
        for horizon, value in sorted(summary["markouts"].items()):
            adverse = summary["adverse_selection"].get(horizon, 0.0)
            lines.append(
                f"- Markout {horizon}m: {value:+.4f}, adverse_selection={adverse:.2%}"
            )
        fill_ratio = summary["fill_ratio"]
        lines.append(
            f"- Fill ratio: {fill_ratio['fill_ratio']:.2%} "
            f"({fill_ratio['filled_quantity']:.2f}/{fill_ratio['requested_quantity']:.2f})"
        )
        lines.extend(["", "## PnL"])
        for item in summary["pnl"]:
            lines.append(
                f"- {item['strategy_name']}: "
                f"pre_resolution={item['pnl_pre_resolution']:+.2f}, "
                f"hold_to_resolution={item['pnl_hold_to_resolution']:+.2f}"
            )
        lines.extend(["", "## Calibration"])
        for bucket in summary["calibration"]:
            lines.append(
                f"- bucket {bucket['bucket']}: "
                f"forecast_mean={bucket['forecast_mean']:.2f}, "
                f"realized_rate={bucket['realized_rate']:.2f}, n={bucket['n']}"
            )
        lines.extend(["", "## Risk-Adjusted"])
        for item in summary["sharpe_like"]:
            lines.append(f"- {item['strategy_name']}: sharpe_like={item['sharpe_like']:.4f}")
        lines.extend(["", "## Edge Decay"])
        for horizon, value in sorted(summary["edge_decay"].items()):
            lines.append(f"- {horizon}m retention: {value:.4f}")
        return "\n".join(lines)
