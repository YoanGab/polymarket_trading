"""Production safety guards for live trading."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class TradingGuards:
    """Checks that must pass before any trade is executed in production."""

    def __init__(
        self,
        max_stale_minutes: int = 30,
        max_drawdown_pct: float = 0.20,
        min_confidence: float = 0.60,
        feature_shift_threshold: float = 3.0,
    ) -> None:
        self.max_stale_minutes = max_stale_minutes
        self.max_drawdown_pct = max_drawdown_pct
        self.min_confidence = min_confidence
        self.feature_shift_threshold = feature_shift_threshold
        self._peak_equity = 0.0
        self._training_feature_stats: dict[str, tuple[float, float]] = {}

    def check_staleness(self, last_snapshot_ts: datetime, now: datetime) -> tuple[bool, str]:
        """Reject if data is too old."""
        age = (now - last_snapshot_ts).total_seconds() / 60.0
        if age > self.max_stale_minutes:
            return False, f"Data is {age:.0f}min old (max {self.max_stale_minutes})"
        return True, "OK"

    def check_drawdown(self, current_equity: float, starting_cash: float) -> tuple[bool, str]:
        """Circuit breaker: stop trading if drawdown exceeds threshold."""
        self._peak_equity = max(self._peak_equity, starting_cash, current_equity)
        if self._peak_equity <= 0:
            return True, "OK"

        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        if drawdown > self.max_drawdown_pct:
            return False, f"Drawdown {drawdown:.1%} exceeds {self.max_drawdown_pct:.1%}"
        return True, "OK"

    def check_feature_distribution(self, features: dict[str, float]) -> tuple[bool, str]:
        """Detect distribution shift from training data."""
        if not self._training_feature_stats:
            return True, "No training stats loaded"

        violations: list[str] = []
        for name, raw_value in features.items():
            stats = self._training_feature_stats.get(name)
            if stats is None:
                continue

            mean, std = stats
            if std <= 0:
                continue

            value = float(raw_value)
            if not np.isfinite(value):
                violations.append(f"{name}: non-finite")
                continue

            z_score = abs(value - mean) / std
            if z_score > self.feature_shift_threshold:
                violations.append(f"{name}: z={z_score:.1f}")

        if violations:
            return False, f"Feature shift: {', '.join(violations[:3])}"
        return True, "OK"

    def check_all(
        self,
        last_snapshot_ts: datetime,
        now: datetime,
        current_equity: float,
        starting_cash: float,
        features: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Run all guards. Returns (should_trade, list_of_warnings)."""
        warnings: list[str] = []
        checks = [
            self.check_staleness(last_snapshot_ts, now),
            self.check_drawdown(current_equity, starting_cash),
            self.check_feature_distribution(features),
        ]

        all_ok = True
        for ok, msg in checks:
            if not ok:
                all_ok = False
                warnings.append(msg)
        return all_ok, warnings

    def load_training_stats(self, model_path: str | Path) -> dict[str, tuple[float, float]]:
        """Load feature mean/std from a model pickle or sidecar stats file."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with path.open("rb") as f:
            payload = pickle.load(f)  # noqa: S301

        stats = _extract_feature_stats(payload)
        if stats is None:
            for sidecar_path in _candidate_stats_paths(path):
                if not sidecar_path.exists():
                    continue
                with sidecar_path.open() as f:
                    stats = _extract_feature_stats(json.load(f))
                if stats is not None:
                    break

        self._training_feature_stats = stats or {}
        return dict(self._training_feature_stats)


def _candidate_stats_paths(model_path: Path) -> list[Path]:
    candidates = [
        model_path.with_suffix(".stats.json"),
        model_path.with_name(f"{model_path.stem}_stats.json"),
        model_path.with_name(f"{model_path.stem}_feature_stats.json"),
        model_path.with_name(f"{model_path.stem}_metadata.json"),
    ]

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_paths.append(candidate)
    return unique_paths


def _extract_feature_stats(data: Any) -> dict[str, tuple[float, float]] | None:
    if not isinstance(data, dict):
        return None

    for key in ("training_feature_stats", "feature_stats", "training_stats"):
        stats = _normalize_feature_stats(data.get(key))
        if stats is not None:
            return stats

    metadata = data.get("metadata")
    if metadata is not None:
        stats = _extract_feature_stats(metadata)
        if stats is not None:
            return stats

    return _normalize_feature_stats(data)


def _normalize_feature_stats(raw_stats: Any) -> dict[str, tuple[float, float]] | None:
    if not isinstance(raw_stats, dict) or not raw_stats:
        return None

    normalized: dict[str, tuple[float, float]] = {}
    for name, raw_value in raw_stats.items():
        stats = _coerce_mean_std(raw_value)
        if stats is None:
            return None
        normalized[str(name)] = stats
    return normalized


def _coerce_mean_std(raw_value: Any) -> tuple[float, float] | None:
    if isinstance(raw_value, dict):
        mean = raw_value.get("mean")
        std = raw_value.get("std")
    elif isinstance(raw_value, list | tuple | np.ndarray) and len(raw_value) == 2:
        mean, std = raw_value
    else:
        return None

    try:
        mean_value = float(mean)
        std_value = float(std)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(mean_value) or not np.isfinite(std_value):
        return None
    return mean_value, max(std_value, 0.0)
