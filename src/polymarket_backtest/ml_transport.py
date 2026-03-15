"""ML-based ForecastTransport that uses a trained model for probability estimation.

Loads a trained model from models/ and uses it as a forecaster in the replay engine.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import extract_snapshot_features

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


@dataclass
class MLModelTransport:
    """Forecast transport that uses a trained ML model.

    Loads a pickled model and feature_names from models/ directory.
    Extracts features from the context_bundle and predicts probability.
    """

    model_path: str = "lightgbm_model.pkl"
    agent_name: str = "ml_model"
    model_id: str = "lightgbm"
    is_live_safe: bool = False

    def __post_init__(self) -> None:
        path = MODELS_DIR / self.model_path
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run train_model.py first.")
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._model = data["model"]
        self._feature_names: list[str] = data["feature_names"]

    def complete(
        self,
        *,
        model_release: str,
        system_prompt: str,
        context_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        market = context_bundle["market"]
        as_of = context_bundle.get("as_of")
        mid = float(market["mid"])
        best_ask = float(market["best_ask"])
        best_bid = float(market["best_bid"])

        # Build a fake sqlite3.Row-like dict for feature extraction
        row = _market_to_row(market, as_of=as_of)
        prev_snapshots = context_bundle.get("prev_snapshots", [])
        prev_rows = [_market_to_row(snapshot) for snapshot in prev_snapshots]
        features = extract_snapshot_features(row, prev_rows)

        # Add tag features from context_bundle
        market_tags = set(market.get("tags", []))
        for name in self._feature_names:
            if name.startswith("tag_"):
                tag = name[4:]  # remove "tag_" prefix
                features[name] = 1.0 if tag in market_tags else 0.0
        if "n_tags" in self._feature_names:
            features["n_tags"] = float(len(market_tags))

        # Create feature vector in correct order
        feature_vector = np.array(
            [[features.get(name, 0.0) for name in self._feature_names]],
            dtype=np.float32,
        )
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

        # Predict
        probability = self._predict(feature_vector)[0]
        probability = float(np.clip(probability, 0.005, 0.995))

        edge_bps = round((probability - best_ask) * 10_000.0, 2)
        if probability < best_bid:
            thesis = "ML model: sell signal (predicted below bid)"
        elif edge_bps > 0:
            thesis = "ML model: buy signal (predicted above ask)"
        else:
            thesis = "ML model: no edge"

        confidence = _edge_based_confidence(probability, best_bid, best_ask)

        return {
            "agent_name": self.agent_name,
            "model_id": self.model_id,
            "model_release": model_release,
            "probability_yes": round(probability, 4),
            "confidence": round(confidence, 4),
            "expected_edge_bps": edge_bps,
            "thesis": thesis,
            "reasoning": f"ML prediction: {probability:.4f} (mid={mid:.4f})",
            "evidence": [],
        }

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return _predict_model(self._model, X)


def _predict_model(model: object, X: np.ndarray) -> np.ndarray:
    """Recursively predict from any supported model type."""
    if isinstance(model, dict) and "ensemble" in model:
        preds = [_predict_model(m, X) * w for m, w in zip(model["ensemble"], model["weights"])]
        return np.clip(sum(preds), 0.001, 0.999)  # type: ignore[arg-type]
    if isinstance(model, dict) and "boosted_lr" in model:
        lr_preds = _predict_model(model["lr"], X)
        residual = model["lgb_residual"].predict(X)
        return np.clip(lr_preds + residual, 0.001, 0.999)
    if isinstance(model, dict) and "pytorch_deep_ensemble" in model:
        import torch

        scaler = model["scaler"]
        X_scaled = scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        member_preds = []
        for m in model["pytorch_deep_ensemble"]:
            m.eval()
            with torch.no_grad():
                logits = m(X_t)
                probs = torch.sigmoid(logits).squeeze().numpy()
                if probs.ndim == 0:
                    probs = np.array([float(probs)])
                member_preds.append(probs)
        return np.clip(np.mean(member_preds, axis=0), 0.001, 0.999)
    if isinstance(model, dict) and "pytorch_mlp" in model:
        import torch

        scaler = model["scaler"]
        X_scaled = scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        net = model["pytorch_mlp"]
        net.eval()
        with torch.no_grad():
            logits = net(X_t)
            probs = torch.sigmoid(logits).squeeze().numpy()
        # Handle scalar case (single sample)
        if probs.ndim == 0:
            probs = np.array([float(probs)])
        return np.clip(probs, 0.001, 0.999)
    if isinstance(model, dict) and "xgb_model" in model:
        import xgboost as xgb

        X_scaled = model["scaler"].transform(X)
        dmat = xgb.DMatrix(X_scaled)
        raw = model["xgb_model"].predict(dmat)
        if "calibrators" in model:
            raw = np.mean([c.transform(raw) for c in model["calibrators"]], axis=0)
        elif "calibrator" in model:
            raw = model["calibrator"].transform(raw)
        if "platt" in model:
            raw = model["platt"].predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.clip(raw, 0.001, 0.999)
    if isinstance(model, dict) and "catboost" in model:
        raw = model["catboost"].predict_proba(X)[:, 1]
        return np.clip(raw, 0.001, 0.999)
    if hasattr(model, "predict"):
        raw = model.predict(X)  # type: ignore[union-attr]
        return np.clip(raw, 0.001, 0.999)
    if isinstance(model, dict) and "scaler" in model:
        X_scaled = model["scaler"].transform(X)
        probs = model["model"].predict_proba(X_scaled)[:, 1]
        return np.clip(probs, 0.001, 0.999)
    raise ValueError(f"Unknown model type: {type(model)}")


class _DictRow:
    """Minimal sqlite3.Row-like wrapper for dicts."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def keys(self) -> list[str]:
        return list(self._data.keys())


def _edge_based_confidence(probability: float, best_bid: float, best_ask: float) -> float:
    """Use tradable edge as a simple confidence proxy."""
    tradable_edge_bps = max(
        (probability - best_ask) * 10_000.0,
        (best_bid - probability) * 10_000.0,
        0.0,
    )
    return min(0.95, 0.5 + tradable_edge_bps / 1_000.0)


def _market_to_row(market: dict[str, Any], *, as_of: Any = None) -> Any:
    """Convert a context_bundle market dict to a row-like object for feature extraction."""
    return _DictRow(
        {
            "mid": market.get("mid", 0.5),
            "best_bid": market.get("best_bid", 0.49),
            "best_ask": market.get("best_ask", 0.51),
            "last_trade": market.get("last_trade", 0.5),
            "volume_1m": market.get("volume_1m", 0.0),
            "volume_24h": market.get("volume_24h", 0.0),
            "open_interest": market.get("open_interest", 0.0),
            "resolution_ts": market.get("resolution_ts"),
            "ts": market.get("ts", as_of or market.get("as_of", "")),
        }
    )
