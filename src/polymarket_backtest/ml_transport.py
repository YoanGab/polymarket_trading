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

        # Try to load GRU model for blending (optional)
        self._gru_model = None
        self._gru_blend_weight = 0.12  # 88% XGBoost + 12% GRU
        gru_path = MODELS_DIR.parent / "data" / "prepared" / "gru_model.pt"
        if gru_path.exists():
            try:
                import torch

                checkpoint = torch.load(gru_path, weights_only=False)
                from polymarket_backtest.features import (  # noqa: F811
                    extract_snapshot_features as _esf,
                )

                # Store GRU model and scaler for inference
                class _SimpleGRU:
                    def __init__(self, state_dict, scaler_mean, scaler_scale, n_features):
                        import torch.nn as nn
                        from torch.nn.utils.rnn import pack_padded_sequence

                        self.gru = nn.GRU(n_features, 32, 1, batch_first=True)
                        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
                        # Load only matching keys
                        self.gru.load_state_dict(
                            {k.replace("gru.", ""): v for k, v in state_dict.items() if k.startswith("gru.")}
                        )
                        self.head.load_state_dict(
                            {k.replace("head.", ""): v for k, v in state_dict.items() if k.startswith("head.")}
                        )
                        self.gru.eval()
                        self.head.eval()
                        self.scaler_mean = scaler_mean
                        self.scaler_scale = scaler_scale

                    def predict_proba(self, features_sequence):
                        import torch

                        scaled = (features_sequence - self.scaler_mean) / self.scaler_scale
                        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
                        length = torch.tensor([len(scaled)], dtype=torch.long)
                        packed = pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
                        _, hidden = self.gru(packed)
                        logit = self.head(hidden[-1]).squeeze()
                        return float(torch.sigmoid(logit).item())

                n_features = len(checkpoint.get("scaler_mean", []))
                if n_features > 0:
                    self._gru_model = _SimpleGRU(
                        checkpoint["state_dict"],
                        checkpoint["scaler_mean"],
                        checkpoint["scaler_scale"],
                        n_features,
                    )
            except Exception:
                self._gru_model = None  # GRU loading failed, use XGBoost only

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

        # Predict (XGBoost)
        xgb_probability = self._predict(feature_vector)[0]

        # Blend with GRU if available
        # GRU needs per-snapshot feature vectors. Build from prev_snapshots (basic features only).
        if self._gru_model is not None and len(prev_rows) >= 3:
            try:
                seq_features = []
                for pr in prev_rows[-64:]:  # limit to last 64 for speed
                    f = {
                        "mid": float(pr["mid"]),
                        "best_bid": float(pr["best_bid"]),
                        "best_ask": float(pr["best_ask"]),
                        "last_trade": float(pr["last_trade"]),
                        "volume_1m": float(pr["volume_1m"]),
                        "volume_24h": float(pr["volume_24h"]),
                        "open_interest": float(pr["open_interest"]),
                    }
                    # Compute basic derived features
                    f["spread"] = f["best_ask"] - f["best_bid"]
                    f["spread_pct"] = f["spread"] / max(f["mid"], 0.001)
                    f["price_vs_half"] = abs(f["mid"] - 0.5)
                    f["price_extreme"] = max(f["mid"], 1 - f["mid"])
                    f["volume_oi_ratio"] = f["volume_24h"] / max(f["open_interest"], 1.0)
                    f["trend"] = f["mid"] - f["last_trade"]
                    f["trend_pct"] = f["trend"] / max(f["mid"], 0.001)
                    # Add all other features as 0 (tags, momentum, etc.)
                    vec = [f.get(n, features.get(n, 0.0)) for n in self._feature_names]
                    seq_features.append(vec)
                # Current snapshot
                seq_features.append([features.get(n, 0.0) for n in self._feature_names])
                seq_arr = np.array(seq_features, dtype=np.float32)
                seq_arr = np.nan_to_num(seq_arr, nan=0.0, posinf=1e6, neginf=-1e6)
                gru_prob = self._gru_model.predict_proba(seq_arr)
                probability = (1 - self._gru_blend_weight) * xgb_probability + self._gru_blend_weight * gru_prob
            except Exception:
                probability = xgb_probability
        else:
            probability = xgb_probability

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
        if "calibrator" in model:
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
