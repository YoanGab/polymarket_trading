"""Fast ML model evaluation — computes trading Sharpe directly from predictions.

Bypasses the slow ReplayEngine by simulating trades from model predictions
on the pre-built dataset. Much faster than eval_strategies.py (~5s vs ~3min).

Usage:
    uv run python scripts/fast_eval.py [--edge-threshold 100] [--kelly 0.15]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PREPARED_DIR = Path(__file__).resolve().parent.parent / "data" / "prepared"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model(model_path: str = "lightgbm_model.pkl"):
    """Load the trained ML model."""
    import pickle

    path = MODELS_DIR / model_path
    if not path.exists():
        print(f"ERROR: Model not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    return data


def predict_model(model: object, X: np.ndarray) -> np.ndarray:
    """Get probability predictions from trained model."""
    if isinstance(model, dict) and "xgb_model" in model:
        import xgboost as xgb

        X_scaled = model["scaler"].transform(X)
        dmat = xgb.DMatrix(X_scaled)
        xgb_model = model["xgb_model"]
        if isinstance(xgb_model, list):
            raw = np.mean([m.predict(dmat) for m in xgb_model], axis=0)
        else:
            raw = xgb_model.predict(dmat)
        if "calibrator" in model:
            raw = model["calibrator"].transform(raw)
        return np.clip(raw, 0.001, 0.999)
    if isinstance(model, dict) and "boosted_lr" in model:
        lr_preds = predict_model(model["lr"], X)
        residual = model["lgb_residual"].predict(X)
        return np.clip(lr_preds + residual, 0.001, 0.999)
    if isinstance(model, dict) and "ensemble" in model:
        preds = [predict_model(m, X) * w for m, w in zip(model["ensemble"], model["weights"])]
        return np.clip(sum(preds), 0.001, 0.999)  # type: ignore[arg-type]
    if isinstance(model, dict) and "catboost" in model:
        raw = model["catboost"].predict_proba(X)[:, 1]
        return np.clip(raw, 0.001, 0.999)
    if isinstance(model, dict) and "scaler" in model:
        X_scaled = model["scaler"].transform(X)
        probs = model["model"].predict_proba(X_scaled)[:, 1]
        return np.clip(probs, 0.001, 0.999)
    if hasattr(model, "predict"):
        raw = model.predict(X)  # type: ignore[union-attr]
        return np.clip(raw, 0.001, 0.999)
    raise ValueError(f"Unknown model type: {type(model)}")


def simulate_trades(
    predictions: np.ndarray,
    labels: np.ndarray,
    market_ids: np.ndarray,
    mid_prices: np.ndarray,
    *,
    edge_threshold_bps: float = 100.0,
    kelly_fraction: float = 0.15,
    max_position: float = 500.0,
    starting_cash: float = 1000.0,
) -> dict[str, float]:
    """Simulate edge-based trading and compute Sharpe.

    For each snapshot where abs(prediction - mid) > edge_threshold:
    - Buy YES if prediction > mid (model thinks probability is higher)
    - Buy NO if prediction < mid (model thinks probability is lower)
    - PnL is computed at resolution (known outcome).
    """
    edge_threshold = edge_threshold_bps / 10000.0
    trades_per_market: dict[str, list[float]] = {}

    for i in range(len(predictions)):
        pred = predictions[i]
        mid = mid_prices[i]
        label = labels[i]  # resolved outcome (0 or 1)
        market_id = str(market_ids[i])

        edge = pred - mid
        if abs(edge) < edge_threshold:
            continue

        # Only take one trade per market (use first signal)
        if market_id in trades_per_market:
            continue

        # Position size = Kelly fraction * edge / odds
        position_size = min(kelly_fraction * abs(edge) * starting_cash, max_position)

        if edge > 0:
            # Buy YES: pay mid, receive label (1 or 0)
            pnl = position_size * (label - mid) / max(mid, 0.01)
        else:
            # Buy NO: pay (1-mid), receive (1-label)
            pnl = position_size * ((1 - label) - (1 - mid)) / max(1 - mid, 0.01)

        trades_per_market[market_id] = [pnl]

    if not trades_per_market:
        return {"sharpe": 0.0, "pnl": 0.0, "trades": 0, "win_rate": 0.0}

    pnls = [sum(t) for t in trades_per_market.values()]
    pnl_arr = np.array(pnls)
    total_pnl = float(np.sum(pnl_arr))
    sharpe = float(np.mean(pnl_arr) / max(np.std(pnl_arr), 1e-8)) if len(pnl_arr) > 1 else 0.0
    win_rate = float(np.mean(pnl_arr > 0))
    n_trades = len(pnls)

    return {
        "sharpe": round(sharpe, 4),
        "pnl": round(total_pnl, 2),
        "trades": n_trades,
        "win_rate": round(win_rate, 4),
        "avg_pnl_per_trade": round(total_pnl / max(n_trades, 1), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Fast ML model evaluation")
    parser.add_argument("--edge-threshold", type=float, default=100.0, help="Edge threshold in bps")
    parser.add_argument("--kelly", type=float, default=0.15, help="Kelly fraction")
    parser.add_argument("--max-position", type=float, default=500.0, help="Max position size")
    parser.add_argument("--model", default="lightgbm_model.pkl", help="Model filename")
    args = parser.parse_args()

    # Load metadata and feature names
    meta_path = PREPARED_DIR / "meta.json"
    if not meta_path.exists():
        print("ERROR: Run scripts/prepare.py first", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    all_feature_names = meta["feature_names"]

    # Load val data
    start = time.monotonic()
    val_data = np.load(PREPARED_DIR / "val.npz", allow_pickle=True)
    X_val = val_data["X"]
    y_val = val_data["y"]
    market_ids = val_data["market_ids"]

    # Get mid price index
    mid_idx = all_feature_names.index("mid")
    mid_prices = X_val[:, mid_idx]

    # Load model — pickle contains {"model": <model_obj>, "feature_names": [...]}
    model_data = load_model(args.model)
    model = model_data["model"]
    model_feature_names = model_data["feature_names"]

    # Filter val features to match model's expected features
    keep_idx = [all_feature_names.index(f) for f in model_feature_names if f in all_feature_names]
    X_val_filtered = X_val[:, keep_idx]

    # Predict
    predictions = predict_model(model, X_val_filtered)
    load_time = time.monotonic() - start

    # Simulate trades with different thresholds
    print(f"Loaded {len(predictions)} val samples in {load_time:.1f}s")
    print(f"Unique val markets: {len(set(market_ids))}")
    print()

    for edge_bps in [50, 100, 150, 200, 300]:
        result = simulate_trades(
            predictions,
            y_val,
            market_ids,
            mid_prices,
            edge_threshold_bps=edge_bps,
            kelly_fraction=args.kelly,
            max_position=args.max_position,
        )
        print(
            f"  edge={edge_bps:4d}bps"
            f"  Sharpe={result['sharpe']:+.4f}"
            f"  PnL={result['pnl']:+10.2f}"
            f"  trades={result['trades']:6d}"
            f"  win_rate={result['win_rate']:.3f}"
            f"  avg_pnl={result['avg_pnl_per_trade']:+.2f}"
        )

    # Report the main result at the requested threshold
    result = simulate_trades(
        predictions,
        y_val,
        market_ids,
        mid_prices,
        edge_threshold_bps=args.edge_threshold,
        kelly_fraction=args.kelly,
        max_position=args.max_position,
    )
    print(
        f"\nRESULT"
        f"\tSHARPE={result['sharpe']:.6f}"
        f"\tPNL={result['pnl']:.2f}"
        f"\tTRADES={result['trades']}"
        f"\tWIN_RATE={result['win_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
