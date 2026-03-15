"""Train ML models for market outcome prediction.

Usage:
    uv run python scripts/train_model.py [--model lightgbm|xgboost|logistic|ridge]

Trains a model on historical market data using walk-forward validation.
Saves the trained model to models/ directory.
Outputs calibration stats and Brier score on the held-out test set.

The autoresearch agent can edit this file to try different:
- Model architectures
- Feature subsets
- Hyperparameters
- Training procedures
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_backtest.features import (
    WalkForwardSplit,
    build_dataset,
    walk_forward_split,
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest_v2.sqlite"
PREPARED_DIR = Path(__file__).resolve().parent.parent / "data" / "prepared"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_TSV = Path(__file__).resolve().parent.parent / "results_ml.tsv"


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lower is better. Perfect = 0, random = 0.25."""
    return float(np.mean((y_true - y_pred) ** 2))


def calibration_stats(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Compute calibration error (ECE) and reliability stats."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_pred = y_pred[mask].mean()
        bin_true = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_pred - bin_true)
    return {
        "ece": round(ece, 6),
        "mean_pred": round(float(y_pred.mean()), 4),
        "mean_true": round(float(y_true.mean()), 4),
        "std_pred": round(float(y_pred.std()), 4),
    }


def train_lightgbm(train_X: np.ndarray, train_y: np.ndarray, val_X: np.ndarray, val_y: np.ndarray) -> object:
    """Train a LightGBM model with early stopping on validation set."""
    import lightgbm as lgb

    dtrain = lgb.Dataset(train_X, label=train_y)
    dval = lgb.Dataset(val_X, label=val_y, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.02,
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 100,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.6,
        "bagging_freq": 5,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


def train_logistic(train_X: np.ndarray, train_y: np.ndarray) -> object:
    """Train a logistic regression baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_scaled, train_y)

    # Return both scaler and model
    return {"scaler": scaler, "model": model}


def train_mlp(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Train an MLP neural network with early stopping."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,  # L2 regularization
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )
    model.fit(X_scaled, train_y)
    return {"scaler": scaler, "model": model}


def train_rf(train_X: np.ndarray, train_y: np.ndarray) -> object:
    """Train a Random Forest — averaging ensemble, often well-calibrated."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=50,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, train_y)
    return {"scaler": scaler, "model": model}


def train_stacking(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Train a stacking ensemble: LR + MLP + RF → LR meta-learner."""
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)

    estimators = [
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=0.01,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=42,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=50,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.5, max_iter=1000),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )
    model.fit(X_scaled, train_y)
    return {"scaler": scaler, "model": model}


def train_elastic(train_X: np.ndarray, train_y: np.ndarray) -> object:
    """Train elastic net logistic regression (L1+L2 mix for feature selection)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)

    model = LogisticRegression(
        penalty="elasticnet",
        C=1.0,
        l1_ratio=0.5,
        solver="saga",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_scaled, train_y)
    return {"scaler": scaler, "model": model}


def train_hybrid_blend(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Hybrid: blend LR + MLP + RF with optimal weights on validation set."""
    lr_model = train_logistic(train_X, train_y)
    mlp_model = train_mlp(train_X, train_y, val_X, val_y)
    rf_model = train_rf(train_X, train_y)

    # Find optimal blend weights on validation set
    lr_preds = predict(lr_model, val_X)
    mlp_preds = predict(mlp_model, val_X)
    rf_preds = predict(rf_model, val_X)

    best_weights = (1.0, 0.0, 0.0)
    best_brier = float("inf")
    # Grid search over 3 weights (must sum to 1)
    for w1 in np.arange(0.0, 1.05, 0.1):
        for w2 in np.arange(0.0, 1.05 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < -0.01:
                continue
            blend = w1 * lr_preds + w2 * mlp_preds + w3 * rf_preds
            bs = brier_score(val_y, blend)
            if bs < best_brier:
                best_weights = (float(w1), float(w2), float(w3))
                best_brier = bs

    print(
        f"  Hybrid weights: LR={best_weights[0]:.1f}, MLP={best_weights[1]:.1f}, RF={best_weights[2]:.1f} (val brier={best_brier:.6f})"
    )
    return {
        "ensemble": [lr_model, mlp_model, rf_model],
        "weights": list(best_weights),
    }


def train_boosted_lr(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Boosted Logistic: LR base + small LightGBM on residuals.

    The LR captures the well-calibrated linear signal.
    The LGB learns nonlinear corrections from LR's mistakes.
    """
    import lightgbm as lgb

    # Stage 1: Train LR
    lr_model = train_logistic(train_X, train_y)
    lr_train_preds = predict(lr_model, train_X)
    lr_val_preds = predict(lr_model, val_X)

    # Stage 2: Train LGB on residuals (actual - lr_prediction)
    train_residuals = train_y - lr_train_preds
    val_residuals = val_y - lr_val_preds

    dtrain = lgb.Dataset(train_X, label=train_residuals)
    dval = lgb.Dataset(val_X, label=val_residuals, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.01,
        "num_leaves": 4,  # Very small tree — just corrections
        "max_depth": 2,
        "min_child_samples": 200,
        "feature_fraction": 0.3,
        "bagging_fraction": 0.5,
        "bagging_freq": 5,
        "reg_alpha": 5.0,
        "reg_lambda": 20.0,
        "verbose": -1,
        "seed": 42,
    }

    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    return {"boosted_lr": True, "lr": lr_model, "lgb_residual": lgb_model}


def train_xgboost(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Train XGBoost with strong regularization."""
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)
    val_scaled = scaler.transform(val_X)

    dtrain = xgb.DMatrix(X_scaled, label=train_y)
    dval = xgb.DMatrix(val_scaled, label=val_y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.01,
        "max_depth": 0,  # unlimited depth for lossguide
        "max_leaves": 512,
        "grow_policy": "lossguide",
        "min_child_weight": 200,
        "subsample": 0.5,
        "colsample_bytree": 0.3,
        "reg_alpha": 8.0,
        "reg_lambda": 25.0,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2500,
        evals=[(dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )

    # Isotonic calibration on held-out portion of training data
    from sklearn.isotonic import IsotonicRegression

    cal_size = min(len(train_y) // 10, 500000)
    rng = np.random.RandomState(42)
    cal_idx = rng.choice(len(train_y), cal_size, replace=False)
    cal_preds = model.predict(xgb.DMatrix(X_scaled[cal_idx]))
    cal_labels = train_y[cal_idx]

    calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    calibrator.fit(cal_preds, cal_labels)
    print(f"  XGBoost: isotonic calibration on {cal_size} train samples")

    return {"xgb_model": model, "scaler": scaler, "calibrator": calibrator}


def train_catboost(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Train CatBoost — good at small datasets, built-in calibration."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=3,
        l2_leaf_reg=10.0,
        min_data_in_leaf=100,
        random_seed=42,
        verbose=0,
        eval_metric="Logloss",
        use_best_model=True,
    )
    model.fit(train_X, train_y, eval_set=(val_X, val_y))
    return {"catboost": model}


def train_pytorch_mlp(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Modern PyTorch MLP with batch norm, dropout, mixup, and cosine annealing.

    Uses techniques proven to work for small tabular data:
    - BatchNorm for internal covariate shift
    - Dropout for regularization
    - Mixup augmentation to expand small dataset
    - Cosine annealing learning rate schedule
    - Label smoothing to improve calibration
    """
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch_models import TabularMLP

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_X)
    X_val = scaler.transform(val_X)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1)

    n_features = X_train.shape[1]

    model = TabularMLP(n_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # Label smoothing via BCE with smoothed targets
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(300):
        model.train()
        # Mixup augmentation
        lam = np.random.beta(0.2, 0.2) if np.random.random() < 0.5 else 1.0
        if lam < 1.0:
            idx = torch.randperm(len(X_train_t))
            X_mix = lam * X_train_t + (1 - lam) * X_train_t[idx]
            y_mix = lam * y_train_t + (1 - lam) * y_train_t[idx]
        else:
            X_mix, y_mix = X_train_t, y_train_t

        # Label smoothing
        y_smooth = y_mix * 0.95 + 0.025

        logits = model(X_mix)
        loss = criterion(logits, y_smooth)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t)
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    model.eval()
    print(f"  PyTorch MLP: stopped at epoch {epoch + 1}, best val_loss={best_val_loss:.6f}")
    return {"pytorch_mlp": model, "scaler": scaler}


def train_pytorch_residual(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Residual network: skip connections help gradient flow in deeper nets."""
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch_models import TabularResNet

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_X)
    X_val = scaler.transform(val_X)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1)

    n_features = X_train.shape[1]

    torch.manual_seed(42)
    np.random.seed(42)
    model = TabularResNet(n_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(300):
        model.train()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t * 0.95 + 0.025)  # label smoothing
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t)
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    model.eval()
    print(f"  ResNet: stopped at epoch {epoch + 1}, best val_loss={best_val_loss:.6f}")
    return {"pytorch_mlp": model, "scaler": scaler}


def train_dl_lr_blend(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Hybrid: Logistic Regression + PyTorch MLP blend with optimal weights."""
    lr_model = train_logistic(train_X, train_y)
    dl_model = train_pytorch_mlp(train_X, train_y, val_X, val_y)

    lr_preds = predict(lr_model, val_X)
    dl_preds = predict(dl_model, val_X)

    best_w, best_brier = 1.0, float("inf")
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * lr_preds + (1 - w) * dl_preds
        bs = brier_score(val_y, blend)
        if bs < best_brier:
            best_w, best_brier = float(w), bs
    print(f"  DL+LR blend: LR={best_w:.2f}, DL={1 - best_w:.2f} (val brier={best_brier:.6f})")
    return {"ensemble": [lr_model, dl_model], "weights": [best_w, 1 - best_w]}


def train_large_resnet(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Larger ResNet with 5 blocks, 256-wide layers."""
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch_models import LargeResNet

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_X)
    X_val = scaler.transform(val_X)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1)

    model = LargeResNet(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 40
    no_improve = 0

    for epoch in range(500):
        model.train()
        # Mixup
        lam = np.random.beta(0.4, 0.4) if np.random.random() < 0.5 else 1.0
        if lam < 1.0:
            idx = torch.randperm(len(X_train_t))
            X_mix = lam * X_train_t + (1 - lam) * X_train_t[idx]
            y_mix = lam * y_train_t + (1 - lam) * y_train_t[idx]
        else:
            X_mix, y_mix = X_train_t, y_train_t

        y_smooth = y_mix * 0.95 + 0.025  # label smoothing
        logits = model(X_mix)
        loss = criterion(logits, y_smooth)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t)
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    model.eval()
    print(f"  LargeResNet: stopped at epoch {epoch + 1}, best val_loss={best_val_loss:.6f}")
    return {"pytorch_mlp": model, "scaler": scaler}


def train_ft_transformer(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Feature Tokenizer + Transformer — attention-based tabular model."""
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch_models import FTTransformer

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_X)
    X_val = scaler.transform(val_X)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1)

    model = FTTransformer(
        n_features=X_train.shape[1],
        d_token=64,
        n_heads=4,
        n_layers=3,
        dropout=0.2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 40
    no_improve = 0

    for epoch in range(400):
        model.train()
        logits = model(X_train_t)
        y_smooth = y_train_t * 0.95 + 0.025
        loss = criterion(logits, y_smooth)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t)
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    model.eval()
    print(f"  FT-Transformer: stopped at epoch {epoch + 1}, best val_loss={best_val_loss:.6f}")
    return {"pytorch_mlp": model, "scaler": scaler}


def train_transformer_lr_blend(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Hybrid: LR + FT-Transformer blend."""
    lr_model = train_logistic(train_X, train_y)
    ft_model = train_ft_transformer(train_X, train_y, val_X, val_y)

    lr_preds = predict(lr_model, val_X)
    ft_preds = predict(ft_model, val_X)

    best_w, best_brier = 1.0, float("inf")
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * lr_preds + (1 - w) * ft_preds
        bs = brier_score(val_y, blend)
        if bs < best_brier:
            best_w, best_brier = float(w), bs
    print(f"  FT+LR blend: LR={best_w:.2f}, FT={1 - best_w:.2f} (val brier={best_brier:.6f})")
    return {"ensemble": [lr_model, ft_model], "weights": [best_w, 1 - best_w]}


def train_deep_ensemble_lr(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Deep Ensemble: average 5 ResNets with different seeds + LR blend.

    Deep ensembles are proven to improve calibration and uncertainty estimation.
    Each member captures different aspects of the data due to random init.
    """
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch_models import TabularResNet

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_X)
    X_val = scaler.transform(val_X)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1)

    n_features = X_train.shape[1]
    n_members = 5
    members = []

    for seed in range(n_members):
        torch.manual_seed(seed * 42 + 7)
        model = TabularResNet(n_features)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        criterion = torch.nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(300):
            model.train()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t * 0.95 + 0.025)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t)
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= 30:
                        break

        model.load_state_dict(best_state)  # type: ignore[arg-type]
        model.eval()
        members.append(model)
        print(f"    Member {seed + 1}/{n_members}: val_loss={best_val_loss:.6f}, epochs={epoch + 1}")

    # Average ensemble predictions
    class DeepEnsemble:
        def __init__(self, models: list) -> None:  # type: ignore[type-arg]
            self.models = models

    ensemble_model = {"pytorch_deep_ensemble": members, "scaler": scaler}

    # Now blend with LR
    lr_model = train_logistic(train_X, train_y)
    lr_preds = predict(lr_model, val_X)

    # Get ensemble predictions on val
    X_val_scaled = torch.tensor(scaler.transform(val_X), dtype=torch.float32)
    with torch.no_grad():
        member_preds = []
        for m in members:
            m.eval()
            logits = m(X_val_scaled)
            probs = torch.sigmoid(logits).squeeze().numpy()
            member_preds.append(probs)
    ensemble_preds = np.clip(np.mean(member_preds, axis=0), 0.001, 0.999)

    best_w, best_brier = 1.0, float("inf")
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * lr_preds + (1 - w) * ensemble_preds
        bs = brier_score(val_y, blend)
        if bs < best_brier:
            best_w, best_brier = float(w), bs
    print(f"  Deep Ensemble+LR: LR={best_w:.2f}, Ensemble={1 - best_w:.2f} (val brier={best_brier:.6f})")
    return {"ensemble": [lr_model, ensemble_model], "weights": [best_w, 1 - best_w]}


def train_large_resnet_lr_blend(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Hybrid: LR + Large ResNet blend."""
    lr_model = train_logistic(train_X, train_y)
    resnet_model = train_large_resnet(train_X, train_y, val_X, val_y)

    lr_preds = predict(lr_model, val_X)
    resnet_preds = predict(resnet_model, val_X)

    best_w, best_brier = 1.0, float("inf")
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * lr_preds + (1 - w) * resnet_preds
        bs = brier_score(val_y, blend)
        if bs < best_brier:
            best_w, best_brier = float(w), bs
    print(f"  LargeResNet+LR blend: LR={best_w:.2f}, ResNet={1 - best_w:.2f} (val brier={best_brier:.6f})")
    return {"ensemble": [lr_model, resnet_model], "weights": [best_w, 1 - best_w]}


def train_resnet_lr_blend(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Hybrid: LR (best Brier) + ResNet (best ECE) blend."""
    lr_model = train_logistic(train_X, train_y)
    resnet_model = train_pytorch_residual(train_X, train_y, val_X, val_y)

    lr_preds = predict(lr_model, val_X)
    resnet_preds = predict(resnet_model, val_X)

    best_w, best_brier = 1.0, float("inf")
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * lr_preds + (1 - w) * resnet_preds
        bs = brier_score(val_y, blend)
        if bs < best_brier:
            best_w, best_brier = float(w), bs
    print(f"  ResNet+LR blend: LR={best_w:.2f}, ResNet={1 - best_w:.2f} (val brier={best_brier:.6f})")
    return {"ensemble": [lr_model, resnet_model], "weights": [best_w, 1 - best_w]}


def train_mega_ensemble(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Mega ensemble: LR + ResNet + LightGBM + CatBoost — diverse model zoo."""
    print("  Training 4-model mega ensemble...")
    lr_model = train_logistic(train_X, train_y)
    resnet_model = train_pytorch_residual(train_X, train_y, val_X, val_y)
    lgb_model = train_lightgbm(train_X, train_y, val_X, val_y)
    cat_model = train_catboost(train_X, train_y, val_X, val_y)

    models = [lr_model, resnet_model, lgb_model, cat_model]
    names = ["LR", "ResNet", "LGB", "CatBoost"]
    val_preds = [predict(m, val_X) for m in models]

    # Grid search over 4 weights (simplex)
    best_weights = [1.0, 0.0, 0.0, 0.0]
    best_brier = float("inf")
    step = 0.1
    for w1 in np.arange(0.0, 1.05, step):
        for w2 in np.arange(0.0, 1.05 - w1, step):
            for w3 in np.arange(0.0, 1.05 - w1 - w2, step):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -0.01:
                    continue
                blend = sum(w * p for w, p in zip([w1, w2, w3, w4], val_preds))
                bs = brier_score(val_y, np.clip(blend, 0.001, 0.999))  # type: ignore[arg-type]
                if bs < best_brier:
                    best_weights = [float(w1), float(w2), float(w3), float(w4)]
                    best_brier = bs

    wt_str = ", ".join(f"{n}={w:.1f}" for n, w in zip(names, best_weights))
    print(f"  Mega weights: {wt_str} (val brier={best_brier:.6f})")
    return {"ensemble": models, "weights": best_weights}


def train_xgb_lr_blend(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> object:
    """Blend XGBoost + LR with optimal weights."""
    lr_model = train_logistic(train_X, train_y)
    xgb_model = train_xgboost(train_X, train_y, val_X, val_y)

    lr_preds = predict(lr_model, val_X)
    xgb_preds = predict(xgb_model, val_X)

    best_w, best_brier = 0.5, float("inf")
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * lr_preds + (1 - w) * xgb_preds
        bs = brier_score(val_y, blend)
        if bs < best_brier:
            best_w, best_brier = float(w), bs
    print(f"  XGB+LR blend: LR={best_w:.2f}, XGB={1 - best_w:.2f} (val brier={best_brier:.6f})")
    return {"ensemble": [lr_model, xgb_model], "weights": [best_w, 1 - best_w]}


def predict(model: object, X: np.ndarray) -> np.ndarray:
    """Get probability predictions from any supported model type."""
    if isinstance(model, dict) and "ensemble" in model:
        # Weighted ensemble
        preds = [predict(m, X) * w for m, w in zip(model["ensemble"], model["weights"])]
        return np.clip(sum(preds), 0.001, 0.999)  # type: ignore[arg-type]
    if isinstance(model, dict) and "boosted_lr" in model:
        # Boosted logistic: LR base + LGB residual correction
        lr_preds = predict(model["lr"], X)
        residual_correction = model["lgb_residual"].predict(X)
        return np.clip(lr_preds + residual_correction, 0.001, 0.999)
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
        # LightGBM Booster
        raw = model.predict(X)  # type: ignore[union-attr]
        return np.clip(raw, 0.001, 0.999)
    if isinstance(model, dict) and "scaler" in model:
        # Logistic/MLP/RF/Elastic with scaler
        X_scaled = model["scaler"].transform(X)
        probs = model["model"].predict_proba(X_scaled)[:, 1]
        return np.clip(probs, 0.001, 0.999)
    raise ValueError(f"Unknown model type: {type(model)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train ML model for market prediction")
    parser.add_argument(
        "--model",
        choices=[
            "lightgbm",
            "logistic",
            "ensemble",
            "mlp",
            "rf",
            "stacking",
            "elastic",
            "hybrid",
            "boosted_lr",
            "xgboost",
            "catboost",
            "xgb_lr",
            "pytorch_mlp",
            "pytorch_resnet",
            "dl_lr",
            "resnet_lr",
            "mega",
            "large_resnet",
            "ft_transformer",
            "transformer_lr",
            "large_resnet_lr",
            "deep_ensemble_lr",
        ],
        default="logistic",
    )
    parser.add_argument("--stride", type=int, default=6, help="Snapshot stride (1=all, 6=every 6h)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset — prefer pre-built .npz files from prepare.py
    meta_path = PREPARED_DIR / "meta.json"
    if meta_path.exists():
        print("Loading pre-built dataset from data/prepared/ ...")
        start = time.monotonic()
        with open(meta_path) as f:
            meta = json.load(f)
        feature_names = meta["feature_names"]

        train_data = np.load(PREPARED_DIR / "train.npz", allow_pickle=True)
        val_data = np.load(PREPARED_DIR / "val.npz", allow_pickle=True)
        test_data = np.load(PREPARED_DIR / "test.npz", allow_pickle=True)

        split = WalkForwardSplit(
            train_X=train_data["X"],
            train_y=train_data["y"],
            val_X=val_data["X"],
            val_y=val_data["y"],
            test_X=test_data["X"],
            test_y=test_data["y"],
            train_market_ids=list(train_data["market_ids"]),
            val_market_ids=list(val_data["market_ids"]),
            test_market_ids=list(test_data["market_ids"]),
        )
        total_samples = split.train_X.shape[0] + split.val_X.shape[0] + split.test_X.shape[0]
        print(
            f"  Dataset: {total_samples} samples, {len(feature_names)} features (loaded in {time.monotonic() - start:.1f}s)"
        )
    else:
        print("No pre-built dataset found. Run: uv run python scripts/prepare.py")
        print("Falling back to SQLite extraction...")
        if not DB_PATH.exists():
            print("ERROR: Database not found. Run download_data.py first.", file=sys.stderr)
            sys.exit(1)
        start = time.monotonic()
        dataset = build_dataset(DB_PATH, snapshot_stride=args.stride)
        feature_names = feature_names
        print(f"  Dataset: {dataset.X.shape[0]} samples, {dataset.X.shape[1]} features")
        print(f"  Markets: {len(set(dataset.market_ids))}")
        split = walk_forward_split(dataset)

    # Drop features unavailable at inference.
    # ml_transport.py now passes prev_snapshots from context_bundle,
    # so momentum/volatility features ARE available at inference.
    # Still exclude: best_bid, best_ask, last_trade (copied to output).
    # Tag features and n_tags ARE available — market.tags is populated
    # during replay and passed through the context_bundle.
    INFERENCE_AVAILABLE = {
        "mid",
        "spread",
        "spread_pct",
        "price_vs_half",
        "price_extreme",
        "volume_1m",
        "volume_24h",
        "open_interest",
        "volume_oi_ratio",
        "trend",
        "trend_pct",
        "extreme_x_spread",
        "extreme_x_volume",
        "hours_to_resolution",
        "log_hours_to_resolution",
        "resolution_proximity",
        # Momentum features — now available via prev_snapshots
        "momentum_3h",
        "momentum_3h_pct",
        "momentum_6h",
        "momentum_6h_pct",
        "momentum_12h",
        "momentum_12h_pct",
        "momentum_24h",
        "momentum_24h_pct",
        # Volatility and range — now available via prev_snapshots
        "volatility_24h",
        "volume_trend",
        "price_range_24h",
        # Longer-horizon momentum — available via prev_snapshots
        "momentum_72h",
        "momentum_72h_pct",
        "momentum_168h",
        "momentum_168h_pct",
        # Longer-horizon volatility/range — available via prev_snapshots
        "volatility_168h",
        "price_range_168h",
        "price_vs_mean_168h",
        # 720h features — available via prev_snapshots
        "price_range_720h",
        "price_vs_mean_720h",
        "distance_to_720h_high",
        "distance_to_720h_low",
        # Tag features — available via market.tags in context_bundle
        "n_tags",
    }
    # Also include all tag_* features (dynamic names from meta.json top_tags)
    INFERENCE_AVAILABLE |= {name for name in feature_names if name.startswith("tag_")}
    keep_idx = [i for i, name in enumerate(feature_names) if name in INFERENCE_AVAILABLE]
    dropped = [name for name in feature_names if name not in INFERENCE_AVAILABLE]
    feature_names = [feature_names[i] for i in keep_idx]
    split = WalkForwardSplit(
        train_X=split.train_X[:, keep_idx],
        train_y=split.train_y,
        val_X=split.val_X[:, keep_idx],
        val_y=split.val_y,
        test_X=split.test_X[:, keep_idx],
        test_y=split.test_y,
        train_market_ids=split.train_market_ids,
        val_market_ids=split.val_market_ids,
        test_market_ids=split.test_market_ids,
    )
    print(f"  Features: {len(feature_names)} kept, {len(dropped)} dropped (unavailable at inference)")

    print(f"  Label balance: {split.train_y.mean():.2%} positive (train)")
    print(f"  Train: {split.train_X.shape[0]} samples ({len(set(split.train_market_ids))} markets)")
    print(f"  Val:   {split.val_X.shape[0]} samples ({len(set(split.val_market_ids))} markets)")
    print(f"  Test:  {split.test_X.shape[0]} samples ({len(set(split.test_market_ids))} markets)")

    # Train
    print(f"\nTraining {args.model}...")
    if args.model == "lightgbm":
        model = train_lightgbm(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "logistic":
        model = train_logistic(split.train_X, split.train_y)
    elif args.model == "mlp":
        model = train_mlp(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "rf":
        model = train_rf(split.train_X, split.train_y)
    elif args.model == "stacking":
        model = train_stacking(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "elastic":
        model = train_elastic(split.train_X, split.train_y)
    elif args.model == "hybrid":
        model = train_hybrid_blend(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "boosted_lr":
        model = train_boosted_lr(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "xgboost":
        model = train_xgboost(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "catboost":
        model = train_catboost(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "xgb_lr":
        model = train_xgb_lr_blend(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "pytorch_mlp":
        model = train_pytorch_mlp(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "pytorch_resnet":
        model = train_pytorch_residual(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "dl_lr":
        model = train_dl_lr_blend(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "resnet_lr":
        model = train_resnet_lr_blend(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "large_resnet":
        model = train_large_resnet(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "ft_transformer":
        model = train_ft_transformer(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "transformer_lr":
        model = train_transformer_lr_blend(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "large_resnet_lr":
        model = train_large_resnet_lr_blend(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "deep_ensemble_lr":
        model = train_deep_ensemble_lr(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "mega":
        model = train_mega_ensemble(split.train_X, split.train_y, split.val_X, split.val_y)
    elif args.model == "ensemble":
        lgb_model = train_lightgbm(split.train_X, split.train_y, split.val_X, split.val_y)
        lr_model = train_logistic(split.train_X, split.train_y)
        # Find optimal weight on validation set
        lgb_preds = predict(lgb_model, split.val_X)
        lr_preds = predict(lr_model, split.val_X)
        best_w, best_brier = 0.5, float("inf")
        for w in np.arange(0.0, 1.05, 0.05):
            blend = w * lgb_preds + (1 - w) * lr_preds
            bs = brier_score(split.val_y, blend)
            if bs < best_brier:
                best_w, best_brier = float(w), bs
        print(f"  Best ensemble weight: LightGBM={best_w:.2f}, Logistic={1 - best_w:.2f} (val brier={best_brier:.6f})")
        model = {"ensemble": [lgb_model, lr_model], "weights": [best_w, 1 - best_w]}
    else:
        print(f"Unknown model: {args.model}", file=sys.stderr)
        sys.exit(1)

    train_time = time.monotonic() - start

    # Evaluate on all splits
    for name, X, y in [
        ("train", split.train_X, split.train_y),
        ("val", split.val_X, split.val_y),
        ("test", split.test_X, split.test_y),
    ]:
        preds = predict(model, X)
        bs = brier_score(y, preds)
        cal = calibration_stats(y, preds)
        baseline_bs = brier_score(y, np.full_like(y, y.mean()))
        improvement = baseline_bs - bs

        print(f"\n  {name.upper()} set:")
        print(f"    Brier score:      {bs:.6f} (baseline: {baseline_bs:.6f}, improvement: {improvement:+.6f})")
        print(f"    ECE:              {cal['ece']:.6f}")
        print(f"    Mean prediction:  {cal['mean_pred']:.4f} (actual: {cal['mean_true']:.4f})")

    # Feature importance (LightGBM)
    if hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")  # type: ignore[union-attr]
        sorted_idx = np.argsort(importance)[::-1]
        print("\n  Top 10 features (by gain):")
        for i in sorted_idx[:10]:
            print(f"    {feature_names[i]:30s}  {importance[i]:.1f}")

    # Save model
    model_path = MODELS_DIR / f"{args.model}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    print(f"\n  Model saved to: {model_path}")

    # Also save as default model for ml_transport
    default_path = MODELS_DIR / "lightgbm_model.pkl"
    if model_path != default_path:
        with open(default_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": feature_names}, f)
        print(f"  Also saved as default: {default_path}")

    # Save metadata
    test_preds = predict(model, split.test_X)
    test_bs = brier_score(split.test_y, test_preds)
    test_baseline = brier_score(split.test_y, np.full_like(split.test_y, split.test_y.mean()))
    test_cal = calibration_stats(split.test_y, test_preds)

    metadata = {
        "model_type": args.model,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_train_samples": int(split.train_X.shape[0]),
        "n_val_samples": int(split.val_X.shape[0]),
        "n_test_samples": int(split.test_X.shape[0]),
        "test_brier_score": test_bs,
        "test_brier_baseline": test_baseline,
        "test_brier_improvement": test_baseline - test_bs,
        "test_ece": test_cal["ece"],
        "train_time_seconds": round(train_time, 1),
    }
    meta_path = MODELS_DIR / f"{args.model}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Append to results TSV
    header_needed = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a") as f:
        if header_needed:
            f.write("timestamp\tmodel\ttest_brier\ttest_improvement\ttest_ece\tn_features\ttrain_time\n")
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t"
            f"{args.model}\t"
            f"{test_bs:.6f}\t"
            f"{test_baseline - test_bs:+.6f}\t"
            f"{test_cal['ece']:.6f}\t"
            f"{len(feature_names)}\t"
            f"{train_time:.1f}\n"
        )

    # Machine-readable summary
    print(
        f"\nRESULT\tMODEL={args.model}"
        f"\tTEST_BRIER={test_bs:.6f}"
        f"\tIMPROVEMENT={test_baseline - test_bs:+.6f}"
        f"\tECE={test_cal['ece']:.6f}"
        f"\tTRAIN_TIME={train_time:.1f}s"
    )


if __name__ == "__main__":
    main()
