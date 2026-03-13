"""Per-category performance analysis.

Breaks down model Brier score and strategy PnL by market category
using real tags from data/market_tags.json.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_backtest.features import build_dataset, walk_forward_split

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_backtest.sqlite"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TAGS_PATH = Path(__file__).resolve().parent.parent / "data" / "market_tags.json"


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def load_model():
    model_path = MODELS_DIR / "logistic_model.pkl"
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_names"]


def predict(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, dict) and "scaler" in model:
        X_scaled = model["scaler"].transform(X)
        probs = model["model"].predict_proba(X_scaled)[:, 1]
        return np.clip(probs, 0.001, 0.999)
    raise ValueError(f"Unknown model type: {type(model)}")


def get_category(market_id: str, market_tags: dict) -> str:
    tags = market_tags.get(market_id, [])
    return tags[0] if tags else "Unknown"


def main() -> None:
    # Load tags
    with open(TAGS_PATH) as f:
        market_tags = json.load(f)

    # Build dataset and split
    print("Building dataset...")
    dataset = build_dataset(DB_PATH, snapshot_stride=6)
    split = walk_forward_split(dataset)

    # Load model
    model, feature_names = load_model()

    # Analyze each split
    for split_name, X, y, market_ids in [
        ("TRAIN", split.train_X, split.train_y, split.train_market_ids),
        ("VAL", split.val_X, split.val_y, split.val_market_ids),
        ("TEST", split.test_X, split.test_y, split.test_market_ids),
    ]:
        preds = predict(model, X)

        # Group by category
        cat_data: dict[str, dict] = defaultdict(lambda: {"y": [], "pred": [], "markets": set()})
        for i, mid in enumerate(market_ids):
            cat = get_category(mid, market_tags)
            cat_data[cat]["y"].append(y[i])
            cat_data[cat]["pred"].append(preds[i])
            cat_data[cat]["markets"].add(mid)

        print(f"\n{'=' * 70}")
        print(f"  {split_name} SET — Per-Category Brier Score")
        print(f"{'=' * 70}")
        print(f"{'Category':<15} {'Markets':>8} {'Samples':>8} {'Brier':>8} {'Baseline':>8} {'Improve':>9} {'%Pos':>6}")
        print(f"{'-' * 15} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 6}")

        total_brier = brier_score(y, preds)
        total_baseline = brier_score(y, np.full_like(y, y.mean()))

        rows = []
        for cat, data in sorted(cat_data.items(), key=lambda x: len(x[1]["markets"]), reverse=True):
            y_cat = np.array(data["y"])
            p_cat = np.array(data["pred"])
            bs = brier_score(y_cat, p_cat)
            baseline = brier_score(y_cat, np.full_like(y_cat, y_cat.mean()))
            improvement = baseline - bs
            rows.append((cat, len(data["markets"]), len(y_cat), bs, baseline, improvement, y_cat.mean()))

        for cat, n_markets, n_samples, bs, baseline, improvement, pct_pos in rows:
            sign = "+" if improvement > 0 else ""
            print(
                f"{cat:<15} {n_markets:>8} {n_samples:>8} {bs:>8.4f} {baseline:>8.4f} {sign}{improvement:>8.4f} {pct_pos:>5.1%}"
            )

        print(
            f"{'TOTAL':<15} {len(set(market_ids)):>8} {len(y):>8} {total_brier:>8.4f} {total_baseline:>8.4f} {'+' if total_baseline - total_brier > 0 else ''}{total_baseline - total_brier:>8.4f} {y.mean():>5.1%}"
        )

        # Highlight categories where model HURTS (negative improvement)
        hurting = [(cat, n_m, bs, bl, imp) for cat, n_m, n_s, bs, bl, imp, _ in rows if imp < 0]
        if hurting:
            print(f"\n  WARNING: Model HURTS in these categories:")
            for cat, n_m, bs, bl, imp in hurting:
                print(f"    {cat}: Brier {bs:.4f} vs baseline {bl:.4f} (worse by {abs(imp):.4f}, {n_m} markets)")

    # Category distribution across splits
    print(f"\n{'=' * 70}")
    print(f"  Category Distribution Across Splits")
    print(f"{'=' * 70}")
    all_cats = set()
    split_cats: dict[str, dict[str, int]] = {}
    for split_name, market_ids in [
        ("TRAIN", split.train_market_ids),
        ("VAL", split.val_market_ids),
        ("TEST", split.test_market_ids),
    ]:
        cat_count: dict[str, int] = defaultdict(int)
        seen = set()
        for mid in market_ids:
            if mid not in seen:
                seen.add(mid)
                cat = get_category(mid, market_tags)
                cat_count[cat] += 1
                all_cats.add(cat)
        split_cats[split_name] = dict(cat_count)

    print(f"{'Category':<15} {'Train':>8} {'Val':>8} {'Test':>8}")
    for cat in sorted(all_cats):
        t = split_cats.get("TRAIN", {}).get(cat, 0)
        v = split_cats.get("VAL", {}).get(cat, 0)
        te = split_cats.get("TEST", {}).get(cat, 0)
        print(f"{cat:<15} {t:>8} {v:>8} {te:>8}")


if __name__ == "__main__":
    main()
