"""exp002 baseline: Per-target mean predictor with CV/holdout evaluation.

This script implements a very lightweight baseline model that predicts the
mean target value for each `target_name`. It avoids external dependencies and
follows the hypothesis-driven experiment workflow in AGENTS.md.

Usage:
    python -m project.experiments.exp002.baseline --data-dir data

Outputs:
    - Logs under `project/results/exp002/` (JSON run file)
    - Aggregated index at `project/results/experiments.csv`
    - Submission CSV at `project/results/exp002/submission.csv`
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from project.utils.experiment_logger import ExperimentLogger


TARGET_WEIGHTS: Dict[str, float] = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


@dataclass
class Config:
    """Configuration for the baseline run."""

    data_dir: str = "data"
    experiment_id: str = "exp002"
    hypothesis_id: str = "H-01"
    seed: int = 42
    n_splits: int = 5
    holdout_ratio: float = 0.2


def set_seed(seed: int) -> None:
    """Set RNG seeds for numpy/pandas reproducibility."""
    np.random.seed(seed)


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs.

    Expects:
        - `{data_dir}/train.csv` with columns including `target_name`, `target`.
        - `{data_dir}/test.csv` with columns including `target_name`.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def compute_group_means(df: pd.DataFrame) -> Dict[str, float]:
    """Compute per-`target_name` mean of `target`."""
    return df.groupby("target_name")["target"].mean().to_dict()


def predict_from_means(df: pd.DataFrame, means: Dict[str, float]) -> np.ndarray:
    """Predict using per-target means, defaulting to global mean if unseen."""
    global_mean = float(np.mean(list(means.values()))) if means else 0.0
    return df["target_name"].map(means).fillna(global_mean).to_numpy(dtype=float)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2 (defined as in docs/overview.md)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        # If variance is zero, define R^2 as 0 (no explanatory power beyond mean)
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def weighted_final_r2(df: pd.DataFrame, y_pred: np.ndarray) -> float:
    """Compute weighted Final Score (sum of weights * per-target R^2)."""
    scores = []
    for t, w in TARGET_WEIGHTS.items():
        mask = df["target_name"] == t
        if not np.any(mask):
            continue
        r2 = r2_score(df.loc[mask, "target"].to_numpy(), y_pred[mask])
        scores.append(w * r2)
    return float(np.sum(scores)) if scores else 0.0


def cross_validate(df: pd.DataFrame, n_splits: int, seed: int) -> Dict[str, float]:
    """Run simple KFold CV on rows, averaging metrics across folds."""
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    folds = np.array_split(idx, n_splits)

    rmses, maes, finals = [], [], []
    for k in range(n_splits):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != k])

        df_tr = df.iloc[train_idx]
        df_va = df.iloc[val_idx]

        means = compute_group_means(df_tr)
        y_pred_va = predict_from_means(df_va, means)
        y_true_va = df_va["target"].to_numpy()

        rmses.append(rmse(y_true_va, y_pred_va))
        maes.append(mae(y_true_va, y_pred_va))
        finals.append(weighted_final_r2(df_va, y_pred_va))

    return {
        "cv_rmse": float(np.mean(rmses)),
        "cv_mae": float(np.mean(maes)),
        "cv_final_r2": float(np.mean(finals)),
    }


def holdout_eval(df: pd.DataFrame, ratio: float, seed: int) -> Dict[str, float]:
    """Random holdout evaluation with the same metrics."""
    n = len(df)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    df_tr, df_va = df.iloc[train_idx], df.iloc[val_idx]

    means = compute_group_means(df_tr)
    y_pred_va = predict_from_means(df_va, means)
    y_true_va = df_va["target"].to_numpy()

    return {
        "holdout_rmse": rmse(y_true_va, y_pred_va),
        "holdout_mae": mae(y_true_va, y_pred_va),
        "holdout_final_r2": weighted_final_r2(df_va, y_pred_va),
    }


def save_submission(test_df: pd.DataFrame, means: Dict[str, float], out_path: str) -> None:
    """Generate and save submission CSV using means trained on full train set."""
    preds = test_df["target_name"].map(means).fillna(np.mean(list(means.values())))
    sub = pd.DataFrame({"sample_id": test_df["sample_id"], "target": preds.astype(float)})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)


def main(cfg: Config) -> None:
    set_seed(cfg.seed)

    # Define hypothesis and create logger
    hypothesis = (
        "target_name のワンホット（同値なクラス平均）だけでも、"
        "情報リークを避けつつ安定したベースライン指標を得られる"
    )
    expected = (
        "train/test 共通の特徴に限定することで再現性の高い基準線を確立する。"
        "評価は RMSE/MAE および重み付き Final R^2 を用いる。"
    )
    logger = ExperimentLogger(
        experiment_id=cfg.experiment_id,
        hypothesis_id=cfg.hypothesis_id,
        change_type="model",
        hypothesis=hypothesis,
        expected_effect=expected,
        metrics=["cv_rmse", "cv_mae", "cv_final_r2", "holdout_rmse", "holdout_mae", "holdout_final_r2"],
    )

    # Load data
    train_df, test_df = load_data(cfg.data_dir)
    logger.log_params(
        {
            "data_dir": cfg.data_dir,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "seed": cfg.seed,
            "n_splits": cfg.n_splits,
            "holdout_ratio": cfg.holdout_ratio,
            "target_weights": TARGET_WEIGHTS,
        }
    )

    # CV and holdout
    cv_metrics = cross_validate(train_df, cfg.n_splits, cfg.seed)
    logger.log_metrics(cv_metrics)

    ho_metrics = holdout_eval(train_df, cfg.holdout_ratio, cfg.seed)
    logger.log_metrics(ho_metrics)

    # Train on full train for submission
    means_full = compute_group_means(train_df)
    out_sub = os.path.join("project", "results", cfg.experiment_id, "submission.csv")
    save_submission(test_df, means_full, out_sub)
    logger.add_note(f"Saved submission to {out_sub}")

    # Finalize log
    logger.finalize()

    # Also print a short summary for CLI users
    summary = {**cv_metrics, **ho_metrics}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp002 baseline (per-target mean)")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset root containing train.csv/test.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        seed=args.seed,
        n_splits=args.n_splits,
        holdout_ratio=args.holdout_ratio,
    )
    main(cfg)

