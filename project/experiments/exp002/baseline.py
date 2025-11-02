"""Baseline model for Image2Biomass (exp002).

Hypothesis H-01:
    Using simple tabular features derived from `train.csv` (date parts,
    numeric NDVI/height, and categorical one-hot for state/species/target_name)
    with a linear model (Ridge) will produce a stable, reasonable baseline
    RMSE via 5-fold CV and a comparable holdout score.

Change Type:
    model (baseline introduction)

Evaluation:
    - 5-fold KFold (shuffle=True, random_state=42)
    - Holdout split: 80/20 random split with same random_state
    - Metrics: RMSE (root mean squared error), MAE

Outputs:
    - Logs under `project/results/exp002/` via ExperimentLogger
    - Submission file at `project/results/exp002/submission.csv`

Usage:
    python -m project.experiments.exp002.baseline --data-dir data
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when sklearn missing
    SKLEARN_AVAILABLE = False

from project.utils.experiment_logger import ExperimentLogger


def resolve_data_dir(path: str | None) -> Path:
    if path:
        return Path(path)
    env = os.getenv("CSIRO_BIOMASS_DATA")
    if env:
        return Path(env)
    return Path("data")


def load_train_test(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # Type conversions
    for c in ["Pre_GSHH_NDVI", "Height_Ave_cm", "target"]:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")
        if c != "target" and c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce")

    # Date parts
    for df in (train, test):
        if "Sampling_Date" in df.columns:
            dt = pd.to_datetime(df["Sampling_Date"], errors="coerce", infer_datetime_format=True)
            df["day_of_year"] = dt.dt.dayofyear
            df["year"] = dt.dt.year
            df["month"] = dt.dt.month
        else:
            df["day_of_year"] = np.nan
            df["year"] = np.nan
            df["month"] = np.nan

    return train, test


def build_pipeline(numeric_features: List[str], categorical_features: List[str]):
    """Builds either an sklearn pipeline (if available) or a NumPy fallback.

    Returns an object with `.fit(X, y)` and `.predict(X)`.
    """
    if SKLEARN_AVAILABLE:
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ])
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )
        model = Ridge(alpha=1.0, random_state=42)
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        return pipe

    # NumPy fallback: manual standardization + one-hot + ridge closed form
    class NumpyRidge:
        def __init__(self, alpha: float = 1.0):
            self.alpha = float(alpha)
            self.means_: dict[str, float] = {}
            self.stds_: dict[str, float] = {}
            self.cat_levels_: dict[str, List[str]] = {}
            self.coef_: np.ndarray | None = None
            self.cols_: List[str] = []

        def _standardize(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
            df = df.copy()
            for c in numeric_features:
                if fit:
                    self.means_[c] = float(df[c].mean())
                    self.stds_[c] = float(df[c].std() or 1.0)
                mean = self.means_[c]
                std = self.stds_[c] or 1.0
                df[c] = (df[c] - mean) / std
            return df

        def _one_hot(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
            # Determine levels
            if fit:
                for c in categorical_features:
                    self.cat_levels_[c] = sorted(df[c].astype(str).unique().tolist())
            # Build one-hot columns in fixed order
            oh_cols = []
            for c in categorical_features:
                levels = self.cat_levels_[c]
                vals = df[c].astype(str)
                for lv in levels:
                    col = f"{c}__{lv}"
                    oh_cols.append(col)
                    df[col] = (vals == lv).astype(float)
            return df, oh_cols

        def _design(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
            df = self._standardize(df, fit=fit)
            df, oh_cols = self._one_hot(df, fit=fit)
            used_cols = list(numeric_features) + oh_cols
            if fit:
                self.cols_ = used_cols
            X = df[self.cols_].to_numpy(dtype=float)
            # Add bias term
            ones = np.ones((X.shape[0], 1), dtype=float)
            return np.hstack([ones, X])

        def fit(self, Xdf: pd.DataFrame, y: pd.Series):
            X = self._design(Xdf, fit=True)
            yv = y.to_numpy(dtype=float).reshape(-1, 1)
            n_features = X.shape[1]
            I = np.eye(n_features)
            I[0, 0] = 0.0  # do not regularize bias
            A = X.T @ X + self.alpha * I
            b = X.T @ yv
            self.coef_ = np.linalg.pinv(A) @ b  # (n_features, 1)
            return self

        def predict(self, Xdf: pd.DataFrame) -> np.ndarray:
            X = self._design(Xdf, fit=False)
            if self.coef_ is None:
                raise RuntimeError("Model not fitted")
            return (X @ self.coef_).ravel()

    return NumpyRidge(alpha=1.0)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if SKLEARN_AVAILABLE:
        return math.sqrt(mean_squared_error(y_true, y_pred))
    return math.sqrt(_mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if SKLEARN_AVAILABLE:
        from sklearn.metrics import mean_absolute_error as _sk_mae
        return float(_sk_mae(y_true, y_pred))
    return float(np.mean(np.abs(y_true - y_pred)))


def kfold_cv(pipe, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42) -> Tuple[float, float]:
    if SKLEARN_AVAILABLE:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rmses: List[float] = []
        maes: List[float] = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_val)
            rmses.append(rmse(y_val, pred))
            maes.append(mae(y_val, pred))
        return float(np.mean(rmses)), float(np.mean(maes))

    # NumPy fallback CV
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    rmses: List[float] = []
    maes: List[float] = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        rmses.append(rmse(y_val.to_numpy(), pred))
        maes.append(mae(y_val.to_numpy(), pred))
    return float(np.mean(rmses)), float(np.mean(maes))


def holdout_eval(pipe, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42) -> Tuple[float, float]:
    if SKLEARN_AVAILABLE:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        split = int(len(idx) * (1 - test_size))
        tr_idx, val_idx = idx[:split], idx[split:]
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_val)
    return rmse(y_val.to_numpy() if hasattr(y_val, 'to_numpy') else y_val, pred), mae(y_val.to_numpy() if hasattr(y_val, 'to_numpy') else y_val, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline model (exp002)")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory (contains train.csv/test.csv)")
    parser.add_argument("--results-dir", type=str, default=os.path.join("project", "results", "exp002"))
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment logger
    logger = ExperimentLogger(
        experiment_id="exp002",
        hypothesis_id="H-01",
        change_type="model",
        hypothesis=(
            "Tabular baseline (date parts + NDVI + height + one-hot categoricals)"
            " with Ridge will yield reasonable RMSE."
        ),
        expected_effect="Establish reproducible baseline and metrics (CV≈HO).",
        metrics=["rmse_cv", "mae_cv", "rmse_holdout", "mae_holdout"],
    )
    logger.log_params({
        "model": "Ridge",
        "alpha": 1.0,
        "cv_folds": 5,
        "holdout_ratio": 0.2,
        "random_state": 42,
    })

    # Load data
    train, test = load_train_test(data_dir)

    target_col = "target"
    # Use only features available in BOTH train and test to enable submission
    common_cols = set(train.columns).intersection(set(test.columns))
    features_num = [c for c in ["Pre_GSHH_NDVI", "Height_Ave_cm", "day_of_year", "year", "month"] if c in common_cols]
    features_cat = [c for c in ["State", "Species", "target_name"] if c in common_cols]
    # If intersection removes all numeric features (likely), fall back to target_name only
    if not features_num and not features_cat:
        features_cat = [c for c in ["target_name"] if c in common_cols]

    # Basic imputations for numeric features
    for c in features_num:
        med = float(pd.to_numeric(train[c], errors="coerce").median())
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(med)
        if c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce").fillna(med)

    # For categoricals, fill missing with 'UNK'
    for c in features_cat:
        train[c] = train[c].fillna("UNK").astype(str)
        if c in test.columns:
            test[c] = test[c].fillna("UNK").astype(str)

    X = train[features_num + features_cat].copy()
    y = pd.to_numeric(train[target_col], errors="coerce").fillna(0.0)

    pipe = build_pipeline(features_num, features_cat)

    # Cross-validation
    rmse_cv, mae_cv = kfold_cv(pipe, X, y, n_splits=5, seed=42)
    logger.log_metrics({"rmse_cv": rmse_cv, "mae_cv": mae_cv})

    # Holdout evaluation
    rmse_ho, mae_ho = holdout_eval(pipe, X, y, test_size=0.2, seed=42)
    logger.log_metrics({"rmse_holdout": rmse_ho, "mae_holdout": mae_ho})

    # Train on full data, predict test for submission
    pipe.fit(X, y)
    test_feats = test[features_num + features_cat].copy()
    test_pred = pipe.predict(test_feats)

    sub = pd.read_csv(data_dir / "sample_submission.csv")
    # sample_submission likely expects a column matching target_name rows; here we assume single column 'prediction'
    # If column differs, keep the schema from sample file.
    sub_cols = sub.columns.tolist()
    if len(sub_cols) >= 2 and sub_cols[0].lower().startswith("sample"):
        # Kaggle-like format: id + target
        target_out_col = sub_cols[1]
    elif "prediction" in sub_cols:
        target_out_col = "prediction"
    else:
        # Fallback to the last column
        target_out_col = sub_cols[-1]

    # Align by order: test.csv order should match sample_submission order by sample_id
    if "sample_id" in sub.columns and "sample_id" in test.columns:
        test_ordered = test.set_index("sample_id").reindex(sub["sample_id"]).reset_index()
        test_pred = pipe.predict(test_ordered[features_num + features_cat])

    sub[target_out_col] = test_pred.astype(float)
    sub_path = results_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)

    # Persist params/metrics and a quick summary
    logger.add_note(json.dumps({
        "features_num": features_num,
        "features_cat": features_cat,
        "submission": str(sub_path),
    }))
    logger.finalize()

    print(json.dumps({
        "rmse_cv": rmse_cv,
        "mae_cv": mae_cv,
        "rmse_holdout": rmse_ho,
        "mae_holdout": mae_ho,
        "submission": str(sub_path),
    }, indent=2))


if __name__ == "__main__":
    main()
