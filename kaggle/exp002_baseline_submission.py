"""Kaggle submission script: exp002 Baseline (H-01).

- Dataset path (read-only):
    PATH_DATA = '/kaggle/input/csiro-biomass'
    PATH_TRAIN_CSV = os.path.join(PATH_DATA, 'train.csv')
    PATH_TRAIN_IMG = os.path.join(PATH_DATA, 'train')
    PATH_TEST_IMG  = os.path.join(PATH_DATA, 'test')

This script builds a minimal baseline using only features present in BOTH
train and test CSVs, which in this competition typically includes `target_name`.
We avoid information leakage by not using columns missing in test.

Model: Ridge (scikit-learn). If scikit-learn is unavailable, falls back to a
NumPy ridge implementation. The script prints simple CV metrics (RMSE/MAE)
for reference and writes a submission to `/kaggle/working/submission.csv`.
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# Paths (Kaggle)
PATH_DATA = "/kaggle/input/csiro-biomass"
PATH_TRAIN_CSV = os.path.join(PATH_DATA, "train.csv")
PATH_TEST_CSV = os.path.join(PATH_DATA, "test.csv")
PATH_SAMPLE_SUB = os.path.join(PATH_DATA, "sample_submission.csv")
PATH_TRAIN_IMG = os.path.join(PATH_DATA, "train")
PATH_TEST_IMG = os.path.join(PATH_DATA, "test")
OUTPUT_SUB = "/kaggle/working/submission.csv"


# Try scikit-learn
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def load_train_test() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PATH_TRAIN_CSV)
    test = pd.read_csv(PATH_TEST_CSV)
    sample = pd.read_csv(PATH_SAMPLE_SUB)

    # Numeric conversions if present
    for c in ["Pre_GSHH_NDVI", "Height_Ave_cm", "target"]:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")
        if c != "target" and c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce")

    # Date parts where available
    for df in (train, test):
        if "Sampling_Date" in df.columns:
            dt = pd.to_datetime(df["Sampling_Date"], errors="coerce")
            df["day_of_year"] = dt.dt.dayofyear
            df["year"] = dt.dt.year
            df["month"] = dt.dt.month

    return train, test, sample


def build_pipeline(numeric_features: List[str], categorical_features: List[str]):
    if SKLEARN_AVAILABLE:
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ])
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])
        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )
        model = Ridge(alpha=1.0, random_state=42)
        return Pipeline(steps=[("pre", pre), ("model", model)])

    # NumPy fallback (closed-form ridge)
    class NumpyRidge:
        def __init__(self, alpha: float = 1.0):
            self.alpha = float(alpha)
            self.means_: dict[str, float] = {}
            self.stds_: dict[str, float] = {}
            self.cat_levels_: dict[str, List[str]] = {}
            self.cols_: List[str] = []
            self.coef_: np.ndarray | None = None

        def _standardize(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
            df = df.copy()
            for c in numeric_features:
                if fit:
                    m = float(df[c].mean())
                    s = float(df[c].std() or 1.0)
                    self.means_[c] = m
                    self.stds_[c] = s
                df[c] = (df[c] - self.means_[c]) / (self.stds_[c] or 1.0)
            return df

        def _one_hot(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
            if fit:
                for c in categorical_features:
                    self.cat_levels_[c] = sorted(df[c].astype(str).unique().tolist())
            oh_cols: List[str] = []
            for c in categorical_features:
                levels = self.cat_levels_[c]
                v = df[c].astype(str)
                for lv in levels:
                    col = f"{c}__{lv}"
                    df[col] = (v == lv).astype(float)
                    oh_cols.append(col)
            return df, oh_cols

        def _design(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
            df = self._standardize(df, fit=fit)
            df, oh_cols = self._one_hot(df, fit=fit)
            used = list(numeric_features) + oh_cols
            if fit:
                self.cols_ = used
            X = df[self.cols_].to_numpy(dtype=float)
            return np.hstack([np.ones((len(df), 1)), X])

        def fit(self, Xdf: pd.DataFrame, y: pd.Series):
            X = self._design(Xdf, fit=True)
            yv = y.to_numpy(dtype=float).reshape(-1, 1)
            I = np.eye(X.shape[1]); I[0, 0] = 0.0
            A = X.T @ X + self.alpha * I
            b = X.T @ yv
            self.coef_ = np.linalg.pinv(A) @ b
            return self

        def predict(self, Xdf: pd.DataFrame) -> np.ndarray:
            X = self._design(Xdf, fit=False)
            return (X @ self.coef_).ravel()

    return NumpyRidge(alpha=1.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if SKLEARN_AVAILABLE:
        return math.sqrt(mean_squared_error(y_true, y_pred))  # type: ignore[name-defined]
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if SKLEARN_AVAILABLE:
        return float(mean_absolute_error(y_true, y_pred))  # type: ignore[name-defined]
    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> None:
    train, test, sample = load_train_test()

    # Use intersection of columns to avoid leakage; test typically has only `sample_id`, `image_path`, `target_name`.
    common = set(train.columns).intersection(set(test.columns))
    features_num = [c for c in ["Pre_GSHH_NDVI", "Height_Ave_cm", "day_of_year", "year", "month"] if c in common]
    features_cat = [c for c in ["State", "Species", "target_name"] if c in common]
    if not features_num and not features_cat:
        features_cat = [c for c in ["target_name"] if c in common]

    # Impute/format
    for c in features_num:
        med = float(pd.to_numeric(train[c], errors="coerce").median())
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(med)
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(med) if c in test.columns else train[c]
    for c in features_cat:
        train[c] = train[c].fillna("UNK").astype(str)
        if c in test.columns:
            test[c] = test[c].fillna("UNK").astype(str)

    X = train[features_num + features_cat].copy()
    y = pd.to_numeric(train["target"], errors="coerce").fillna(0.0)

    pipe = build_pipeline(features_num, features_cat)

    # Simple CV for sanity
    if SKLEARN_AVAILABLE and len(train) >= 5:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses, maes = [], []
        for tr, va in kf.split(X):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[va])
            rmses.append(rmse(y.iloc[va].to_numpy(), pred))
            maes.append(mae(y.iloc[va].to_numpy(), pred))
        print(json.dumps({"rmse_cv": float(np.mean(rmses)), "mae_cv": float(np.mean(maes))}, indent=2))

    # Train full and predict test
    pipe.fit(X, y)
    X_test = test[features_num + features_cat].copy()

    # Align to sample_submission order by sample_id
    if "sample_id" in test.columns and "sample_id" in sample.columns:
        test_idxed = test.set_index("sample_id").reindex(sample["sample_id"]).reset_index()
        X_test = test_idxed[features_num + features_cat]

    pred = pipe.predict(X_test)
    # Ensure correct target column name
    sub = sample.copy()
    target_col = "target" if "target" in sub.columns else sub.columns[-1]
    sub[target_col] = pred.astype(float)
    sub.to_csv(OUTPUT_SUB, index=False)
    print(f"Saved submission to {OUTPUT_SUB}")


if __name__ == "__main__":
    main()

