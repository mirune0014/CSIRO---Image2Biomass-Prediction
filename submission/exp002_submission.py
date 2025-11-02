"""Kaggle Submission Script: exp002 (Baseline)

This script trains a simple Ridge regression using features available in BOTH
train and test (typically `target_name` only) and writes a submission file to
`/kaggle/working/submission.csv`.

Paths are set for Kaggle environment as requested:
    PATH_DATA = '/kaggle/input/csiro-biomass'
    PATH_TRAIN_CSV = os.path.join(PATH_DATA, 'train.csv')
    PATH_TRAIN_IMG = os.path.join(PATH_DATA, 'train')
    PATH_TEST_IMG  = os.path.join(PATH_DATA, 'test')

Notes:
- We also load `test.csv` and `sample_submission.csv` from the same dataset.
- If scikit-learn is missing in the Kaggle runtime, the script falls back to a
  NumPy-based ridge implementation so it still runs end-to-end.
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# Kaggle paths
PATH_DATA = "/kaggle/input/csiro-biomass"
PATH_TRAIN_CSV = os.path.join(PATH_DATA, "train.csv")
PATH_TEST_CSV = os.path.join(PATH_DATA, "test.csv")
PATH_SAMPLE_SUB = os.path.join(PATH_DATA, "sample_submission.csv")
PATH_TRAIN_IMG = os.path.join(PATH_DATA, "train")
PATH_TEST_IMG = os.path.join(PATH_DATA, "test")
OUTPUT_SUB = "/kaggle/working/submission.csv"


# Try to use scikit-learn; otherwise fallback to NumPy ridge
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PATH_TRAIN_CSV)
    test = pd.read_csv(PATH_TEST_CSV)
    sample = pd.read_csv(PATH_SAMPLE_SUB)

    # Optional numeric casts if present
    for c in ["Pre_GSHH_NDVI", "Height_Ave_cm", "target"]:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")
        if c != "target" and c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce")

    # Optional date parts
    for df in (train, test):
        if "Sampling_Date" in df.columns:
            dt = pd.to_datetime(df["Sampling_Date"], errors="coerce")
            df["day_of_year"] = dt.dt.dayofyear
            df["year"] = dt.dt.year
            df["month"] = dt.dt.month

    return train, test, sample


def build_pipeline(numeric_features: List[str], categorical_features: List[str]):
    if SKLEARN_AVAILABLE:
        num = Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True))])
        cat = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))])
        pre = ColumnTransformer(
            transformers=[
                ("num", num, numeric_features),
                ("cat", cat, categorical_features),
            ],
            remainder="drop",
        )
        model = Ridge(alpha=1.0, random_state=42)
        return Pipeline(steps=[("pre", pre), ("model", model)])

    # NumPy fallback ridge
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
            for c in categorical_features:
                levels = self.cat_levels_[c]
                v = df[c].astype(str)
                for lv in levels:
                    df[f"{c}__{lv}"] = (v == lv).astype(float)
            return df

        def _design(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
            df = self._standardize(df, fit=fit)
            df = self._one_hot(df, fit=fit)
            if fit:
                # fixed column order
                self.cols_ = list(numeric_features) + [
                    col for col in df.columns if any(col.startswith(f"{c}__") for c in categorical_features)
                ]
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
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))  # type: ignore[name-defined]
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _validate_submission(df_sub: pd.DataFrame) -> None:
    # Basic sanity checks before writing
    if df_sub.isna().any().any():
        na_cols = df_sub.columns[df_sub.isna().any()].tolist()
        raise ValueError(f"Submission contains NaN in columns: {na_cols}")
    # Check finite numeric in target
    tgt_col = "target" if "target" in df_sub.columns else df_sub.columns[-1]
    vals = pd.to_numeric(df_sub[tgt_col], errors="coerce")
    if not np.isfinite(vals).all():
        raise ValueError("Submission contains non-finite values in target column")
    # Row count must match sample
    # (enforced by construction but keep an explicit check for Kaggle safety)


def main() -> None:
    train, test, sample = load_data()

    # Use only columns available in both train and test
    common = set(train.columns).intersection(set(test.columns))
    features_num = [c for c in ["Pre_GSHH_NDVI", "Height_Ave_cm", "day_of_year", "year", "month"] if c in common]
    features_cat = [c for c in ["State", "Species", "target_name"] if c in common]
    if not features_num and not features_cat:
        features_cat = [c for c in ["target_name"] if c in common]

    # Impute numeric, format categoricals
    for c in features_num:
        med = float(pd.to_numeric(train[c], errors="coerce").median())
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(med)
        if c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce").fillna(med)
    for c in features_cat:
        train[c] = train[c].fillna("UNK").astype(str)
        if c in test.columns:
            test[c] = test[c].fillna("UNK").astype(str)

    X = train[features_num + features_cat].copy()
    y = pd.to_numeric(train["target"], errors="coerce").fillna(0.0)

    pipe = build_pipeline(features_num, features_cat)

    # Quick CV printout (optional)
    if SKLEARN_AVAILABLE and len(train) >= 5:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        vals = []
        for tr, va in kf.split(X):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[va])
            vals.append(rmse(y.iloc[va].to_numpy(), pred))
        print({"rmse_cv": float(np.mean(vals))})

    # Train full and predict
    pipe.fit(X, y)

    X_test = test[features_num + features_cat].copy()
    # Align to sample order by sample_id if present
    if "sample_id" in test.columns and "sample_id" in sample.columns:
        test_idx = test.set_index("sample_id").reindex(sample["sample_id"]).reset_index()
        X_test = test_idx[features_num + features_cat]

    pred = pipe.predict(X_test)
    # Clip to non-negative (biomass cannot be negative). Prevents potential Kaggle validation errors.
    pred = np.clip(pred.astype(float), 0.0, None)

    # Ensure target column name matches sample
    sub = sample.copy()
    target_col = "target" if "target" in sub.columns else sub.columns[-1]
    sub[target_col] = pred.astype(float)
    _validate_submission(sub)
    sub.to_csv(OUTPUT_SUB, index=False)
    print(f"Saved submission to {OUTPUT_SUB}")


if __name__ == "__main__":
    main()
