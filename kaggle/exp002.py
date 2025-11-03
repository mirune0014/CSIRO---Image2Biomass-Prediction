"""Kaggle inference script for exp002 (tabular + EfficientNet embeddings).

Recreates the local experiment exp002 with fixed Kaggle paths:
- Data root: /kaggle/input/csiro-biomass
- Output:   /kaggle/working/submission.csv
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PATH_DATA = Path("/kaggle/input/csiro-biomass")
PATH_TRAIN = PATH_DATA / "train.csv"
PATH_TEST = PATH_DATA / "test.csv"
PATH_SAMPLE = PATH_DATA / "sample_submission.csv"
OUTPUT_SUB = Path("/kaggle/working/submission.csv")

NUMERIC_FEATURES = ["Pre_GSHH_NDVI", "Height_Ave_cm", "dayofyear_sin", "dayofyear_cos"]
TEXT_FEATURES = ["State", "Species", "target_name"]
MODEL_NAME = "tf_efficientnet_b0"
PCA_COMPONENTS = 128
ALPHA = 0.5
BATCH_SIZE = 32
CV_FOLDS = 5
SEED = 42


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Kaggle CSVs and derive sine/cosine seasonal features."""
    train = pd.read_csv(PATH_TRAIN)
    test = pd.read_csv(PATH_TEST)
    sample = pd.read_csv(PATH_SAMPLE)

    numeric_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm", "target"]
    for df in (train, test):
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        dt = pd.to_datetime(df.get("Sampling_Date"), errors="coerce")
        df["day_of_year"] = dt.dt.dayofyear
        angle = 2 * math.pi * (df["day_of_year"].fillna(0) / 366.0)
        df["dayofyear_sin"] = np.sin(angle)
        df["dayofyear_cos"] = np.cos(angle)

    return train, test, sample


def compute_embeddings(
    paths: Iterable[str],
    root: Path,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Compute embeddings for relative paths (train/xxx.jpg or test/xxx.jpg)."""
    unique = sorted(set(paths))
    embeddings: Dict[str, np.ndarray] = {}
    batch: List[torch.Tensor] = []
    keys: List[str] = []

    for rel_path in unique:
        img_path = root / rel_path
        with Image.open(img_path).convert("RGB") as img:
            batch.append(transform(img))
        keys.append(rel_path)
        if len(batch) >= batch_size:
            with torch.no_grad():
                feats = model(torch.stack(batch).to(device)).cpu().numpy()
            embeddings.update({k: f.astype(np.float32) for k, f in zip(keys, feats)})
            batch.clear()
            keys.clear()
    if batch:
        with torch.no_grad():
            feats = model(torch.stack(batch).to(device)).cpu().numpy()
        embeddings.update({k: f.astype(np.float32) for k, f in zip(keys, feats)})
    return embeddings


class TabularImageRegressor:
    """Ridge regression on tabular features + PCA-reduced image embeddings."""

    def __init__(self, alpha: float, pca_components: int, random_state: int | None = None) -> None:
        self.alpha = float(alpha)
        self.pca_components = int(pca_components)
        self.random_state = random_state
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.numeric_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.pca: PCA | None = None
        self.model = Ridge(alpha=self.alpha, random_state=random_state)

    def _tabular(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        num = df.reindex(columns=NUMERIC_FEATURES).apply(pd.to_numeric, errors="coerce")
        cat = df.reindex(columns=TEXT_FEATURES).fillna("UNK").astype(str)
        if fit:
            num_scaled = self.numeric_scaler.fit_transform(self.numeric_imputer.fit_transform(num))
            cat_encoded = self.categorical_encoder.fit_transform(cat)
        else:
            num_scaled = self.numeric_scaler.transform(self.numeric_imputer.transform(num))
            cat_encoded = self.categorical_encoder.transform(cat)
        return np.hstack([num_scaled, cat_encoded])

    def _image(self, embeddings: np.ndarray, fit: bool) -> np.ndarray:
        if self.pca_components <= 0 or embeddings.shape[1] <= self.pca_components:
            if fit:
                self.pca = None
            return embeddings
        if fit:
            comps = min(self.pca_components, embeddings.shape[0] - 1)
            self.pca = PCA(n_components=max(1, comps), random_state=self.random_state)
            return self.pca.fit_transform(embeddings)
        if self.pca is None:
            return embeddings
        return self.pca.transform(embeddings)

    def fit(self, df: pd.DataFrame, embeddings: np.ndarray, target: np.ndarray) -> None:
        tab = self._tabular(df, fit=True)
        img = self._image(embeddings, fit=True)
        self.model.fit(np.hstack([tab, img]), target)

    def predict(self, df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        tab = self._tabular(df, fit=False)
        img = self._image(embeddings, fit=False)
        return self.model.predict(np.hstack([tab, img]))


def cross_validate(
    df: pd.DataFrame,
    target: np.ndarray,
    embeddings: np.ndarray,
    alpha: float,
    pca_components: int,
    folds: int,
    seed: int,
) -> Tuple[float, float]:
    """Perform GroupKFold CV grouped by image_path for sanity."""
    gkf = GroupKFold(n_splits=folds)
    groups = df["image_path"].astype(str).to_numpy()
    rmses, maes = [], []
    for tr_idx, va_idx in gkf.split(df, target, groups=groups):
        model = TabularImageRegressor(alpha, pca_components, seed)
        model.fit(df.iloc[tr_idx], embeddings[tr_idx], target[tr_idx])
        preds = model.predict(df.iloc[va_idx], embeddings[va_idx])
        rmses.append(math.sqrt(mean_squared_error(target[va_idx], preds)))
        maes.append(mean_absolute_error(target[va_idx], preds))
    return float(np.mean(rmses)), float(np.mean(maes))


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_df, test_df, sample_df = load_data()
    y = train_df["target"].to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, global_pool="avg").to(device)
    model.eval()
    transform = create_transform(**resolve_data_config({}, model=model), is_training=False)

    train_paths = train_df["image_path"].astype(str).tolist()
    test_paths = test_df["image_path"].astype(str).tolist()

    print("Extracting train embeddings...")
    train_map = compute_embeddings(train_paths, PATH_DATA, model, transform, device, BATCH_SIZE)
    train_embeddings = np.stack([train_map[p] for p in train_paths])

    print("Extracting test embeddings...")
    test_map = compute_embeddings(test_paths, PATH_DATA, model, transform, device, BATCH_SIZE)
    test_embeddings = np.stack([test_map[p] for p in test_paths])

    cv_rmse, cv_mae = cross_validate(
        train_df,
        y,
        train_embeddings,
        alpha=ALPHA,
        pca_components=PCA_COMPONENTS,
        folds=CV_FOLDS,
        seed=SEED,
    )
    print(f"CV RMSE: {cv_rmse:.4f}, CV MAE: {cv_mae:.4f}")

    final_model = TabularImageRegressor(ALPHA, PCA_COMPONENTS, SEED)
    final_model.fit(train_df, train_embeddings, y)
    test_preds = final_model.predict(test_df, test_embeddings).astype(float)

    pred_map = dict(zip(test_df["sample_id"].astype(str), test_preds))
    submission = sample_df.copy()
    target_col = "target" if "target" in submission.columns else submission.columns[-1]
    submission[target_col] = submission["sample_id"].map(pred_map).astype(float)
    if submission[target_col].isna().any():
        raise ValueError("Missing predictions for some sample_ids.")

    submission.to_csv(OUTPUT_SUB, index=False)
    print(f"Saved submission to {OUTPUT_SUB}")


if __name__ == "__main__":
    main()
