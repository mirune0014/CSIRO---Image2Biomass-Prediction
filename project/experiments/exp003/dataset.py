"""Datasets for exp003 training.

Implements TrainBiomassDataset that loads images, splits into left/right
halves, and applies synchronized Albumentations transforms via ReplayCompose.
Also includes a helper to pivot the training CSV to 3-target wide format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_MAP = {
    "Dry_Total_g": "Dry_Total_g",
    "GDM_g": "GDM_g",
    "Dry_Green_g": "Dry_Green_g",
}


def prepare_training_dataframe(train_csv_path: str, target_cols: List[str]) -> pd.DataFrame:
    """Load train.csv and pivot to one row per image with 3 target columns.

    Args:
        train_csv_path: Path to train.csv in long format.
        target_cols: List of targets to keep (3 entries expected).

    Returns:
        DataFrame with columns: image_path, and each target in `target_cols`.
    """
    df = pd.read_csv(train_csv_path)
    keep_cols = ["image_path", "target_name", "target"]
    df = df[keep_cols]
    df = df[df["target_name"].isin(target_cols)].copy()
    wide = df.pivot_table(index="image_path", columns="target_name", values="target", aggfunc="mean")
    wide = wide.reset_index()
    # Ensure all target columns exist
    for c in target_cols:
        if c not in wide.columns:
            wide[c] = np.nan
    # Drop rows with any missing labels among targets
    wide = wide.dropna(subset=target_cols).reset_index(drop=True)
    return wide[["image_path", *target_cols]]


class TrainBiomassDataset(Dataset):
    """Training dataset producing left/right tensors and 3-target labels.

    Expects a DataFrame with columns: image_path, [targets...].
    """

    def __init__(self, df: pd.DataFrame, transforms) -> None:
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.target_cols = [c for c in df.columns if c != "image_path"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        mid = w // 2
        img_left = img[:, :mid, :]
        img_right = img[:, mid:, :]

        # Apply identical augmentations via ReplayCompose
        t_left = self.transforms(image=img_left)
        left_tensor = t_left["image"]
        t_right = self.transforms.replay(t_left["replay"], image=img_right)
        right_tensor = t_right["image"]

        labels = torch.tensor([row[c] for c in self.target_cols], dtype=torch.float32)
        return left_tensor, right_tensor, labels

