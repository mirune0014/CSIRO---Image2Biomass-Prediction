"""Kaggle submission script: exp002 per-target mean baseline.

This script:
  - Loads train/test CSVs from Kaggle `/kaggle/input/<dataset>` auto-detected
    folder (or from `--data-dir` when provided, falling back to `./data`).
  - Computes per-`target_name` mean from train.
  - Predicts test targets using these means.
  - Writes `submission.csv` to the current working directory.

No external dependencies beyond pandas/numpy (available on Kaggle) are used.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


def find_data_dir(preferred: Optional[str] = None) -> str:
    """Resolve a dataset directory containing `train.csv` and `test.csv`.

    Search order:
      1) `preferred` (if provided)
      2) Any subdir under `/kaggle/input/` that has both CSVs
      3) `./data`
    """
    def has_csvs(path: str) -> bool:
        return os.path.exists(os.path.join(path, "train.csv")) and os.path.exists(
            os.path.join(path, "test.csv")
        )

    if preferred and has_csvs(preferred):
        return preferred

    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        try:
            for name in os.listdir(kaggle_input):
                cand = os.path.join(kaggle_input, name)
                if os.path.isdir(cand) and has_csvs(cand):
                    return cand
        except Exception:
            pass

    fallback = os.path.join(".", "data")
    if has_csvs(fallback):
        return fallback

    # As a last resort, return preferred or fallback (error likely later)
    return preferred or fallback


def compute_group_means(train_df: pd.DataFrame) -> Dict[str, float]:
    """Compute mean `target` per `target_name`."""
    return train_df.groupby("target_name")["target"].mean().to_dict()


def make_submission(data_dir: str, out_path: str = "submission.csv") -> None:
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    means = compute_group_means(train_df)
    global_mean = float(np.mean(list(means.values()))) if means else 0.0

    preds = (
        test_df["target_name"].map(means).fillna(global_mean).astype(float)
    )
    sub = pd.DataFrame({"sample_id": test_df["sample_id"], "target": preds})
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="exp002 Kaggle baseline: per-target mean")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset containing train.csv/test.csv")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV filename")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    print(f"Using data dir: {data_dir}")
    make_submission(data_dir, out_path=args.output)


if __name__ == "__main__":
    main()

