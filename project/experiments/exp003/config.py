"""Configuration for exp003 training.

This module defines a dataclass `CFG` used to control training behavior,
paths, model name, target columns, and runtime parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CFG:
    """Training configuration parameters.

    Update values via CLI in `train.py` as needed. Designed to be compatible
    with an inference that expects `convnext_tiny`, `IMG_SIZE=768`, and
    exactly 3 output targets: [Dry_Total_g, GDM_g, Dry_Green_g].
    """

    # Paths
    train_csv: str = "data/train.csv"
    image_dir: str = "data/train"
    output_dir: str = "project/results/exp003"

    # Model/targets
    model_name: str = "convnext_tiny"
    img_size: int = 768
    target_cols: List[str] = field(
        default_factory=lambda: [
            "Dry_Total_g",
            "GDM_g",
            "Dry_Green_g",
        ]
    )

    # Training params
    epochs: int = 10
    train_batch_size: int = 8
    valid_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"  # or "None"
    n_folds: int = 5
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 42
    patience: int = 3  # early stopping patience (epochs)

