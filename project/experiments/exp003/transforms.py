"""Albumentations transforms for exp003.

Provides train/valid transforms using ReplayCompose to apply identical random
augmentations to left/right image halves.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int) -> A.ReplayCompose:
    """Return ReplayCompose for training augmentations.

    Ensures identical random ops can be replayed on paired images.
    """
    return A.ReplayCompose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=10, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(img_size: int) -> A.ReplayCompose:
    """Return ReplayCompose for validation/test augmentations."""
    return A.ReplayCompose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

