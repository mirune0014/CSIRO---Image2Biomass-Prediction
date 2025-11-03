"""Kaggle single-file training: two-stream 3-output model (exp003).

Implements the training pipeline described in project/experiments/task.md in a
single file suitable for Kaggle. Trains a Siamese two-stream model (left/right
halves) with `convnext_tiny` backbone and three regression heads for:
  - Dry_Total_g, GDM_g, Dry_Green_g

Saves best checkpoints per fold as `best_model_fold{fold}.pth` into the output
directory (default: `/kaggle/working/exp003`).

Usage (Kaggle):
  - Add the competition dataset as an input (must include train.csv, test.csv,
    and train/ images).
  - Run: `python submission/exp003_train.py`
  - Options: see `--help` for epochs, batch size, etc.

Note: If pretrained weights cannot be fetched (no internet), script falls back
to `pretrained=False` for the encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
try:
    import timm  # type: ignore
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False


# ----------------------------- configuration ---------------------------------


@dataclass
class CFG:
    # Paths
    data_dir: str = ""
    output_dir: str = "/kaggle/working/exp003"
    # Model/targets
    model_name: str = "convnext_tiny"
    img_size: int = 768
    target_cols: List[str] = field(
        default_factory=lambda: ["Dry_Total_g", "GDM_g", "Dry_Green_g"]
    )
    # Training
    epochs: int = 3
    train_batch_size: int = 8
    valid_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"  # or "None"
    n_folds: int = 5
    patience: int = 3
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------- data utilities ----------------------------------


def find_data_dir(preferred: str | None = None) -> str:
    """Find a directory containing train.csv and test.csv.

    Search order: preferred -> any child of /kaggle/input -> ./data
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
    return preferred or fallback


def prepare_training_dataframe(train_csv_path: str, target_cols: List[str]) -> pd.DataFrame:
    """Pivot long-format train.csv -> wide per-image with target_cols."""
    df = pd.read_csv(train_csv_path)
    df = df[["image_path", "target_name", "target"]]
    df = df[df["target_name"].isin(target_cols)].copy()
    wide = df.pivot_table(index="image_path", columns="target_name", values="target", aggfunc="mean").reset_index()
    for c in target_cols:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide.dropna(subset=target_cols).reset_index(drop=True)
    return wide[["image_path", *target_cols]]


# ------------------------------ dataset --------------------------------------


class TrainBiomassDataset(Dataset):
    """Dataset returning left/right tensors and 3-target labels.

    Applies synchronized augmentations using ReplayCompose.
    """

    def __init__(self, df: pd.DataFrame, transforms: A.ReplayCompose) -> None:
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

        t_left = self.transforms(image=img_left)
        left_tensor = t_left["image"]
        t_right = self.transforms.replay(t_left["replay"], image=img_right)
        right_tensor = t_right["image"]

        labels = torch.tensor([row[c] for c in self.target_cols], dtype=torch.float32)
        return left_tensor, right_tensor, labels


# ------------------------------ transforms -----------------------------------


def get_train_transforms(img_size: int) -> A.ReplayCompose:
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
    return A.ReplayCompose(
        [A.Resize(img_size, img_size), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )


# --------------------------------- model -------------------------------------


class BiomassModel(nn.Module):
    """Two-stream regressor: shared encoder, three regression heads."""

    def __init__(self, model_name: str, pretrained: bool, n_targets: int) -> None:
        super().__init__()
        # try timm encoder; if unavailable or weights can't be fetched, fallback
        if _HAS_TIMM:
            try:
                self.encoder = timm.create_model(
                    model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
                )
            except Exception:
                self.encoder = timm.create_model(
                    model_name, pretrained=False, num_classes=0, global_pool="avg"
                )
            feat_dim = getattr(self.encoder, "num_features", 768)
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            feat_dim = 256
        combined = feat_dim * 2
        hidden = max(256, feat_dim // 2)

        self.head_total = nn.Sequential(nn.Linear(combined, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
        self.head_gdm = nn.Sequential(nn.Linear(combined, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
        self.head_green = nn.Sequential(nn.Linear(combined, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        f_left = self.encoder(x_left)
        f_right = self.encoder(x_right)
        feats = torch.cat([f_left, f_right], dim=1)
        return self.head_total(feats), self.head_gdm(feats), self.head_green(feats)


def build_model(model_name: str, n_targets: int, device: str) -> BiomassModel:
    m = BiomassModel(model_name=model_name, pretrained=True, n_targets=n_targets)
    return m.to(device)


# ------------------------------- training ------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(params, cfg: CFG):
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def make_scheduler(optimizer, cfg: CFG):
    if cfg.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "None":
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


def compute_loss(outputs, labels) -> torch.Tensor:
    out_total, out_gdm, out_green = outputs
    mse = nn.MSELoss()
    loss = (mse(out_total, labels[:, 0:1]) + mse(out_gdm, labels[:, 1:2]) + mse(out_green, labels[:, 2:3])) / 3.0
    return loss


def train_one_epoch(loader: DataLoader, model, optimizer, scheduler, device: str) -> float:
    model.train()
    losses = []
    for left, right, labels in loader:
        left, right, labels = left.to(device), right.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(left, right)
        loss = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate_one_epoch(loader: DataLoader, model, device: str) -> float:
    model.eval()
    losses = []
    for left, right, labels in loader:
        left, right, labels = left.to(device), right.to(device), labels.to(device)
        loss = compute_loss(model(left, right), labels)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def kfold_train(cfg: CFG, wide_df: pd.DataFrame) -> Dict[str, List[float]]:
    os.makedirs(cfg.output_dir, exist_ok=True)

    idx = np.arange(len(wide_df))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, cfg.n_folds)

    best_losses: List[float] = []
    for fold in range(cfg.n_folds):
        val_idx = folds[fold]
        train_idx = np.concatenate([folds[i] for i in range(cfg.n_folds) if i != fold])
        df_tr = wide_df.iloc[train_idx].reset_index(drop=True)
        df_va = wide_df.iloc[val_idx].reset_index(drop=True)

        train_ds = TrainBiomassDataset(df_tr, transforms=get_train_transforms(cfg.img_size))
        valid_ds = TrainBiomassDataset(df_va, transforms=get_valid_transforms(cfg.img_size))
        train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=cfg.valid_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        model = build_model(cfg.model_name, n_targets=len(cfg.target_cols), device=cfg.device)
        optimizer = make_optimizer(model.parameters(), cfg)
        scheduler = make_scheduler(optimizer, cfg)

        best_val = float("inf")
        no_improve = 0
        ckpt = os.path.join(cfg.output_dir, f"best_model_fold{fold}.pth")
        for epoch in range(cfg.epochs):
            tr_loss = train_one_epoch(train_loader, model, optimizer, scheduler, cfg.device)
            va_loss = validate_one_epoch(valid_loader, model, cfg.device)
            print(json.dumps({"fold": fold, "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss}))
            if va_loss < best_val - 1e-6:
                best_val = va_loss
                no_improve = 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_improve += 1
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            if no_improve >= cfg.patience:
                break
        best_losses.append(best_val)
        print(f"Fold {fold} best val loss: {best_val:.6f} -> saved {ckpt}")

    return {"best_losses": best_losses, "mean_best_val_loss": [float(np.mean(best_losses))]}


# --------------------------------- main --------------------------------------


def build_image_paths(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    # train.csv uses relative like 'train/IDxxxx.jpg'; prepend data_dir
    def _p(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(data_dir, p)

    out = df.copy()
    out["image_path"] = out["image_path"].apply(_p)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kaggle two-stream training (exp003)")
    p.add_argument("--data-dir", type=str, default=None, help="Dataset root containing train.csv/test.csv")
    p.add_argument("--output-dir", type=str, default=CFG.output_dir)
    p.add_argument("--model-name", type=str, default=CFG.model_name)
    p.add_argument("--img-size", type=int, default=CFG.img_size)
    p.add_argument("--epochs", type=int, default=CFG.epochs)
    p.add_argument("--train-batch-size", type=int, default=CFG.train_batch_size)
    p.add_argument("--valid-batch-size", type=int, default=CFG.valid_batch_size)
    p.add_argument("--lr", type=float, default=CFG.learning_rate)
    p.add_argument("--weight-decay", type=float, default=CFG.weight_decay)
    p.add_argument("--optimizer", type=str, default=CFG.optimizer)
    p.add_argument("--scheduler", type=str, default=CFG.scheduler)
    p.add_argument("--n-folds", type=int, default=CFG.n_folds)
    p.add_argument("--patience", type=int, default=CFG.patience)
    p.add_argument("--num-workers", type=int, default=CFG.num_workers)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--device", type=str, default=CFG.device)
    # In notebook environments (Kaggle/Colab), extra args like `-f` may be passed.
    # Use parse_known_args to ignore unknowns and avoid SystemExit.
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    data_dir = find_data_dir(args.data_dir)
    cfg = CFG(
        data_dir=data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        img_size=args.img_size,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        n_folds=args.n_folds,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    train_csv_path = os.path.join(cfg.data_dir, "train.csv")
    wide_df = prepare_training_dataframe(train_csv_path, cfg.target_cols)
    wide_df = build_image_paths(wide_df, cfg.data_dir)

    summary = kfold_train(cfg, wide_df)
    # Save summary JSON
    with open(os.path.join(cfg.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
