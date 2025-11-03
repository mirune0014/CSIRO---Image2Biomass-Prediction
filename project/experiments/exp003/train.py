"""exp003 training runner.

Implements the training pipeline described in project/experiments/task.md:
 - CFG parameters
 - Albumentations train/valid transforms (ReplayCompose)
 - TrainBiomassDataset with synchronized left/right augmentations
 - Two-stream BiomassModel with three regression heads
 - KFold training with best checkpoint saving per fold

Usage:
    python -m project.experiments.exp003.train \
        --train-csv data/train.csv \
        --image-dir data/train \
        --output-dir project/results/exp003
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from project.experiments.exp003.config import CFG as DefaultCFG, CFG
from project.experiments.exp003.dataset import (
    TrainBiomassDataset,
    prepare_training_dataframe,
)
from project.experiments.exp003.model import build_model
from project.experiments.exp003.transforms import get_train_transforms, get_valid_transforms
from project.utils.experiment_logger import ExperimentLogger


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
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
    """Compute average MSE across three outputs."""
    out_total, out_gdm, out_green = outputs
    tgt_total = labels[:, 0:1]
    tgt_gdm = labels[:, 1:2]
    tgt_green = labels[:, 2:3]
    mse = nn.MSELoss()
    loss_total = mse(out_total, tgt_total)
    loss_gdm = mse(out_gdm, tgt_gdm)
    loss_green = mse(out_green, tgt_green)
    return (loss_total + loss_gdm + loss_green) / 3.0


def train_one_epoch(loader: DataLoader, model, optimizer, scheduler, device: str) -> float:
    model.train()
    losses = []
    for left, right, labels in loader:
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(left, right)
        loss = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate_one_epoch(loader: DataLoader, model, device: str) -> float:
    model.eval()
    losses = []
    for left, right, labels in loader:
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)
        outputs = model(left, right)
        loss = compute_loss(outputs, labels)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def run_training(cfg: CFG) -> Dict[str, float]:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Prepare data (wide format per image)
    wide_df = prepare_training_dataframe(cfg.train_csv, cfg.target_cols)
    # Prepend image_dir to relative paths if needed
    if cfg.image_dir and not os.path.isabs(cfg.image_dir):
        # Replace any leading 'train/' with actual dir when necessary
        wide_df["image_path"] = wide_df["image_path"].apply(
            lambda p: p if os.path.isabs(p) else os.path.join(os.path.dirname(cfg.train_csv), p)
        )

    # KFold split on rows (unique images)
    idx = np.arange(len(wide_df))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, cfg.n_folds)

    # Logger
    logger = ExperimentLogger(
        experiment_id="exp003",
        hypothesis_id="H-02",
        change_type="model",
        hypothesis=(
            "2ストリーム画像エンコーダ（convnext_tiny）により、"
            "3出力（Total/GDM/Green）回帰を同時学習すると、画像のみでも安定した損失低下が得られる"
        ),
        expected_effect="KFold 検証で val loss が継続的に改善し、汎化性能が向上する",
        metrics=["val_loss"],
    )
    logger.log_params(
        {
            "train_csv": cfg.train_csv,
            "image_dir": cfg.image_dir,
            "output_dir": cfg.output_dir,
            "model_name": cfg.model_name,
            "img_size": cfg.img_size,
            "target_cols": cfg.target_cols,
            "epochs": cfg.epochs,
            "train_batch_size": cfg.train_batch_size,
            "valid_batch_size": cfg.valid_batch_size,
            "learning_rate": cfg.learning_rate,
            "optimizer": cfg.optimizer,
            "scheduler": cfg.scheduler,
            "n_folds": cfg.n_folds,
            "seed": cfg.seed,
        }
    )

    best_losses = []
    for fold in range(cfg.n_folds):
        val_idx = folds[fold]
        train_idx = np.concatenate([folds[i] for i in range(cfg.n_folds) if i != fold])

        df_tr = wide_df.iloc[train_idx].reset_index(drop=True)
        df_va = wide_df.iloc[val_idx].reset_index(drop=True)

        train_ds = TrainBiomassDataset(df_tr, transforms=get_train_transforms(cfg.img_size))
        valid_ds = TrainBiomassDataset(df_va, transforms=get_valid_transforms(cfg.img_size))

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=cfg.valid_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        model = build_model(cfg.model_name, n_targets=len(cfg.target_cols), device=cfg.device)
        optimizer = make_optimizer(model.parameters(), cfg)
        scheduler = make_scheduler(optimizer, cfg)

        best_val = float("inf")
        epochs_no_improve = 0
        ckpt_path = os.path.join(cfg.output_dir, f"best_model_fold{fold}.pth")
        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(train_loader, model, optimizer, scheduler, cfg.device)
            val_loss = validate_one_epoch(valid_loader, model, cfg.device)

            # Early stopping on val_loss
            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1

            # ReduceLROnPlateau support (if added later)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

            # Log progress snapshot per epoch
            logger.log_metrics({f"fold{fold}_epoch{epoch}_val_loss": float(val_loss)})

            if epochs_no_improve >= cfg.patience:
                break

        best_losses.append(best_val)
        logger.add_note(f"Fold {fold}: best_val_loss={best_val:.6f}, saved={ckpt_path}")

    summary = {"mean_best_val_loss": float(np.mean(best_losses)), "best_losses": best_losses}
    logger.log_metrics(summary)
    logger.finalize()
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp003: two-stream 3-output training")
    p.add_argument("--train-csv", type=str, default=DefaultCFG.train_csv)
    p.add_argument("--image-dir", type=str, default=DefaultCFG.image_dir)
    p.add_argument("--output-dir", type=str, default=DefaultCFG.output_dir)
    p.add_argument("--model-name", type=str, default=DefaultCFG.model_name)
    p.add_argument("--img-size", type=int, default=DefaultCFG.img_size)
    p.add_argument("--epochs", type=int, default=DefaultCFG.epochs)
    p.add_argument("--train-batch-size", type=int, default=DefaultCFG.train_batch_size)
    p.add_argument("--valid-batch-size", type=int, default=DefaultCFG.valid_batch_size)
    p.add_argument("--lr", type=float, default=DefaultCFG.learning_rate)
    p.add_argument("--weight-decay", type=float, default=DefaultCFG.weight_decay)
    p.add_argument("--optimizer", type=str, default=DefaultCFG.optimizer)
    p.add_argument("--scheduler", type=str, default=DefaultCFG.scheduler)
    p.add_argument("--n-folds", type=int, default=DefaultCFG.n_folds)
    p.add_argument("--num-workers", type=int, default=DefaultCFG.num_workers)
    p.add_argument("--device", type=str, default=DefaultCFG.device)
    p.add_argument("--seed", type=int, default=DefaultCFG.seed)
    p.add_argument("--patience", type=int, default=DefaultCFG.patience)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DefaultCFG(
        train_csv=args.train_csv,
        image_dir=args.image_dir,
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
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
    )

    summary = run_training(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

