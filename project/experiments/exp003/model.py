"""Two-stream BiomassModel for exp003.

This model applies a shared backbone (from timm) to left/right images,
concatenates the pooled features, and predicts three regression outputs:
Dry_Total_g, GDM_g, Dry_Green_g.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import timm


class BiomassModel(nn.Module):
    """Two-stream regression model with shared encoder and three heads.

    Args:
        model_name: Backbone name for timm.create_model.
        pretrained: Whether to load pretrained weights for the backbone.
        n_targets: Number of regression outputs (expected 3).
    """

    def __init__(self, model_name: str, pretrained: bool, n_targets: int) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        # Determine feature dimension from encoder
        if hasattr(self.encoder, "num_features"):
            feat_dim = self.encoder.num_features
        else:
            # Fallback: run a dummy forward to infer dim (rarely needed)
            feat_dim = 768

        combined_dim = feat_dim * 2
        hidden = max(256, feat_dim // 2)

        self.head_total = nn.Sequential(
            nn.Linear(combined_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )
        self.head_gdm = nn.Sequential(
            nn.Linear(combined_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )
        self.head_green = nn.Sequential(
            nn.Linear(combined_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning three outputs (B, 1) each."""
        f_left = self.encoder(x_left)
        f_right = self.encoder(x_right)
        feats = torch.cat([f_left, f_right], dim=1)
        out_total = self.head_total(feats)
        out_gdm = self.head_gdm(feats)
        out_green = self.head_green(feats)
        return out_total, out_gdm, out_green


def build_model(model_name: str, n_targets: int, device: str = "cuda") -> BiomassModel:
    """Factory to build and move the model to device with pretrained=True."""
    model = BiomassModel(model_name=model_name, pretrained=True, n_targets=n_targets)
    return model.to(device)

