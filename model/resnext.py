# model/resnext.py
"""
Part A — ResNeXt50 Feature Extractor

Uses pretrained resnext50_32x4d as a per-frame feature extractor.
Removes the classification head and replaces it with Identity so the
last average pooling layer outputs raw 2048-d feature vectors.

Fine-tuning strategy:
  - Layers 1–3 : FROZEN  (low-level edges/textures, reuse ImageNet weights)
  - Layer 4    : UNFROZEN (high-level semantics, adapt to face/deepfake data)
  - Extra head : Two added FC layers to project and enrich features
"""

import torch
import torch.nn as nn
from torchvision import models


class Identity(nn.Module):
    """Passthrough layer used to strip the ResNeXt classifier head."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResNeXtExtractor(nn.Module):
    """
    Pretrained ResNeXt50-32x4d with the FC head replaced by an
    enrichment block. Outputs a 2048-d feature vector per frame.

    Input:  (batch, 3, 112, 112)   — one frame per sample
    Output: (batch, 2048)          — feature vector per frame
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── Load pretrained backbone ──────────────────────────────────────────
        weights = (
            models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        backbone = models.resnext50_32x4d(weights=weights)

        # ── Remove the final FC classifier (outputs 1000 ImageNet classes) ───
        # Replace with Identity so forward() returns the 2048-d pooled vector
        backbone.fc = Identity()

        # ── Split backbone into named stages for selective freezing ───────────
        self.conv1   = backbone.conv1      # stem conv
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1     # residual block 1
        self.layer2  = backbone.layer2     # residual block 2
        self.layer3  = backbone.layer3     # residual block 3
        self.layer4  = backbone.layer4     # residual block 4  ← fine-tuned
        self.avgpool = backbone.avgpool    # global average pool → (batch, 2048, 1, 1)

        # ── Freeze early layers (stem + blocks 1–3) ───────────────────────────
        frozen_modules = [
            self.conv1, self.bn1,
            self.layer1, self.layer2, self.layer3
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

        # ── layer4 stays unfrozen — adapts to face/deepfake domain ───────────
        for param in self.layer4.parameters():
            param.requires_grad = True

        # ── Extra fine-tuning head (added layers as specified) ────────────────
        # Projects 2048 → 2048 with normalisation and activation.
        # Gives the extractor capacity to learn deepfake-specific features
        # beyond what the ImageNet backbone captured.
        self.extra_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, 3, 112, 112) — single frame batch

        Returns:
            features : (batch, 2048) — feature vector per frame
        """
        # ResNeXt backbone forward (without FC)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)          # (batch, 2048, 1, 1)
        x = torch.flatten(x, 1)     # (batch, 2048)

        # Extra fine-tuning head
        x = self.extra_head(x)      # (batch, 2048)

        return x