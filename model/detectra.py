# model/Detectra.py
"""
Detectra — Hybrid ResNeXt50 + LSTM deepfake detection model.

Full forward pass:
  1. Input video tensor  : (batch, seq, 3, 112, 112)
  2. Per-frame ResNeXt   : each frame → 2048-d feature vector
  3. Stack into sequence : (batch, seq, 2048)
  4. LSTM classifier     : temporal modelling → (batch, 2)
  5. Output              : logits (training) or softmax probs (inference)
"""

import torch
import torch.nn as nn

from model.resnext import ResNeXtExtractor
from model.lstm import LSTMClassifier


class Detectra(nn.Module):
    """
    Hybrid ResNeXt50 + LSTM model for binary deepfake detection.

    Input:
        x : (batch, sequence_length, 3, 112, 112)
            A batch of videos, each represented as a sequence of face frames.

    Output (training):
        logits : (batch, 2)  — raw scores, feed directly to CrossEntropyLoss

    Output (inference, return_probs=True):
        probs  : (batch, 2)  — [P(Real), P(Fake)], confidence percentages
    """

    def __init__(
        self,
        pretrained:      bool  = True,
        lstm_hidden:     int   = 2048,
        lstm_layers:     int   = 1,
        lstm_dropout:    float = 0.4,
        num_classes:     int   = 2,
    ):
        super().__init__()

        # ── Part A: ResNeXt50 feature extractor ───────────────────────────────
        self.feature_extractor = ResNeXtExtractor(pretrained=pretrained)

        # ── Part B: LSTM temporal classifier ──────────────────────────────────
        self.lstm_classifier = LSTMClassifier(
            input_size=2048,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            num_classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x            : (batch, seq, 3, 112, 112)
            return_probs : False during training, True during inference

        Returns:
            (batch, 2) — logits or probabilities
        """
        batch_size, seq_len, C, H, W = x.shape

        # ── Step 1: Reshape so ResNeXt processes all frames independently ──────
        # Merge batch and sequence dims: (batch*seq, 3, 112, 112)
        x = x.view(batch_size * seq_len, C, H, W)

        # ── Step 2: Extract per-frame features ────────────────────────────────
        # ResNeXt maps each (3, 112, 112) frame → 2048-d vector
        features = self.feature_extractor(x)     # (batch*seq, 2048)

        # ── Step 3: Restore sequence structure ────────────────────────────────
        # Split merged dim back: (batch, seq, 2048)
        features = features.view(batch_size, seq_len, -1)

        # ── Step 4: LSTM temporal classification ──────────────────────────────
        output = self.lstm_classifier(features, return_probs=return_probs)

        return output                             # (batch, 2)

    def get_trainable_params(self) -> dict:
        """
        Returns a summary of trainable vs frozen parameter counts.
        Useful for verifying the fine-tuning strategy is applied correctly.
        """
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params    = total_params - trainable_params

        return {
            "total":     total_params,
            "trainable": trainable_params,
            "frozen":    frozen_params,
            "trainable_pct": f"{100 * trainable_params / total_params:.1f}%"
        }