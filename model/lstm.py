# model/lstm.py
"""
Part B — LSTM Classifier

Takes a sequence of 2048-d frame feature vectors (output of ResNeXtExtractor),
models temporal dependencies across frames, and outputs a 2-class prediction
(Real / Fake) with a Softmax confidence score.

Architecture per spec:
  - Single LSTM layer, input_size=2048, hidden_size=2048
  - Dropout 0.4 applied to LSTM output
  - LeakyReLU activation
  - Adaptive average pooling across sequence dimension
  - Linear layer: 2048 → 2
  - Sequential layer for ordered frame processing
  - Softmax for confidence score (inference only)
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM-based temporal classifier for deepfake detection.

    Input:  (batch, sequence_length, 2048)  — sequence of frame features
    Output: (batch, 2)                      — logits [real_score, fake_score]
            OR
            (batch, 2)                      — softmax probabilities (inference)
    """

    def __init__(
        self,
        input_size:  int   = 2048,    # matches ResNeXt output dimension
        hidden_size: int   = 2048,    # 2048 hidden units as specified
        num_layers:  int   = 1,       # single LSTM layer as specified
        dropout:     float = 0.4,     # dropout on LSTM output
        num_classes: int   = 2,       # binary: Real / Fake
    ):
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout_p   = dropout

        # ── Sequential layer for ordered frame processing ─────────────────────
        # batch_first=True means input/output shape is (batch, seq, feature)
        # which is the natural ordering for our (batch, frames, 2048) tensor
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,        # (batch, seq, feature) ordering
            bidirectional=False,     # unidirectional: frame order matters
        )

        # ── Dropout applied manually to LSTM output ───────────────────────────
        # Note: PyTorch's LSTM dropout param only applies between layers
        # (no effect with num_layers=1). We apply it explicitly here.
        self.dropout = nn.Dropout(p=dropout)

        # ── LeakyReLU activation ──────────────────────────────────────────────
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # ── Adaptive average pooling across the sequence dimension ────────────
        # Reduces (batch, seq, 2048) → (batch, 1, 2048) → (batch, 2048)
        # This pools temporal information from all frames into one vector
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # ── Linear classifier: 2048 → 2 ──────────────────────────────────────
        self.classifier = nn.Linear(hidden_size, num_classes)

        # ── Softmax for confidence score output (inference only) ──────────────
        # NOT used during training — CrossEntropyLoss handles that internally
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x            : (batch, sequence_length, 2048) — frame feature sequence
            return_probs : if True, apply Softmax and return probabilities
                           (use at inference time only, never during training)

        Returns:
            logits or probs : (batch, 2)
        """
        # ── LSTM forward pass ─────────────────────────────────────────────────
        # lstm_out : (batch, seq, hidden_size) — output at every time step
        # (h_n, c_n) : final hidden and cell states (not used directly)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # ── Dropout + activation on full sequence output ──────────────────────
        lstm_out = self.dropout(lstm_out)       # (batch, seq, 2048)
        lstm_out = self.leaky_relu(lstm_out)    # (batch, seq, 2048)

        # ── Adaptive average pooling across time steps ────────────────────────
        # AdaptiveAvgPool1d expects (batch, channels, length)
        # Our tensor is  (batch, seq, 2048) → permute to (batch, 2048, seq)
        lstm_out = lstm_out.permute(0, 2, 1)    # (batch, 2048, seq)
        lstm_out = self.adaptive_pool(lstm_out) # (batch, 2048, 1)
        lstm_out = lstm_out.squeeze(2)          # (batch, 2048)

        # ── Linear classification head ─────────────────────────────────────────
        logits = self.classifier(lstm_out)      # (batch, 2)

        # ── Softmax at inference time for human-readable confidence scores ─────
        if return_probs:
            return self.softmax(logits)         # (batch, 2) — probabilities sum to 1

        return logits                           # (batch, 2) — raw logits for training