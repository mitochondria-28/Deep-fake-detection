# model/gradcam.py
"""
Grad-CAM for the Detectra ResNeXt50 feature extractor.

Design notes (read before modifying):
  - Hooks attach to `feature_extractor.layer4`, the last conv block
    before `avgpool`. This is the deepest layer that still has spatial
    structure (batch, 2048, 4, 4) — required for a heatmap.
  - No weights are modified. Hooks only intercept activations/gradients
    at runtime; the checkpoint is loaded and used exactly as trained.
  - Per-frame scoring: each face frame is run through `feature_extractor`
    individually, then passed through the model's REAL trained LSTM
    classifier as a sequence of length 1. This produces a genuine class
    score using the actual trained decision head — not a proxy metric —
    while still letting us isolate which frame produced which heatmap.
  - This is an inference-time explainability tool. It must never be
    called during training and never touches checkpoint weights.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM hooked to Detectra's ResNeXt50 layer4.

    Usage:
        cam = GradCAM(model, device)
        heatmap, label, conf = cam.generate(face_frame_rgb_uint8)
        cam.remove_hooks()   # always call when done
    """

    def __init__(self, model, device: torch.device):
        """
        Args:
            model  : the loaded Detectra instance (already .eval(), weights loaded)
            device : torch.device used for inference
        """
        self.model  = model
        self.device = device

        # Target layer — last conv block before global average pooling
        self.target_layer = model.feature_extractor.layer4

        self._activations = None   # forward output of target_layer
        self._gradients    = None  # gradient of score w.r.t. target_layer output

        self._fwd_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )
        self._bwd_handle = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def _forward_hook(self, module, input, output):
        """Captures layer4's output feature maps: (1, 2048, 4, 4)."""
        self._activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        """Captures gradient of the score w.r.t. layer4's output."""
        self._gradients = grad_output[0]

    def remove_hooks(self):
        """Always call this after use to avoid hook buildup across requests."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    # ── Core Grad-CAM computation ─────────────────────────────────────────────

    def _frame_to_input_tensor(self, face_frame_rgb: np.ndarray) -> torch.Tensor:
        """
        Convert a single (112,112,3) uint8 RGB face frame into the
        normalised tensor the model expects: (1, 3, 112, 112).
        """
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        tensor = transform(face_frame_rgb)          # (3, 112, 112)
        return tensor.unsqueeze(0).to(self.device)  # (1, 3, 112, 112)

    def generate(self, face_frame_rgb: np.ndarray, target_class: int = None) -> tuple:
        """
        Generate a Grad-CAM heatmap for a single face frame.

        Args:
            face_frame_rgb : (112, 112, 3) uint8 RGB numpy array
            target_class   : 0=Real, 1=Fake, or None to use model's predicted class

        Returns:
            heatmap    : (112, 112) float32 array, values in [0, 1]
            pred_label : "REAL" or "FAKE" — class Grad-CAM explains
            pred_conf  : float — softmax confidence for that class, this frame alone
        """
        self.model.zero_grad()

        # ── 1. Forward single frame through feature_extractor ─────────────────
        frame_tensor = self._frame_to_input_tensor(face_frame_rgb)  # (1,3,112,112)

        # feature_extractor forward triggers the forward hook above
        features = self.model.feature_extractor(frame_tensor)      # (1, 2048)

        # ── 2. Run through the REAL trained LSTM as a length-1 sequence ───────
        # This reuses actual trained weights — no new layers, no retraining.
        seq_features = features.unsqueeze(1)                       # (1, 1, 2048)
        logits = self.model.lstm_classifier(seq_features, return_probs=False)  # (1, 2)

        probs = F.softmax(logits, dim=1)                            # (1, 2)

        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())

        pred_label = "FAKE" if target_class == 1 else "REAL"
        pred_conf  = float(probs[0, target_class].item()) * 100

        # ── 3. Backward pass — gradient of class score w.r.t. layer4 output ───
        score = logits[0, target_class]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Check that layer4 is reachable in the forward pass."
            )

        activations = self._activations.detach()   # (1, 2048, 4, 4)
        gradients   = self._gradients.detach()      # (1, 2048, 4, 4)

        # ── 4. Global-average-pool gradients → channel importance weights ─────
        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, 2048, 1, 1)

        # ── 5. Weighted combination of activation maps ────────────────────────
        cam = (weights * activations).sum(dim=1, keepdim=True)   # (1, 1, 4, 4)
        cam = F.relu(cam)   # only positive influence, standard Grad-CAM step

        # ── 6. Upsample to 112×112 and normalise to [0, 1] ─────────────────────
        cam = F.interpolate(
            cam, size=(112, 112), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()   # (112, 112)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)   # flat gradient — avoid divide-by-zero

        return cam, pred_label, pred_conf


def overlay_heatmap(face_frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto the original face frame.

    Args:
        face_frame_rgb : (112, 112, 3) uint8 RGB original frame
        heatmap        : (112, 112) float32 array in [0, 1]
        alpha          : blend strength of the heatmap (0=invisible, 1=opaque)

    Returns:
        (112, 112, 3) uint8 RGB image with heatmap overlaid.
        Red/yellow = high importance, blue/dark = low importance.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (
        face_frame_rgb.astype(np.float32) * (1 - alpha) +
        heatmap_color.astype(np.float32) * alpha
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)