# webapp/detector/inference.py
"""
Deepfakedetection Inference Engine for Django

Handles:
  - Model + MTCNN loading (called once at startup by apps.py)
  - Per-request video preprocessing (same pipeline as Step 1)
  - Model inference with confidence score
  - Face region data for display

MPS (Apple Silicon) is deliberately skipped because Adaptive Average
Pooling used in ResNeXt is not yet fully implemented on MPS.
See: https://github.com/pytorch/pytorch/issues/96056
"""

import os
import sys
import cv2
import uuid
import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module-level singletons (loaded once, shared across all requests) ─────────
_model  = None
_mtcnn  = None
_device = None


def load_models() -> None:
    """
    Load Deepfakedetection model and MTCNN detector into module-level globals.
    Called once by detector/apps.py at Django startup.
    """
    global _model, _mtcnn, _device

    import django.conf as conf
    settings = conf.settings

    # ── Device ────────────────────────────────────────────────────────────────
    # MPS skipped intentionally — Adaptive Pool not supported on MPS yet
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    logger.info(f"Inference device: {_device}")

    # ── Deepfakedetection model ────────────────────────────────────────────────────────
    checkpoint_path = settings.CHECKPOINT_PATH
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Complete Step 5 training before starting the web app."
        )

    # Add Deepfakedetection root to path so model package is importable
    Deepfakedetection_root = str(Path(checkpoint_path).parent.parent)
    if Deepfakedetection_root not in sys.path:
        sys.path.insert(0, Deepfakedetection_root)

    from model.deepfakedetection import Deepfakedetection

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state"]
    precision  = checkpoint.get("precision", "float32")

    # Convert float16 weights to float32 for CPU compatibility
    if precision == "float16":
        state_dict = {
            k: v.float() if v.is_floating_point() else v
            for k, v in state_dict.items()
        }

    model = Deepfakedetection(pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(_device)
    model.eval()
    _model = model
    logger.info(
        f"Deepfakedetection loaded — epoch {checkpoint.get('epoch', '?')} | "
        f"val_acc={checkpoint.get('val_acc', 0):.2f}%"
    )

    # ── MTCNN face detector ───────────────────────────────────────────────────
    # Forced to CPU regardless of platform — MPS Adaptive Pool bug
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(
        image_size=112,
        margin=20,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        keep_all=False,
        device=torch.device("cpu"),
        post_process=False,
    )
    logger.info("MTCNN loaded")


def _extract_frames(video_path: str, max_frames: int = 150) -> list:
    """
    Extract up to max_frames evenly spaced frames from a video.

    Returns:
        List of RGB numpy arrays or empty list on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        logger.warning(f"Zero frames reported: {video_path}")
        return []

    num_to_extract = min(max_frames, total)
    indices = np.linspace(0, total - 1, num_to_extract, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        # OpenCV reads BGR → convert to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def _detect_face(frame_rgb: np.ndarray) -> tuple:
    """
    Run MTCNN on a single RGB frame.

    Returns:
        face_crop : (112, 112, 3) uint8 numpy array or None
        box       : [x1, y1, x2, y2] int list or None
    """
    pil_img = Image.fromarray(frame_rgb)
    boxes, probs = _mtcnn.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None, None

    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box

    # Clamp to image bounds
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    # Crop and resize with bilinear interpolation
    face_crop = frame_rgb[y1:y2, x1:x2]
    face_resized = cv2.resize(
        face_crop,
        (112, 112),
        interpolation=cv2.INTER_LINEAR
    )
    return face_resized, [int(x1), int(y1), int(x2), int(y2)]


def _frames_to_tensor(face_frames: list, sequence_length: int = 20) -> torch.Tensor:
    """
    Convert list of face frames to model input tensor.

    Returns:
        Tensor of shape (1, sequence_length, 3, 112, 112)
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

    n = len(face_frames)

    # Sample or pad to reach exactly sequence_length frames
    if n >= sequence_length:
        indices = np.linspace(0, n - 1, sequence_length, dtype=int)
        selected = [face_frames[i] for i in indices]
    else:
        # Pad by repeating last frame
        selected = face_frames.copy()
        while len(selected) < sequence_length:
            selected.append(face_frames[-1])

    tensors = [transform(f) for f in selected]   # list of (3, 112, 112)
    stacked = torch.stack(tensors, dim=0)         # (seq, 3, 112, 112)
    return stacked.unsqueeze(0)                   # (1, seq, 3, 112, 112)


def _save_face_grid(
    face_frames: list,
    boxes: list,
    raw_frames: list,
    media_root: str,
    n_display: int = 6,
) -> Optional[str]:
    """
    Save a grid image showing analyzed facial regions for display in the UI.
    Draws the detected bounding box on each frame thumbnail.

    Returns:
        Relative media path string or None on failure.
    """
    try:
        import math
        from PIL import Image as PILImage, ImageDraw

        total = len(raw_frames)
        if total == 0:
            return None

        # Pick evenly spaced frames to display
        display_indices = np.linspace(
            0, total - 1,
            min(n_display, total),
            dtype=int
        )

        thumb_w, thumb_h = 180, 180
        cols    = min(3, len(display_indices))
        rows    = math.ceil(len(display_indices) / cols)
        padding = 8
        grid_w  = cols * thumb_w + (cols + 1) * padding
        grid_h  = rows * thumb_h + (rows + 1) * padding

        grid = PILImage.new("RGB", (grid_w, grid_h), color=(20, 20, 40))

        for i, idx in enumerate(display_indices):
            frame_rgb = raw_frames[idx]
            box       = boxes[idx] if idx < len(boxes) else None

            frame_pil = PILImage.fromarray(frame_rgb)

            # Draw bounding box on detected face region
            if box is not None:
                draw = ImageDraw.Draw(frame_pil)
                draw.rectangle(box, outline=(0, 255, 120), width=3)

            thumb = frame_pil.resize((thumb_w, thumb_h), PILImage.LANCZOS)

            row = i // cols
            col = i % cols
            x   = padding + col * (thumb_w + padding)
            y   = padding + row * (thumb_h + padding)
            grid.paste(thumb, (x, y))

        # Save to media/face_grids/
        grid_dir = Path(media_root) / "face_grids"
        grid_dir.mkdir(parents=True, exist_ok=True)
        filename  = f"faces_{uuid.uuid4().hex[:8]}.jpg"
        grid_path = grid_dir / filename
        grid.save(str(grid_path), quality=85)

        return f"face_grids/{filename}"   # relative to MEDIA_ROOT

    except Exception as e:
        logger.warning(f"Face grid save failed: {e}")
        return None


def run_inference(video_path: str, media_root: str) -> dict:
    """
    Full inference pipeline for a single uploaded video.

    Args:
        video_path : absolute path to the uploaded video file
        media_root : Django MEDIA_ROOT for saving face grid image

    Returns:
        dict with keys:
            label           : "REAL" | "FAKE" | "UNCERTAIN"
            confidence      : float 0-100
            real_prob       : float 0-100
            fake_prob       : float 0-100
            frames_analyzed : int
            face_grid_path  : str or None
            error           : str or None
    """
    global _model, _mtcnn, _device

    if _model is None or _mtcnn is None:
        return {"error": "Model not loaded. Contact administrator."}

    from django.conf import settings

    try:
        # ── 1. Extract frames ─────────────────────────────────────────────────
        raw_frames = _extract_frames(video_path, max_frames=150)
        if not raw_frames:
            return {
                "error": (
                    "Could not extract frames from video. "
                    "File may be corrupted or format unsupported."
                )
            }

        # ── 2. Detect faces on each frame ─────────────────────────────────────
        face_frames   = []
        face_boxes    = []
        raw_with_face = []

        for frame in raw_frames:
            face, box = _detect_face(frame)
            face_boxes.append(box)
            if face is not None:
                face_frames.append(face)
                raw_with_face.append(frame)

        frames_analyzed = len(face_frames)

        if frames_analyzed < 5:
            return {
                "error": (
                    f"Only {frames_analyzed} frames contained a detectable face. "
                    f"Please upload a video with a clearly visible face."
                )
            }

        # ── 3. Build face grid for UI display ─────────────────────────────────
        face_grid_path = _save_face_grid(
            face_frames=face_frames,
            boxes=[b for b in face_boxes if b is not None],
            raw_frames=raw_with_face,
            media_root=media_root,
        )

        # ── 4. Convert frames to tensor ───────────────────────────────────────
        input_tensor = _frames_to_tensor(
            face_frames,
            sequence_length=settings.SEQUENCE_LENGTH
        ).to(_device)

        # ── 5. Run model inference ────────────────────────────────────────────
        with torch.no_grad():
            probs = _model(input_tensor, return_probs=True)  # (1, 2)

        real_prob = float(probs[0][0].cpu()) * 100
        fake_prob = float(probs[0][1].cpu()) * 100

        # ── 6. Determine label with uncertainty check ─────────────────────────
        threshold = settings.CONFIDENCE_THRESHOLD * 100
        max_conf  = max(real_prob, fake_prob)

        if max_conf < threshold:
            label = "UNCERTAIN"
        elif fake_prob > real_prob:
            label = "FAKE"
        else:
            label = "REAL"

        return {
            "label":           label,
            "confidence":      round(max_conf, 2),
            "real_prob":       round(real_prob, 2),
            "fake_prob":       round(fake_prob, 2),
            "frames_analyzed": frames_analyzed,
            "face_grid_path":  face_grid_path,
            "error":           None,
        }

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return {"error": f"Inference failed: {str(e)}"}

    finally:
        # ── 7. Clean up uploaded temp file after inference ────────────────────
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up temp file: {video_path}")
        except Exception as e:
            logger.warning(f"Could not delete temp file {video_path}: {e}")