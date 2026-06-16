# preprocessing/preprocess.py
"""
Step 1 - Data Preprocessing for Deepfakedetection
Pipeline per video:
  1. Extract up to 150 frames with OpenCV VideoCapture
  2. Detect faces per frame with MTCNN
  3. Discard frames with no detected face
  4. Crop detected face region with bilinear interpolation
  5. Resize to 112x112
  6. Recombine into face-only video at 30 FPS
  7. Save to data/processed/Real/ or data/processed/Fake/
"""

import os
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
MAX_FRAMES      = 150       # frames to extract per video
FACE_SIZE       = 112       # output face crop size (square)
OUTPUT_FPS      = 30        # FPS for saved face video
MIN_FACE_FRAMES = 10        # discard video if fewer valid face frames found
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset source map ────────────────────────────────────────────────────────
# Structure: { source_folder: label }
# label: "Real" or "Fake"
# Adjust these paths to match where you placed your raw videos.
SOURCE_MAP = {
     "data/final_dataset/Real":   "Real",
     "data/final_dataset/Fake":   "Fake",
    # "data/raw/DFDC/real":              "Real",
    # "data/raw/DFDC/fake":              "Fake",
    # "data/raw/FaceForensics/real":     "Real",
    # "data/raw/FaceForensics/fake":     "Fake",
    # "data/raw/CelebDF/real":           "Real",
    # "data/raw/CelebDF/fake":           "Fake",
}

OUTPUT_ROOT = Path("data/processed")


def build_mtcnn(device: str) -> MTCNN:
    """
    Initialise MTCNN for face detection.
    keep_all=False → return only the highest-confidence face per frame.
    """
    return MTCNN(
        image_size=FACE_SIZE,
        margin=20,              # padding around detected face box
        min_face_size=40,       # ignore tiny faces
        thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net confidence thresholds
        factor=0.709,
        keep_all=False,
        device=device,
        post_process=False,     # return raw pixel values, not normalised tensors
    )


def extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> list[np.ndarray]:
    """
    Extract up to max_frames evenly spaced frames from a video.

    Returns:
        List of frames as uint8 RGB numpy arrays, or [] on failure.
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

    # Evenly sample up to max_frames indices across the video
    num_to_extract = min(max_frames, total)
    frame_indices = np.linspace(0, total - 1, num_to_extract, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        # OpenCV reads BGR → convert to RGB for MTCNN / PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def detect_and_crop_face(
    frame_rgb: np.ndarray,
    mtcnn: MTCNN,
) -> Optional[np.ndarray]:
    """
    Run MTCNN on a single RGB frame.
    Returns a (112, 112, 3) uint8 numpy array, or None if no face found.
    """
    pil_img = Image.fromarray(frame_rgb)

    # detect() returns (boxes, probs); boxes is None when no face found
    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None

    # Take the highest-confidence detection (first box after MTCNN sorting)
    box = boxes[0].astype(int)          # [x1, y1, x2, y2]
    x1, y1, x2, y2 = box

    # Clamp to image bounds
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    # Crop with numpy slicing (fast, avoids PIL round-trip for the crop)
    face_crop = frame_rgb[y1:y2, x1:x2]

    # Resize to 112×112 with bilinear interpolation
    face_resized = cv2.resize(
        face_crop,
        (FACE_SIZE, FACE_SIZE),
        interpolation=cv2.INTER_LINEAR   # bilinear as specified
    )

    return face_resized  # uint8 RGB (112, 112, 3)


def save_face_video(
    face_frames: list[np.ndarray],
    output_path: str,
    fps: int = OUTPUT_FPS,
) -> bool:
    """
    Write a list of (112, 112, 3) RGB face frames to an .mp4 video.
    Returns True on success.
    """
    if not face_frames:
        return False

    output_path = str(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (FACE_SIZE, FACE_SIZE))

    if not writer.isOpened():
        logger.error(f"VideoWriter failed to open: {output_path}")
        return False

    for frame_rgb in face_frames:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    return True


def process_video(
    video_path: Path,
    label: str,
    mtcnn: MTCNN,
    output_root: Path,
    stats: dict,
) -> None:
    """
    Full pipeline for a single video.
    Saves output to output_root/label/<original_stem>.mp4
    """
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (video_path.stem + ".mp4")

    # Skip already processed videos (allows resuming after interruption)
    if output_path.exists():
        stats["skipped"] += 1
        return

    # 1. Extract frames
    frames = extract_frames(str(video_path))
    if not frames:
        logger.warning(f"[SKIP] No frames extracted: {video_path.name}")
        stats["failed"] += 1
        return

    # 2 & 3. Detect faces, discard empty frames
    face_frames = []
    for frame in frames:
        face = detect_and_crop_face(frame, mtcnn)
        if face is not None:
            face_frames.append(face)

    if len(face_frames) < MIN_FACE_FRAMES:
        logger.warning(
            f"[SKIP] Only {len(face_frames)} face frames found "
            f"(min={MIN_FACE_FRAMES}): {video_path.name}"
        )
        stats["failed"] += 1
        return

    # 4. Save face-only video
    success = save_face_video(face_frames, output_path)
    if success:
        stats["processed"] += 1
    else:
        logger.error(f"[ERROR] Could not write video: {output_path}")
        stats["failed"] += 1


def run_preprocessing(
    source_map: dict = SOURCE_MAP,
    output_root: Path = OUTPUT_ROOT,
    device: str = DEVICE,
) -> None:
    """
    Entry point. Iterates over all source folders, processes every video.
    """
    logger.info(f"Device: {device}")
    logger.info(f"Output root: {output_root}")

    mtcnn = build_mtcnn(device)

    stats = {"processed": 0, "failed": 0, "skipped": 0}

    for source_folder, label in source_map.items():
        source_path = Path(source_folder)
        if not source_path.exists():
            logger.warning(f"Source folder not found, skipping: {source_folder}")
            continue

        video_files = sorted(
            p for p in source_path.rglob("*")
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )

        logger.info(f"Found {len(video_files)} videos in {source_folder} → label={label}")

        for video_path in tqdm(video_files, desc=f"{label} | {source_path.name}"):
            process_video(video_path, label, mtcnn, output_root, stats)

    # ── Final summary ─────────────────────────────────────────────────────────
    total = stats["processed"] + stats["failed"] + stats["skipped"]
    logger.info("=" * 60)
    logger.info(f"PREPROCESSING COMPLETE")
    logger.info(f"  Total videos found : {total}")
    logger.info(f"  Processed          : {stats['processed']}")
    logger.info(f"  Failed / skipped   : {stats['failed']}")
    logger.info(f"  Already done (skip): {stats['skipped']}")

    real_count = len(list((output_root / "Real").glob("*.mp4")))
    fake_count = len(list((output_root / "Fake").glob("*.mp4")))
    logger.info(f"  Output Real/       : {real_count} videos")
    logger.info(f"  Output Fake/       : {fake_count} videos")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_preprocessing()