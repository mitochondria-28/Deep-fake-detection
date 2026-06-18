# dataloader/dataset.py
"""
DeepfakeDataset — reads video paths and labels from a CSV produced
by Step 2, loads face frames from each .mp4, and returns fixed-length
frame sequences as tensors ready for the ResNeXt50 + LSTM pipeline.
"""
import csv
import cv2
import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 20    # fixed number of frames fed to LSTM per video
                        # (evenly sampled from however many frames the video has)
FRAME_SIZE      = 112   # must match Step 1 output (already 112×112)


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for the Detectra deepfake detection pipeline.

    Each item returned is:
        frames : FloatTensor of shape (SEQUENCE_LENGTH, 3, 112, 112)
                 normalised with ImageNet mean/std
        label  : LongTensor scalar — 0 = Real, 1 = Fake
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[transforms.Compose] = None,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        """
        Args:
            csv_path        : path to train.csv or test.csv from Step 2
            transform       : torchvision transforms applied per frame
            sequence_length : number of frames to sample per video
        """
        self.transform       = transform
        self.sequence_length = sequence_length
        self.samples         = []   # list of (path_str, label_int)

        self._load_csv(csv_path)

    def _load_csv(self, csv_path: str) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {csv_path}\n"
                f"Run preprocessing/split.py first."
            )

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = row["path"]
                label      = int(row["label"])
                if Path(video_path).exists():
                    self.samples.append((video_path, label))
                else:
                    logger.warning(f"Missing video, skipping: {video_path}")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples loaded from {csv_path}.\n"
                f"Check that processed video files exist on disk."
            )

        logger.info(
            f"Loaded {len(self.samples)} samples from {csv_path.name}  "
            f"(Real={sum(1 for _,l in self.samples if l==0)}, "
            f"Fake={sum(1 for _,l in self.samples if l==1)})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        video_path, label = self.samples[idx]

        # 1. Load all frames from the processed face video
        frames = self._load_frames(video_path)

        # 2. Sample or pad to fixed SEQUENCE_LENGTH
        frames = self._sample_frames(frames)

        # 3. Apply transforms per frame → stack into tensor
        frame_tensors = []
        for frame in frames:
            if self.transform:
                frame_tensor = self.transform(frame)   # (3, 112, 112)
            else:
                frame_tensor = torch.from_numpy(
                    frame.transpose(2, 0, 1)
                ).float() / 255.0
            frame_tensors.append(frame_tensor)

        # Stack: list of (3,112,112) → (SEQUENCE_LENGTH, 3, 112, 112)
        frames_tensor = torch.stack(frame_tensors, dim=0)
        label_tensor  = torch.tensor(label, dtype=torch.long)

        return frames_tensor, label_tensor

    def _load_frames(self, video_path: str) -> list:
        """
        Read all frames from a processed .mp4 face video.
        Returns list of (112, 112, 3) uint8 RGB numpy arrays.
        Falls back to blank frames if the video is unreadable.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return self._blank_frames(self.sequence_length)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads BGR → convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if len(frames) == 0:
            logger.warning(f"Zero frames read: {video_path}")
            return self._blank_frames(self.sequence_length)

        return frames

    def _sample_frames(self, frames: list) -> list:
        """
        Evenly sample exactly SEQUENCE_LENGTH frames from the loaded list.

        - If video has more frames than needed: evenly subsample
        - If video has fewer frames than needed: repeat-pad from the end
        """
        n = len(frames)
        target = self.sequence_length

        if n >= target:
            # Evenly spaced indices across the full video
            indices = np.linspace(0, n - 1, target, dtype=int)
            return [frames[i] for i in indices]
        else:
            # Repeat last frame to pad up to target length
            padded = frames.copy()
            while len(padded) < target:
                padded.append(frames[-1])
            return padded

    def _blank_frames(self, count: int) -> list:
        """
        Return `count` black (zero) frames as a safe fallback.
        Prevents a single corrupt video from crashing a training run.
        """
        return [np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)] * count