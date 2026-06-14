# dataloader/loader.py
"""
DataLoader factory for Detectra.
Call get_dataloaders() to get ready-to-use train and test loaders.
Run this file directly to smoke-test the full loading pipeline.
"""

import sys
import time
import logging
import platform
import torch
from torch.utils.data import DataLoader

from dataloader.dataset import DeepfakeDataset
from dataloader.transforms import get_train_transforms, get_val_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/dataloader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_CSV   = "data/splits/train.csv"
TEST_CSV    = "data/splits/test.csv"
BATCH_SIZE  = 4          # as specified in pipeline
NUM_WORKERS = 0          # safe default for macOS/CPU; increase to 2-4 on Linux+GPU


def get_dataloaders(
    train_csv:   str = TRAIN_CSV,
    test_csv:    str = TEST_CSV,
    batch_size:  int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple:
    """
    Build and return (train_loader, test_loader).

    Returns:
        train_loader : DataLoader with augmentation, shuffled
        test_loader  : DataLoader without augmentation, not shuffled
    """

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = DeepfakeDataset(
        csv_path=train_csv,
        transform=get_train_transforms(),
    )
    test_dataset = DeepfakeDataset(
        csv_path=test_csv,
        transform=get_val_transforms(),
    )

    # ── macOS multiprocessing safety ──────────────────────────────────────────
    # On macOS, 'spawn' is the default start method which breaks DataLoader
    # workers. Use num_workers=0 (main process only) to avoid freezes.
    mp_context = None
    if platform.system() == "Darwin" and num_workers > 0:
        logger.warning(
            "macOS detected with num_workers>0. "
            "Setting num_workers=0 to avoid multiprocessing freeze. "
            "On Linux/GPU machines you can safely set num_workers=2."
        )
        num_workers = 0

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle each epoch during training
        num_workers=num_workers,
        pin_memory=False,       # set True only when using CUDA GPU
        drop_last=True,         # drop final incomplete batch for stable training
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # never shuffle test set
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,        # evaluate on every test sample
    )

    logger.info(f"Train loader: {len(train_dataset)} samples | "
                f"{len(train_loader)} batches | batch_size={batch_size}")
    logger.info(f"Test loader:  {len(test_dataset)} samples | "
                f"{len(test_loader)} batches | batch_size={batch_size}")

    return train_loader, test_loader


def smoke_test(train_loader: DataLoader, test_loader: DataLoader) -> None:
    """
    Pull 2 batches from each loader and verify shapes, dtypes, and label values.
    Prints a full report — no assertions that crash; warnings instead.
    """
    print("\n" + "=" * 60)
    print("  STEP 3 — DATALOADER SMOKE TEST")
    print("=" * 60)

    for split_name, loader in [("TRAIN", train_loader), ("TEST", test_loader)]:
        print(f"\n── {split_name} LOADER ──────────────────────────────────")

        batch_count = 0
        t0 = time.time()

        for frames, labels in loader:
            batch_count += 1

            # ── Shape checks ──────────────────────────────────────────────────
            # frames should be (batch, sequence, C, H, W) = (4, 20, 3, 112, 112)
            expected_shape = (
                frames.shape[0],   # batch size (may be <4 for last batch)
                20,                # SEQUENCE_LENGTH
                3,                 # RGB channels
                112,               # height
                112,               # width
            )

            shape_ok    = frames.shape[1:] == torch.Size([20, 3, 112, 112])
            dtype_ok    = frames.dtype == torch.float32
            label_ok    = labels.dtype == torch.long
            label_valid = set(labels.tolist()).issubset({0, 1})

            print(f"\n  Batch {batch_count}:")
            print(f"    frames.shape  : {tuple(frames.shape)}  "
                  f"{'✓' if shape_ok else '✗ WRONG SHAPE'}")
            print(f"    frames.dtype  : {frames.dtype}  "
                  f"{'✓' if dtype_ok else '✗ EXPECTED float32'}")
            print(f"    labels        : {labels.tolist()}  "
                  f"{'✓' if label_valid else '✗ INVALID LABEL VALUES'}")
            print(f"    labels.dtype  : {labels.dtype}  "
                  f"{'✓' if label_ok else '✗ EXPECTED torch.long'}")
            print(f"    frames min/max: {frames.min():.3f} / {frames.max():.3f}  "
                  f"(expect approx -2.1 to 2.6 after ImageNet normalisation)")

            if batch_count == 2:
                break   # 2 batches is enough to confirm correctness

        elapsed = time.time() - t0
        print(f"\n  Loaded 2 batches in {elapsed:.2f}s")
        print(f"  Total batches in loader: {len(loader)}")

    print("\n" + "=" * 60)
    print("  SMOKE TEST COMPLETE — DataLoader is ready for Step 4")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logger.info("Building DataLoaders...")
    train_loader, test_loader = get_dataloaders()
    smoke_test(train_loader, test_loader)