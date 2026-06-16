# training/train.py
"""
Deepfakedetection — Step 5 Training Entry Point

Wires together:
  - DataLoaders from Step 3
  - Model from Step 4
  - Trainer from trainer.py

Run:
    python3 -m training.train
"""

import logging
import platform
import torch
from pathlib import Path

from dataloader.loader import get_dataloaders
from model.deepfakedetection import Deepfakedetection
from training.trainer import Trainer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ── Training config ───────────────────────────────────────────────────────────
CONFIG = {
    "learning_rate":    1e-5,     # Adam LR — as specified
    "weight_decay":     1e-5,     # Adam weight decay — as specified
    "batch_size":       4,        # as specified
    "num_epochs":       20,       # adjust upward once pipeline is confirmed working
    "grad_accum_steps": 4,        # effective batch = 4 × 4 = 16; helps CPU stability
    "checkpoint_dir":   "checkpoints",
    "train_csv":        "data/splits/train.csv",
    "test_csv":         "data/splits/test.csv",
    "num_workers":      0,        # 0 = safe on macOS; set 2 on Linux+GPU
}


def get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        # Apple Silicon GPU — faster than CPU for PyTorch on M1/M2/M3
        device = torch.device("mps")
        logger.info("Apple Silicon MPS detected — using GPU acceleration")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected — training on CPU (will be slow)")
    return device


def main() -> None:
    logger.info("=" * 65)
    logger.info("  Deepfakedetection — STEP 5 TRAINING")
    logger.info("=" * 65)

    device = get_device()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logger.info("Building DataLoaders...")
    train_loader, test_loader = get_dataloaders(
        train_csv=CONFIG["train_csv"],
        test_csv=CONFIG["test_csv"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building Deepfakedetection model (pretrained=True)...")
    model = Deepfakedetection(pretrained=True)

    param_info = model.get_trainable_params()
    logger.info(f"  Trainable params : {param_info['trainable']:,} "
                f"({param_info['trainable_pct']})")
    logger.info(f"  Frozen params    : {param_info['frozen']:,}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=CONFIG,
        device=device,
        checkpoint_dir=Path(CONFIG["checkpoint_dir"]),
    )

    # ── Run training ──────────────────────────────────────────────────────────
    history = trainer.run()

    # ── Print final summary table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EPOCH-BY-EPOCH SUMMARY")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>8}")
    print("─" * 65)
    for i, (tl, ta, vl, va) in enumerate(zip(
        history["train_loss"], history["train_acc"],
        history["val_loss"],   history["val_acc"]
    ), start=1):
        # Mark epochs where a checkpoint was saved
        marker = ""
        if vl == min(history["val_loss"]):
            marker += " ← best loss"
        if va == max(history["val_acc"]):
            marker += " ← best acc"
        print(f"  {i:>5} | {tl:>10.4f} | {ta:>8.2f}% | "
              f"{vl:>8.4f} | {va:>7.2f}%{marker}")

    print("=" * 65)
    print(f"  Checkpoints saved to: {CONFIG['checkpoint_dir']}/")
    print(f"    best_loss.pt — lowest validation loss")
    print(f"    best_acc.pt  — highest validation accuracy")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()