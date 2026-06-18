# evaluation/evaluate.py
"""
Detectra — Step 6 Evaluation Entry Point

Run:
    python3 -m evaluation.evaluate
"""

import logging
import platform
import torch
from pathlib import Path

from dataloader.loader import get_dataloaders
from evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/best_acc.pt"   # highest val accuracy checkpoint
TEST_CSV        = "data/splits/test.csv"
RESULTS_DIR     = Path("results")
BATCH_SIZE      = 4
NUM_WORKERS     = 0


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = get_device()
    logger.info(f"Device: {device}")

    # ── Test loader only — no train loader needed for evaluation ──────────────
    _, test_loader = get_dataloaders(
        test_csv=TEST_CSV,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluator = Evaluator(
        checkpoint_path=CHECKPOINT_PATH,
        test_loader=test_loader,
        device=device,
        results_dir=RESULTS_DIR,
    )
    metrics = evaluator.run()

    # ── Final pass/fail check against thesis target ───────────────────────────
    target_acc = 93.59
    actual_acc = metrics["accuracy"] * 100

    print(f"\n  Thesis target accuracy : {target_acc:.2f}%")
    print(f"  Achieved accuracy      : {actual_acc:.2f}%")

    if actual_acc >= target_acc:
        print(f"  ✓ TARGET MET — model exceeds {target_acc}% accuracy")
    else:
        gap = target_acc - actual_acc
        print(f"  ℹ {gap:.2f}% below target — expected on reduced dataset")
        print(f"    Train on full 6000-video dataset to close this gap")


if __name__ == "__main__":
    main()