# preprocessing/split.py  (updated — handles imbalanced source data)
"""
Step 2 - Data Splitting for Detectra
- Scans data/processed/Real/ and data/processed/Fake/
- Balances dataset by trimming majority class to minority class size
- Splits 70% train / 30% test using stratified shuffle split
- Verifies 50/50 class balance is maintained in both splits
- Saves train.csv and test.csv to data/splits/
"""

import csv
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/splitting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_ROOT = Path("data/processed")
SPLITS_DIR     = Path("data/splits")
TRAIN_RATIO    = 0.70
RANDOM_STATE   = 42
LABEL_MAP      = {"Real": 0, "Fake": 1}


def collect_samples(processed_root: Path) -> tuple:
    """
    Walk Real/ and Fake/ folders, collect all .mp4 paths and their labels.
    Returns: (paths, labels)
    """
    paths, labels = [], []

    for folder_name, label in LABEL_MAP.items():
        folder = processed_root / folder_name
        if not folder.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {folder}\n"
                f"Make sure Step 1 preprocessing completed successfully."
            )

        video_files = sorted(folder.glob("*.mp4"))
        if len(video_files) == 0:
            raise ValueError(f"No .mp4 files found in {folder}")

        for vf in video_files:
            paths.append(str(vf.resolve()))
            labels.append(label)

        logger.info(f"  {folder_name}/ → {len(video_files)} videos (label={label})")

    return paths, labels


def balance_dataset(
    paths: list,
    labels: list,
    random_state: int = RANDOM_STATE
) -> tuple:
    """
    Trim the majority class down to the size of the minority class.
    Uses a fixed random seed so trimming is reproducible.

    Returns: balanced (paths, labels)
    """
    labels_arr = np.array(labels)
    n_real = int((labels_arr == 0).sum())
    n_fake = int((labels_arr == 1).sum())
    minority_count = min(n_real, n_fake)

    logger.info(f"  Before balancing → Real: {n_real} | Fake: {n_fake}")
    logger.info(f"  Trimming majority class to {minority_count} samples...")

    real_paths = [p for p, l in zip(paths, labels) if l == 0]
    fake_paths = [p for p, l in zip(paths, labels) if l == 1]

    random.seed(random_state)

    # Trim whichever class is larger
    if n_real > minority_count:
        real_paths = random.sample(real_paths, minority_count)
    if n_fake > minority_count:
        fake_paths = random.sample(fake_paths, minority_count)

    balanced_paths  = real_paths + fake_paths
    balanced_labels = [0] * len(real_paths) + [1] * len(fake_paths)

    logger.info(f"  After balancing  → Real: {len(real_paths)} | Fake: {len(fake_paths)}")
    logger.info(f"  Total balanced dataset size: {len(balanced_paths)}")

    return balanced_paths, balanced_labels


def verify_balance(name: str, labels: list, tolerance: float = 0.05) -> None:
    labels_arr = np.array(labels)
    n_total = len(labels_arr)
    n_real  = int((labels_arr == 0).sum())
    n_fake  = int((labels_arr == 1).sum())
    real_ratio = n_real / n_total
    fake_ratio = n_fake / n_total

    logger.info(
        f"  {name}: {n_total} total | "
        f"Real={n_real} ({real_ratio:.1%}) | "
        f"Fake={n_fake} ({fake_ratio:.1%})"
    )

    assert abs(real_ratio - 0.5) <= tolerance, (
        f"[{name}] Real ratio {real_ratio:.1%} exceeds ±{tolerance:.0%} tolerance."
    )
    logger.info(f"  ✓ {name} balance check passed")


def save_split_csv(filepath: Path, paths: list, labels: list) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    label_name_map = {0: "Real", 1: "Fake"}
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "label_name"])
        for p, l in zip(paths, labels):
            writer.writerow([p, l, label_name_map[l]])
    logger.info(f"  Saved → {filepath}  ({len(paths)} rows)")


def run_splitting(
    processed_root: Path = PROCESSED_ROOT,
    splits_dir: Path     = SPLITS_DIR,
    train_ratio: float   = TRAIN_RATIO,
    random_state: int    = RANDOM_STATE,
) -> dict:

    logger.info("=" * 60)
    logger.info("STEP 2 — DATA SPLITTING")
    logger.info("=" * 60)

    # ── 1. Collect all samples ────────────────────────────────────────────────
    logger.info("Collecting samples from processed folders...")
    all_paths, all_labels = collect_samples(processed_root)
    logger.info(f"Total samples collected: {len(all_paths)}")

    # ── 2. Report raw class distribution ─────────────────────────────────────
    labels_arr = np.array(all_labels)
    n_real_raw = int((labels_arr == 0).sum())
    n_fake_raw = int((labels_arr == 1).sum())
    logger.info(f"Raw distribution → Real: {n_real_raw} | Fake: {n_fake_raw}")

    # ── 3. Balance dataset (trim majority to minority) ────────────────────────
    logger.info("Balancing dataset...")
    all_paths, all_labels = balance_dataset(all_paths, all_labels, random_state)

    # ── 4. Verify balance after trimming ─────────────────────────────────────
    logger.info("Verifying balanced dataset...")
    verify_balance("Full dataset (balanced)", all_labels)

    # ── 5. Stratified train/test split ────────────────────────────────────────
    logger.info(
        f"Splitting {train_ratio:.0%} train / {1-train_ratio:.0%} test "
        f"(random_state={random_state})..."
    )
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=1.0 - train_ratio,
        stratify=all_labels,
        random_state=random_state,
        shuffle=True,
    )

    # ── 6. Verify balance in each split ───────────────────────────────────────
    logger.info("Verifying split balances...")
    verify_balance("Train split", train_labels)
    verify_balance("Test split",  test_labels)

    # ── 7. Save CSVs ──────────────────────────────────────────────────────────
    logger.info("Saving split manifests...")
    train_csv = splits_dir / "train.csv"
    test_csv  = splits_dir / "test.csv"
    save_split_csv(train_csv, train_paths, train_labels)
    save_split_csv(test_csv,  test_paths,  test_labels)

    # ── 8. Final summary ──────────────────────────────────────────────────────
    total_balanced = len(all_paths)
    logger.info("=" * 60)
    logger.info("SPLITTING COMPLETE")
    logger.info(f"  Raw total      : {n_real_raw + n_fake_raw}  "
                f"(Real={n_real_raw}, Fake={n_fake_raw})")
    logger.info(f"  After balance  : {total_balanced}  "
                f"(Real={total_balanced//2}, Fake={total_balanced//2})")
    logger.info(f"  Train          : {len(train_paths)}  "
                f"(Real={train_labels.count(0)}, Fake={train_labels.count(1)})")
    logger.info(f"  Test           : {len(test_paths)}  "
                f"(Real={test_labels.count(0)}, Fake={test_labels.count(1)})")
    logger.info(f"  CSVs saved to  : {splits_dir}/")
    logger.info("=" * 60)

    return {
        "total_raw":      n_real_raw + n_fake_raw,
        "total_balanced": total_balanced,
        "train_size":     len(train_paths),
        "test_size":      len(test_paths),
        "train_csv":      str(train_csv),
        "test_csv":       str(test_csv),
    }


if __name__ == "__main__":
    run_splitting()