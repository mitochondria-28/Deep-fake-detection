# preprocessing/verify_split.py
"""
Reads the saved train.csv and test.csv and prints a full
verification report. Run this after split.py to confirm
everything looks correct before moving to Step 3.
"""

import csv
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SPLITS_DIR = Path("data/splits")


def load_csv(filepath: Path) -> tuple[list[str], list[int]]:
    paths, labels = [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["path"])
            labels.append(int(row["label"]))
    return paths, labels


def report(name: str, paths: list[str], labels: list[int]) -> None:
    counts   = Counter(labels)
    n_total  = len(labels)
    n_real   = counts[0]
    n_fake   = counts[1]

    # Check all files actually exist on disk
    missing = [p for p in paths if not Path(p).exists()]

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Total videos : {n_total}")
    print(f"  Real (0)     : {n_real}  ({n_real/n_total:.1%})")
    print(f"  Fake (1)     : {n_fake}  ({n_fake/n_total:.1%})")
    print(f"  Missing files: {len(missing)}")

    if missing:
        print("  ⚠  Missing paths (first 5):")
        for p in missing[:5]:
            print(f"     {p}")
    else:
        print("  ✓ All files exist on disk")

    # Overlap check (called from main with both sets)
    return set(paths)


def check_overlap(train_paths: set, test_paths: set) -> None:
    overlap = train_paths & test_paths
    print(f"\n{'─'*50}")
    print(f"  Overlap check")
    print(f"{'─'*50}")
    if overlap:
        print(f"  ✗ WARNING: {len(overlap)} videos appear in BOTH splits!")
        for p in list(overlap)[:5]:
            print(f"    {p}")
    else:
        print(f"  ✓ No overlap between train and test sets")


def run_verification() -> None:
    print("\n" + "=" * 50)
    print("  STEP 2 — SPLIT VERIFICATION REPORT")
    print("=" * 50)

    train_csv = SPLITS_DIR / "train.csv"
    test_csv  = SPLITS_DIR / "test.csv"

    for csv_file in [train_csv, test_csv]:
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {csv_file}\n"
                f"Run split.py first."
            )

    train_paths, train_labels = load_csv(train_csv)
    test_paths,  test_labels  = load_csv(test_csv)

    train_set = report("TRAIN SPLIT  (train.csv)", train_paths, train_labels)
    test_set  = report("TEST SPLIT   (test.csv)",  test_paths,  test_labels)
    check_overlap(train_set, test_set)

    total = len(train_paths) + len(test_paths)
    print(f"\n{'─'*50}")
    print(f"  TOTALS")
    print(f"{'─'*50}")
    print(f"  Train + Test : {len(train_paths)} + {len(test_paths)} = {total}")
    actual_train_pct = len(train_paths) / total
    actual_test_pct  = len(test_paths)  / total
    print(f"  Train ratio  : {actual_train_pct:.1%}  (target 70.0%)")
    print(f"  Test ratio   : {actual_test_pct:.1%}   (target 30.0%)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_verification()