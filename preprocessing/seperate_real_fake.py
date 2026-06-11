# preprocessing/separate_real_fake.py
"""
Run this AFTER filter_dfdc.py.
Reads metadata.json from each DFDC part folder and copies
videos into Real/ and Fake/ folders based on their label.
"""

import os
import json
import shutil
from pathlib import Path


def separate_real_fake(dfdc_root: str, output_dir: str) -> dict:
    """
    Reads metadata.json from each DFDC part and separates
    videos into Real/ and Fake/ folders.

    Args:
        dfdc_root:  Path to your DFDC folder, e.g. 'data/raw/DFDC'
        output_dir: Path to save separated videos, e.g. 'data/separated'

    Returns:
        Summary dict with counts.
    """
    dfdc_root = Path(dfdc_root)
    real_dir = Path(output_dir) / "Real"
    fake_dir = Path(output_dir) / "Fake"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    total_real = 0
    total_fake = 0
    total_skipped = 0

    for metadata_file in sorted(dfdc_root.rglob("metadata.json")):
        part_dir = metadata_file.parent

        print(f"[Separator] Processing: {metadata_file}")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        for filename, info in metadata.items():
            video_path = part_dir / filename

            # Skip if video file does not exist
            if not video_path.exists():
                print(f"  [SKIP] Not found: {filename}")
                total_skipped += 1
                continue

            label = info.get("label", "").upper()

            if label == "REAL":
                dst = real_dir / filename
                shutil.copy2(str(video_path), str(dst))
                total_real += 1
                print(f"  [REAL] Copied: {filename}")

            elif label == "FAKE":
                dst = fake_dir / filename
                shutil.copy2(str(video_path), str(dst))
                total_fake += 1
                print(f"  [FAKE] Copied: {filename}")

            else:
                print(f"  [UNKNOWN LABEL] Skipped: {filename}")
                total_skipped += 1

    summary = {
        "real_copied": total_real,
        "fake_copied": total_fake,
        "skipped": total_skipped,
    }
    print("\n[Separator] Summary:", summary)
    return summary


if __name__ == "__main__":
    separate_real_fake(
        dfdc_root="data/raw/DFDC",
        output_dir="data/separated"
    )