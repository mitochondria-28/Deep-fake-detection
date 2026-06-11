# preprocessing/select_1000.py
"""
Run this AFTER separate_real_fake.py.
Randomly selects 500 real and 500 fake videos from the
separated folders and copies them into the final dataset folder.
"""

import os
import random
import shutil
from pathlib import Path


def select_videos(
    separated_dir: str,
    output_dir: str,
    real_count: int = 500,
    fake_count: int = 500,
    seed: int = 42
) -> dict:
    """
    Randomly selects real_count real videos and fake_count fake videos
    and copies them into output_dir/Real and output_dir/Fake.

    Args:
        separated_dir: Path to separated videos, e.g. 'data/separated'
        output_dir:    Path to save selected videos, e.g. 'data/final_dataset'
        real_count:    Number of real videos to select (default 500)
        fake_count:    Number of fake videos to select (default 500)
        seed:          Random seed for reproducibility (default 42)

    Returns:
        Summary dict with counts.
    """
    random.seed(seed)

    real_src = Path(separated_dir) / "Real"
    fake_src = Path(separated_dir) / "Fake"
    real_dst = Path(output_dir) / "Real"
    fake_dst = Path(output_dir) / "Fake"

    real_dst.mkdir(parents=True, exist_ok=True)
    fake_dst.mkdir(parents=True, exist_ok=True)

    # Get all video files only (.mp4)
    real_videos = [
        f for f in real_src.iterdir()
        if f.suffix.lower() == ".mp4"
    ]
    fake_videos = [
        f for f in fake_src.iterdir()
        if f.suffix.lower() == ".mp4"
    ]

    print(f"[Selector] Available real videos : {len(real_videos)}")
    print(f"[Selector] Available fake videos : {len(fake_videos)}")

    # Check if enough videos exist
    if len(real_videos) < real_count:
        print(f"[WARNING] Only {len(real_videos)} real videos available."
              f" Selecting all of them.")
        real_count = len(real_videos)

    if len(fake_videos) < fake_count:
        print(f"[WARNING] Only {len(fake_videos)} fake videos available."
              f" Selecting all of them.")
        fake_count = len(fake_videos)

    # Randomly select
    selected_real = random.sample(real_videos, real_count)
    selected_fake = random.sample(fake_videos, fake_count)

    # Copy selected real videos
    print(f"\n[Selector] Copying {real_count} real videos...")
    for video in selected_real:
        dst = real_dst / video.name
        shutil.copy2(str(video), str(dst))
        print(f"  [REAL] Copied: {video.name}")

    # Copy selected fake videos
    print(f"\n[Selector] Copying {fake_count} fake videos...")
    for video in selected_fake:
        dst = fake_dst / video.name
        shutil.copy2(str(video), str(dst))
        print(f"  [FAKE] Copied: {video.name}")

    summary = {
        "real_selected": real_count,
        "fake_selected": fake_count,
        "total_selected": real_count + fake_count,
        "output_dir": str(output_dir)
    }
    print("\n[Selector] Summary:", summary)
    return summary


if __name__ == "__main__":
    select_videos(
        separated_dir="data/separated",
        output_dir="data/final_dataset",
        real_count=500,
        fake_count=500,
        seed=42
    )