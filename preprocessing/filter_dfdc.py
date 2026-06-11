# preprocessing/filter_dfdc.py
"""
Run this BEFORE preprocess.py.
Reads the DFDC metadata JSON files and moves audio-altered videos
out of the processing path so they are never included in the dataset.
"""

import os
import json
import shutil
from pathlib import Path

def filter_dfdc_audio_altered(dfdc_root: str, rejected_dir: str) -> dict:
    """
    Scans every DFDC part folder for metadata.json.
    Removes videos flagged as audio-altered.

    Args:
        dfdc_root:    Path to your DFDC folder, e.g. 'data/raw/DFDC'
        rejected_dir: Path to move rejected videos to, e.g. 'data/rejected_dfdc'

    Returns:
        Summary dict with counts.
    """
    dfdc_root = Path(dfdc_root)
    rejected_dir = Path(rejected_dir)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    total_scanned = 0
    total_rejected = 0
    total_kept = 0

    for metadata_file in sorted(dfdc_root.rglob("metadata.json")):
        part_dir = metadata_file.parent

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        for filename, info in metadata.items():
            total_scanned += 1
            video_path = part_dir / filename

            if not video_path.exists():
                continue

            # DFDC flags audio-only fakes; skip these entirely
            # The 'split' field is not relevant — we check fake type
            is_audio_fake = (
                info.get("label") == "FAKE" and
                # Audio-altered entries have a 'original' key but manipulation
                # type recorded as audio in some DFDC splits via a nested key.
                # The safest cross-version check: look for 'audio' in any string value.
                any(
                    "audio" in str(v).lower()
                    for v in info.values()
                )
            )

            if is_audio_fake:
                dest = rejected_dir / video_path.name
                shutil.move(str(video_path), str(dest))
                total_rejected += 1
            else:
                total_kept += 1

    summary = {
        "scanned": total_scanned,
        "rejected_audio_altered": total_rejected,
        "kept": total_kept,
    }
    print("[DFDC Filter] Summary:", summary)
    return summary


if __name__ == "__main__":
    filter_dfdc_audio_altered(
        dfdc_root="data/raw/DFDC",
        rejected_dir="data/rejected_dfdc"
    )