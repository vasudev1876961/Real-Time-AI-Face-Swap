"""
Near-Duplicate Detection Tool Using Perceptual Hashing (pHash).
Identifies visual duplicates without destructive deletion.
"""

import os
import csv
import shutil
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
from PIL import Image

from src.utils.image_utils import compute_phash, read_image_safe
from src.utils.logger import get_logger, setup_logging

logger = get_logger("DuplicateDetector")


def hamming_distance(h1: str, h2: str) -> int:
    """Calculates the Hamming distance between two hex hashes."""
    try:
        val1 = int(h1, 16)
        val2 = int(h2, 16)
        xor_val = val1 ^ val2
        return bin(xor_val).count("1")
    except Exception:
        return 999


def detect_and_flag_duplicates(
    input_dir: str = "datasets/raw",
    rejected_dir: str = "datasets/rejected/duplicates",
    report_csv: str = "datasets/metadata/duplicate_report.csv",
    distance_threshold: int = 4,
    dry_run: bool = True,
) -> List[Dict[str, str]]:
    """
    Scans directory for near-identical images and generates duplicate report.
    """
    os.makedirs(rejected_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_csv), exist_ok=True)

    # 1. Compute pHash for all images
    logger.info(f"Computing perceptual hashes for images in '{input_dir}'...")
    hashes: Dict[str, str] = {}
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                fp = os.path.join(root, f)
                img = read_image_safe(fp)
                if img is not None:
                    h = compute_phash(img)
                    if h:
                        hashes[fp] = h

    logger.info(f"Computed {len(hashes)} perceptual hashes. Comparing pairwise...")

    # 2. Pairwise comparison
    duplicates_found: List[Dict[str, str]] = []
    processed_paths = list(hashes.keys())
    seen_duplicates = set()

    for i in range(len(processed_paths)):
        path_a = processed_paths[i]
        if path_a in seen_duplicates:
            continue
        hash_a = hashes[path_a]

        for j in range(i + 1, len(processed_paths)):
            path_b = processed_paths[j]
            if path_b in seen_duplicates:
                continue
            hash_b = hashes[path_b]

            dist = hamming_distance(hash_a, hash_b)
            if dist <= distance_threshold:
                # Flag path_b as duplicate of path_a
                seen_duplicates.add(path_b)
                similarity_pct = round((1.0 - (dist / 64.0)) * 100.0, 1)
                action = "flag" if dry_run else "move"
                duplicates_found.append({
                    "original": path_a,
                    "duplicate": path_b,
                    "similarity": f"{similarity_pct}%",
                    "action": action,
                })

                if not dry_run:
                    dest = os.path.join(rejected_dir, os.path.basename(path_b))
                    shutil.move(path_b, dest)

    # 3. Write Duplicate Report CSV
    try:
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["original", "duplicate", "similarity", "action"])
            writer.writeheader()
            writer.writerows(duplicates_found)
        logger.info(f"Saved duplicate report to '{report_csv}'.")
    except Exception as e:
        logger.error(f"Error writing duplicate CSV: {e}")

    logger.info(f"Duplicate Detection Complete: Identified {len(duplicates_found)} duplicate pairs.")
    return duplicates_found


def main():
    parser = argparse.ArgumentParser(description="Find and flag near-duplicate images using pHash.")
    parser.add_argument("--input", "-i", type=str, default="datasets/raw", help="Directory to scan")
    parser.add_argument("--rejected-dir", type=str, default="datasets/rejected/duplicates", help="Rejected directory")
    parser.add_argument("--thresh", type=int, default=4, help="Maximum Hamming distance threshold (default 4)")
    parser.add_argument("--apply", action="store_true", help="Move duplicates (Default is dry-run)")
    args = parser.parse_args()

    setup_logging()
    detect_and_flag_duplicates(
        input_dir=args.input,
        rejected_dir=args.rejected_dir,
        distance_threshold=args.thresh,
        dry_run=not args.apply,
    )


if __name__ == "__main__":
    main()
