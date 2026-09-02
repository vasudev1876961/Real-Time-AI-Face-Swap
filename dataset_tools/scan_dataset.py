"""
Dataset Scanner Tool.
Recursively inspects image files, detects unsupported formats, tiny dimensions, and corruption.
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from tqdm import tqdm
from PIL import Image

from src.utils.image_utils import read_image_safe
from src.utils.logger import get_logger, setup_logging

logger = get_logger("DatasetScanner")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ScanReport:
    total_files_scanned: int = 0
    valid_images: int = 0
    unsupported_formats: int = 0
    corrupted_images: int = 0
    too_small_images: int = 0
    discovered_categories: Dict[str, int] = field(default_factory=dict)
    discovered_persons: Dict[str, int] = field(default_factory=dict)
    details: List[Dict[str, str]] = field(default_factory=list)


def scan_raw_dataset(
    dataset_root: str = "datasets/raw",
    min_dimension: int = 64,
) -> ScanReport:
    """
    Recursively scans the dataset root directory and analyzes all images.
    """
    report = ScanReport()

    if not os.path.exists(dataset_root):
        logger.warning(f"Dataset root directory '{dataset_root}' does not exist.")
        return report

    logger.info(f"Scanning dataset in '{dataset_root}' (min dimension: {min_dimension}px)...")

    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            report.total_files_scanned += 1
            file_path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()

            if ext not in SUPPORTED_EXTENSIONS:
                report.unsupported_formats += 1
                report.details.append({
                    "path": file_path,
                    "status": "unsupported_format",
                    "reason": f"Extension '{ext}' is not supported.",
                })
                continue

            # Check if image can be safely opened and read
            img = read_image_safe(file_path)
            if img is None:
                report.corrupted_images += 1
                report.details.append({
                    "path": file_path,
                    "status": "corrupted",
                    "reason": "Image failed to decode via OpenCV/PIL.",
                })
                continue

            h, w = img.shape[:2]
            if h < min_dimension or w < min_dimension:
                report.too_small_images += 1
                report.details.append({
                    "path": file_path,
                    "status": "too_small",
                    "reason": f"Dimensions ({w}x{h}) smaller than minimum ({min_dimension}px).",
                })
                continue

            report.valid_images += 1

            # Infer category and person ID from directory structure if available
            rel_path = os.path.relpath(file_path, dataset_root)
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                cat = parts[0]
                report.discovered_categories[cat] = report.discovered_categories.get(cat, 0) + 1
            if len(parts) >= 3:
                person = parts[1]
                report.discovered_persons[person] = report.discovered_persons.get(person, 0) + 1

    logger.info(
        f"Scan Complete: Scanned {report.total_files_scanned} files -> "
        f"Valid: {report.valid_images}, "
        f"Too Small: {report.too_small_images}, "
        f"Corrupted: {report.corrupted_images}, "
        f"Unsupported: {report.unsupported_formats}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Scan and audit dataset images.")
    parser.add_argument("--input", "-i", type=str, default="datasets/raw", help="Path to raw dataset")
    parser.add_argument("--min-dim", type=int, default=64, help="Minimum dimension in pixels")
    args = parser.parse_args()

    setup_logging()
    rep = scan_raw_dataset(args.input, min_dimension=args.min_dim)
    print("\n--- DATASET AUDIT SUMMARY ---")
    print(f"Total files: {rep.total_files_scanned}")
    print(f"Valid images: {rep.valid_images}")
    print(f"Too small images: {rep.too_small_images}")
    print(f"Corrupted images: {rep.corrupted_images}")
    print(f"Unsupported format files: {rep.unsupported_formats}")
    print(f"Categories found: {list(rep.discovered_categories.keys())}")
    print(f"Total identities found: {len(rep.discovered_persons)}")


if __name__ == "__main__":
    main()
