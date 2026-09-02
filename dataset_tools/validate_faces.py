"""
Image Quality and Facial Usability Validation Tool.
Identifies blurry images, absent faces, and ambiguous multiple faces without destructive deletion.
"""

import os
import csv
import shutil
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm

from src.detection.face_detector import FaceDetector, get_face_detector
from src.utils.image_utils import read_image_safe, calculate_blur_score
from src.utils.logger import get_logger, setup_logging

logger = get_logger("FaceValidator")


@dataclass
class ValidationReport:
    total_images_processed: int = 0
    passed_images: int = 0
    blurry_images: int = 0
    no_face_images: int = 0
    multi_face_images: int = 0
    corrupted_images: int = 0
    rejections: List[Dict[str, Any]] = field(default_factory=list)


def validate_and_filter_images(
    input_dir: str = "datasets/raw",
    rejected_dir: str = "datasets/rejected",
    metadata_csv: str = "datasets/metadata/rejection_report.csv",
    blur_threshold: float = 40.0,
    dry_run: bool = True,
    detector: Optional[FaceDetector] = None,
) -> ValidationReport:
    """
    Validates images across dataset directories and documents rejections.
    """
    if detector is None:
        detector = get_face_detector()

    report = ValidationReport()
    rejection_rows = []

    os.makedirs(rejected_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)

    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_paths.append(os.path.join(root, f))

    logger.info(f"Validating {len(image_paths)} images in '{input_dir}' (dry-run: {dry_run})...")

    for file_path in tqdm(image_paths, desc="Validating Images"):
        report.total_images_processed += 1
        rel_path = os.path.relpath(file_path, input_dir)
        parts = rel_path.split(os.sep)
        category = parts[0] if len(parts) >= 2 else "unknown"
        person_id = parts[1] if len(parts) >= 3 else os.path.splitext(os.path.basename(file_path))[0]

        img = read_image_safe(file_path)
        if img is None:
            report.corrupted_images += 1
            reason = "corrupted"
            action = "flag" if dry_run else "move"
            rejection_rows.append({
                "image_path": file_path,
                "person_id": person_id,
                "category": category,
                "reason": reason,
                "blur_score": 0.0,
                "face_count": 0,
                "action": action,
            })
            if not dry_run:
                target_dest = os.path.join(rejected_dir, "corrupted", os.path.basename(file_path))
                os.makedirs(os.path.dirname(target_dest), exist_ok=True)
                shutil.move(file_path, target_dest)
            continue

        # 1. Blur Check
        blur_score = calculate_blur_score(img)
        if blur_score < blur_threshold:
            report.blurry_images += 1
            reason = "blurry"
            action = "flag" if dry_run else "move"
            rejection_rows.append({
                "image_path": file_path,
                "person_id": person_id,
                "category": category,
                "reason": reason,
                "blur_score": round(blur_score, 1),
                "face_count": 0,
                "action": action,
            })
            if not dry_run:
                target_dest = os.path.join(rejected_dir, "blurry", os.path.basename(file_path))
                os.makedirs(os.path.dirname(target_dest), exist_ok=True)
                shutil.move(file_path, target_dest)
            continue

        # 2. Face Detection Check
        faces = detector.detect(img, max_faces=5)
        face_count = len(faces)

        if face_count == 0:
            report.no_face_images += 1
            reason = "no_face"
            action = "flag" if dry_run else "move"
            rejection_rows.append({
                "image_path": file_path,
                "person_id": person_id,
                "category": category,
                "reason": reason,
                "blur_score": round(blur_score, 1),
                "face_count": face_count,
                "action": action,
            })
            if not dry_run:
                target_dest = os.path.join(rejected_dir, "no_face", os.path.basename(file_path))
                os.makedirs(os.path.dirname(target_dest), exist_ok=True)
                shutil.move(file_path, target_dest)
            continue

        # 3. Ambiguous Multiple Faces Check
        if face_count > 1:
            # If 2nd face area is > 55% of primary face area, identity is ambiguous
            primary_area = faces[0].area
            second_area = faces[1].area
            if second_area > primary_area * 0.55:
                report.multi_face_images += 1
                reason = "multiple_faces"
                action = "flag" if dry_run else "move"
                rejection_rows.append({
                    "image_path": file_path,
                    "person_id": person_id,
                    "category": category,
                    "reason": reason,
                    "blur_score": round(blur_score, 1),
                    "face_count": face_count,
                    "action": action,
                })
                if not dry_run:
                    target_dest = os.path.join(rejected_dir, "multiple_faces", os.path.basename(file_path))
                    os.makedirs(os.path.dirname(target_dest), exist_ok=True)
                    shutil.move(file_path, target_dest)
                continue

        # Passed all checks
        report.passed_images += 1

    # Save Rejection Report CSV
    report.rejections = rejection_rows
    try:
        with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image_path", "person_id", "category", "reason", "blur_score", "face_count", "action"],
            )
            writer.writeheader()
            writer.writerows(rejection_rows)
        logger.info(f"Saved validation rejection report to '{metadata_csv}'.")
    except Exception as e:
        logger.error(f"Error saving rejection CSV: {e}")

    logger.info(
        f"Validation Summary: Processed: {report.total_images_processed} -> "
        f"Passed: {report.passed_images}, "
        f"Blurry: {report.blurry_images}, "
        f"No Face: {report.no_face_images}, "
        f"Multi Face: {report.multi_face_images}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality and filter unusable images.")
    parser.add_argument("--input", "-i", type=str, default="datasets/raw", help="Raw dataset root directory")
    parser.add_argument("--rejected-dir", type=str, default="datasets/rejected", help="Directory for rejected files")
    parser.add_argument("--blur-thresh", type=float, default=40.0, help="Minimum blur score threshold")
    parser.add_argument("--apply", action="store_true", help="Execute move operations (Default is dry-run)")
    args = parser.parse_args()

    setup_logging()
    validate_and_filter_images(
        input_dir=args.input,
        rejected_dir=args.rejected_dir,
        blur_threshold=args.blur_thresh,
        dry_run=not args.apply,
    )


if __name__ == "__main__":
    main()
