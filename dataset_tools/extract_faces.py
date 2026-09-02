"""
Face Extraction and Alignment Preprocessing Tool.
Crops, aligns, and standardizes faces into datasets/processed/ without training.
"""

import os
import argparse
from typing import Tuple, Optional
from tqdm import tqdm
import cv2

from src.detection.face_detector import FaceDetector, get_face_detector
from src.alignment.face_alignment import align_face_crop
from src.utils.image_utils import read_image_safe, write_image_safe
from src.utils.logger import get_logger, setup_logging

logger = get_logger("FaceExtractor")


def extract_dataset_faces(
    input_dir: str = "datasets/raw",
    output_dir: str = "datasets/processed",
    crop_size: Tuple[int, int] = (512, 512),
    detector: Optional[FaceDetector] = None,
) -> int:
    """
    Extracts, aligns, and saves normalized face crops from raw directories.
    """
    if detector is None:
        detector = get_face_detector()

    extracted_count = 0
    raw_images = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                raw_images.append(os.path.join(root, f))

    logger.info(f"Extracting faces from {len(raw_images)} images into '{output_dir}' (Crop size: {crop_size})...")

    for file_path in tqdm(raw_images, desc="Extracting Faces"):
        rel_path = os.path.relpath(file_path, input_dir)
        parts = rel_path.split(os.sep)

        category = parts[0] if len(parts) >= 2 else "actresses"
        person_id = parts[1] if len(parts) >= 3 else "unknown"
        filename = parts[-1]

        img = read_image_safe(file_path)
        if img is None:
            continue

        face = detector.detect_single(img)
        if face is None or face.landmarks is None:
            continue

        # Standard 5-point Umeyama alignment
        aligned_face, _, _ = align_face_crop(img, face.landmarks, crop_size=crop_size)

        dest_path = os.path.join(output_dir, category, person_id, filename)
        if write_image_safe(dest_path, aligned_face):
            extracted_count += 1

    logger.info(f"Extraction Complete: Successfully processed and saved {extracted_count} face crops.")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description="Extract and align faces for target datasets.")
    parser.add_argument("--input", "-i", type=str, default="datasets/raw", help="Raw dataset root directory")
    parser.add_argument("--output", "-o", type=str, default="datasets/processed", help="Processed output directory")
    parser.add_argument("--size", type=int, default=512, help="Output face resolution (e.g. 512 or 112)")
    args = parser.parse_args()

    setup_logging()
    extract_dataset_faces(
        input_dir=args.input,
        output_dir=args.output,
        crop_size=(args.size, args.size),
    )


if __name__ == "__main__":
    main()
