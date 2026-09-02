"""
Dataset Metadata Generation Tool.
Generates comprehensive CSV catalogs for actresses, actors, and Telugu heroes.
"""

import os
import csv
import argparse
from typing import List, Dict, Optional, Any
from tqdm import tqdm

from src.detection.face_detector import FaceDetector, get_face_detector
from src.utils.image_utils import read_image_safe, calculate_blur_score
from src.utils.logger import get_logger, setup_logging

logger = get_logger("MetadataGenerator")

METADATA_FIELDS = [
    "person_id",
    "display_name",
    "category",
    "language",
    "region",
    "image_path",
    "face_count",
    "face_quality",
    "blur_score",
    "resolution",
    "usable",
    "consent_status",
    "license_status",
]


def generate_metadata_csvs(
    raw_dir: str = "datasets/raw",
    metadata_dir: str = "datasets/metadata",
    detector: Optional[FaceDetector] = None,
) -> Dict[str, int]:
    """
    Scans dataset categories and writes actresses.csv, actors.csv, telugu_heroes.csv.
    """
    if detector is None:
        detector = get_face_detector()

    os.makedirs(metadata_dir, exist_ok=True)
    categories = ["actresses", "actors", "telugu_heroes"]
    results_count = {}

    for cat in categories:
        cat_dir = os.path.join(raw_dir, cat)
        csv_path = os.path.join(metadata_dir, f"{cat}.csv")
        rows: List[Dict[str, Any]] = []

        if os.path.isdir(cat_dir):
            logger.info(f"Generating metadata catalog for category: '{cat}'...")
            for person_id in os.listdir(cat_dir):
                person_dir = os.path.join(cat_dir, person_id)
                if not os.path.isdir(person_dir):
                    continue

                display_name = person_id.replace("_", " ").title()
                lang = "telugu" if cat == "telugu_heroes" else "unknown"
                region = "telangana_andhra" if cat == "telugu_heroes" else "unknown"

                for f in os.listdir(person_dir):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        img_path = os.path.join(person_dir, f)
                        img = read_image_safe(img_path)

                        if img is None:
                            rows.append({
                                "person_id": person_id,
                                "display_name": display_name,
                                "category": cat,
                                "language": lang,
                                "region": region,
                                "image_path": img_path,
                                "face_count": 0,
                                "face_quality": "corrupted",
                                "blur_score": 0.0,
                                "resolution": "0x0",
                                "usable": "no",
                                "consent_status": "unknown",
                                "license_status": "unknown",
                            })
                            continue

                        h, w = img.shape[:2]
                        blur = calculate_blur_score(img)
                        faces = detector.detect(img, max_faces=3)
                        face_count = len(faces)

                        # Assess usability
                        is_usable = (face_count == 1) and (blur >= 35.0) and (min(w, h) >= 80)
                        quality = "high" if (blur >= 100 and is_usable) else ("medium" if is_usable else "low")

                        rows.append({
                            "person_id": person_id,
                            "display_name": display_name,
                            "category": cat,
                            "language": lang,
                            "region": region,
                            "image_path": img_path,
                            "face_count": face_count,
                            "face_quality": quality,
                            "blur_score": round(blur, 1),
                            "resolution": f"{w}x{h}",
                            "usable": "yes" if is_usable else "no",
                            "consent_status": "unknown",
                            "license_status": "unknown",
                        })

        # Write to CSV
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Wrote {len(rows)} entries to '{csv_path}'.")
            results_count[cat] = len(rows)
        except Exception as e:
            logger.error(f"Failed to write metadata CSV '{csv_path}': {e}")
            results_count[cat] = 0

    return results_count


def main():
    parser = argparse.ArgumentParser(description="Generate dataset metadata catalogs.")
    parser.add_argument("--raw-dir", type=str, default="datasets/raw", help="Path to raw dataset")
    parser.add_argument("--meta-dir", type=str, default="datasets/metadata", help="Path to metadata directory")
    args = parser.parse_args()

    setup_logging()
    generate_metadata_csvs(raw_dir=args.raw_dir, metadata_dir=args.meta_dir)


if __name__ == "__main__":
    main()
