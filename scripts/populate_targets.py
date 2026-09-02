"""
Populate Target Database with Clean Aligned Face Headshots.
Extracts the sharpest face from each person folder in datasets/raw/
and saves a 512x512 cropped headshot as reference.jpg with metadata.json.
"""

import os
import sys
import json
import shutil
import argparse
import cv2
import numpy as np
from tqdm import tqdm

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.detection.face_detector import get_face_detector
from src.alignment.face_alignment import align_face_crop
from src.utils.image_utils import read_image_safe, write_image_safe, calculate_blur_score


def populate_targets_from_raw(
    raw_root: str = "datasets/raw",
    faces_root: str = "faces",
    max_per_category: int = 50,
):
    print("=" * 60)
    print("Extracting Clean Aligned Target Face Headshots...")
    print("=" * 60)

    detector = get_face_detector()
    categories = ["telugu_heroes", "actresses", "actors"]
    total_populated = 0

    for cat in categories:
        cat_dir = os.path.join(raw_root, cat)
        if not os.path.isdir(cat_dir):
            continue

        person_folders = os.listdir(cat_dir)
        print(f"\nProcessing category '{cat}' ({len(person_folders)} identities)...")

        count = 0
        for person_id in tqdm(person_folders, desc=f"Cropping {cat}"):
            if count >= max_per_category:
                break

            person_raw_dir = os.path.join(cat_dir, person_id)
            if not os.path.isdir(person_raw_dir):
                continue

            best_crop = None
            best_score = -1.0

            # Scan images to find sharpest face
            for f in os.listdir(person_raw_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    p = os.path.join(person_raw_dir, f)
                    img = read_image_safe(p)
                    if img is None:
                        continue

                    face = detector.detect_single(img)
                    if face is not None and face.landmarks is not None:
                        # Extract 512x512 headshot
                        aligned_512, _, _ = align_face_crop(img, face.landmarks, crop_size=(512, 512))
                        blur = calculate_blur_score(aligned_512)
                        score = blur * face.score
                        if score > best_score:
                            best_score = score
                            best_crop = aligned_512

            if best_crop is not None:
                target_person_dir = os.path.join(faces_root, cat, person_id)
                os.makedirs(target_person_dir, exist_ok=True)

                # Save clean 512x512 headshot
                dst_ref = os.path.join(target_person_dir, "reference.jpg")
                write_image_safe(dst_ref, best_crop)

                # Metadata
                meta = {
                    "person_id": person_id,
                    "display_name": person_id.replace("_", " ").title(),
                    "category": cat,
                    "language": "telugu" if cat == "telugu_heroes" else "hindi/indian",
                    "region": "south_asia",
                    "source": "TFI-Faces",
                }
                meta_path = os.path.join(target_person_dir, "metadata.json")
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta, mf, indent=2)

                count += 1
                total_populated += 1

    print("\n" + "=" * 60)
    print(f"Successfully populated {total_populated} clean face headshots into '{faces_root}/'")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate clean target face headshots")
    parser.add_argument("--max", type=int, default=100, help="Maximum identities per category")
    args = parser.parse_args()
    populate_targets_from_raw(max_per_category=args.max)
