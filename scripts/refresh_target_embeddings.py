"""
Refresh and Re-extract Authentic 512-D ArcFace Embeddings for All Target Faces.
Scans faces/ directory and generates clean, normalized face embeddings cached to face.npy.
"""

import os
import sys
import numpy as np
from tqdm import tqdm

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.targets.target_embedding import TargetEmbeddingExtractor
from src.utils.image_utils import read_image_safe


def refresh_all_embeddings(faces_root: str = "faces"):
    print("=" * 65)
    print("Refreshing Target Face Embeddings using ArcFace Face Recognition")
    print("=" * 65)

    extractor = TargetEmbeddingExtractor()
    if not extractor.is_ready():
        print("[ERROR] Face recognition model is not ready. Aborting.")
        return

    targets_updated = 0
    targets_failed = 0

    target_folders = []
    for root, dirs, files in os.walk(faces_root):
        has_img = any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files)
        if has_img:
            target_folders.append(root)

    print(f"Found {len(target_folders)} candidate target identity folders.")

    for folder in tqdm(target_folders, desc="Extracting embeddings"):
        ref_img_path = None
        for candidate in ["reference.jpg", "reference.png", "face.jpg", "face.png", "ref.jpg"]:
            p = os.path.join(folder, candidate)
            if os.path.isfile(p):
                ref_img_path = p
                break

        if not ref_img_path:
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    ref_img_path = os.path.join(folder, f)
                    break

        if not ref_img_path:
            continue

        img = read_image_safe(ref_img_path)
        if img is None:
            targets_failed += 1
            continue

        try:
            emb = extractor.extract_embedding(img)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = (emb / norm).astype(np.float32)

            out_path = os.path.join(folder, "face.npy")
            np.save(out_path, emb)
            targets_updated += 1
        except Exception as e:
            print(f"Failed on {folder}: {e}")
            targets_failed += 1

    print("\n" + "=" * 65)
    print(f"Successfully refreshed {targets_updated} target embeddings. (Failed: {targets_failed})")
    print("=" * 65)


if __name__ == "__main__":
    refresh_all_embeddings()
