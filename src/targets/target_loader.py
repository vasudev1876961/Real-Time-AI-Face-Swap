"""
Target Loader for Reading and Caching Face References, Embeddings, and Metadata.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np

from src.targets.target_embedding import TargetEmbeddingExtractor
from src.utils.image_utils import read_image_safe
from src.utils.logger import get_logger

logger = get_logger("TargetLoader")


@dataclass
class TargetMetadata:
    person_id: str
    display_name: str
    category: str
    language: str = "unknown"
    region: str = "unknown"
    consent_status: str = "unknown"
    license_status: str = "unknown"
    source: str = "local"
    image_count: int = 1
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TargetFace:
    target_id: str
    display_name: str
    category: str
    reference_image_path: str
    reference_image: np.ndarray
    embedding: np.ndarray
    metadata: TargetMetadata


def load_target_from_dir(
    target_dir: str,
    category: str,
    embedding_extractor: Optional[TargetEmbeddingExtractor] = None,
    force_refresh: bool = False,
) -> Optional[TargetFace]:
    """
    Loads a single target from a directory containing reference image,
    optional metadata.json, and optional/cached face.npy.
    """
    if not os.path.isdir(target_dir):
        return None

    person_id = os.path.basename(target_dir.rstrip("\\/"))

    # 1. Discover reference image
    ref_image_path = None
    for candidate in ["reference.jpg", "reference.png", "face.jpg", "face.png", "ref.jpg"]:
        p = os.path.join(target_dir, candidate)
        if os.path.isfile(p):
            ref_image_path = p
            break

    if not ref_image_path:
        # Check any image in directory
        for f in os.listdir(target_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                ref_image_path = os.path.join(target_dir, f)
                break

    if not ref_image_path:
        logger.warning(f"Target directory {target_dir} contains no valid reference image.")
        return None

    reference_img = read_image_safe(ref_image_path)
    if reference_img is None:
        logger.error(f"Failed to read image from {ref_image_path}")
        return None

    # 2. Load or construct metadata
    meta_path = os.path.join(target_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metadata = TargetMetadata(
                person_id=data.get("person_id", person_id),
                display_name=data.get("display_name", person_id.replace("_", " ").title()),
                category=data.get("category", category),
                language=data.get("language", "unknown"),
                region=data.get("region", "unknown"),
                consent_status=data.get("consent_status", "unknown"),
                license_status=data.get("license_status", "unknown"),
                source=data.get("source", "local"),
                image_count=data.get("image_count", 1),
                notes=data.get("notes", ""),
            )
        except Exception as e:
            logger.warning(f"Error parsing {meta_path}: {e}")
            metadata = TargetMetadata(
                person_id=person_id,
                display_name=person_id.replace("_", " ").title(),
                category=category,
            )
    else:
        metadata = TargetMetadata(
            person_id=person_id,
            display_name=person_id.replace("_", " ").title(),
            category=category,
        )
        # Optionally write initial metadata template
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2)
        except Exception:
            pass

    # 3. Load or generate embedding (face.npy / embedding.npy)
    embedding = None
    if not force_refresh:
        for emb_name in ["face.npy", "embedding.npy"]:
            emb_path = os.path.join(target_dir, emb_name)
            if os.path.isfile(emb_path):
                try:
                    loaded = np.load(emb_path).astype(np.float32)
                    # Verify shape is 512
                    if loaded.size == 512:
                        embedding = loaded
                        break
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding from {emb_path}: {e}")

    if embedding is None:
        if embedding_extractor is None:
            embedding_extractor = TargetEmbeddingExtractor()
        embedding = embedding_extractor.extract_embedding(reference_img)
        # Save cache as face.npy
        cache_path = os.path.join(target_dir, "face.npy")
        try:
            np.save(cache_path, embedding)
            logger.info(f"Generated and cached embedding to {cache_path}")
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")

    return TargetFace(
        target_id=person_id,
        display_name=metadata.display_name,
        category=category,
        reference_image_path=ref_image_path,
        reference_image=reference_img,
        embedding=embedding,
        metadata=metadata,
    )


def scan_target_database(
    target_root: str = "faces",
    embedding_extractor: Optional[TargetEmbeddingExtractor] = None,
) -> Dict[str, TargetFace]:
    """
    Recursively scans the target database root for categories (actresses, actors, telugu_heroes)
    and returns a dictionary mapping target_id to TargetFace.
    """
    targets: Dict[str, TargetFace] = {}
    if not os.path.isdir(target_root):
        logger.warning(f"Target root directory does not exist: {target_root}")
        return targets

    # Iterate over category directories
    for cat_name in os.listdir(target_root):
        cat_dir = os.path.join(target_root, cat_name)
        if not os.path.isdir(cat_dir):
            continue

        # Iterate over person directories
        for person_name in os.listdir(cat_dir):
            person_dir = os.path.join(cat_dir, person_name)
            if os.path.isdir(person_dir):
                target = load_target_from_dir(person_dir, category=cat_name, embedding_extractor=embedding_extractor)
                if target:
                    targets[target.target_id] = target

    logger.info(f"Scanned target database: Found {len(targets)} targets across categories.")
    return targets
