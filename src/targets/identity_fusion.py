"""
Multi-Portrait Identity Embedding Fusion & Interpolation Subsystem.
Enables:
- Multi-reference identity fusion: averaging multiple photos of a person with outlier pruning
  to create an illumination-invariant, pose-robust identity embedding.
- Spherical Linear Interpolation (Slerp) between two distinct identities for seamless facial hybridization.
- Generation of fused target profiles saved directly into the target database.
"""

import os
import json
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from src.targets.target_loader import TargetFace, TargetMetadata
from src.targets.target_embedding import get_embedding_extractor
from src.utils.logger import get_logger

logger = get_logger("IdentityFusion")


class IdentityEmbeddingFuser:
    """
    Coordinates multi-sample embedding averaging, outlier rejection,
    and spherical identity interpolation.
    """

    @staticmethod
    def l2_normalize(embedding: np.ndarray) -> np.ndarray:
        """Projects a vector onto the 512-dimensional unit hypersphere."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            return embedding
        return embedding / norm

    @classmethod
    def cosine_similarity(cls, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine similarity between two normalized embedding vectors."""
        v1 = cls.l2_normalize(emb1.flatten())
        v2 = cls.l2_normalize(emb2.flatten())
        return float(np.dot(v1, v2))

    @classmethod
    def fuse_embeddings(
        cls,
        embeddings: List[np.ndarray],
        weights: Optional[List[float]] = None,
        min_similarity: float = 0.25,
    ) -> np.ndarray:
        """
        Calculates a robust weighted average embedding with outlier rejection.

        Args:
            embeddings: List of 1D (512,) float32 embedding arrays.
            weights: Optional per-sample weights.
            min_similarity: Outlier rejection threshold relative to median centroid.

        Returns:
            Normalized 512-D float32 fused embedding.
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fusion.")

        if len(embeddings) == 1:
            return cls.l2_normalize(embeddings[0].flatten().astype(np.float32))

        norm_embs = [cls.l2_normalize(e.flatten().astype(np.float32)) for e in embeddings]

        # Initial centroid for outlier filtering
        initial_mean = cls.l2_normalize(np.mean(norm_embs, axis=0))

        # Filter out anomalous or mismatched embeddings
        valid_embs = []
        valid_weights = []
        w_list = weights if (weights and len(weights) == len(norm_embs)) else [1.0] * len(norm_embs)

        for e, w in zip(norm_embs, w_list):
            sim = float(np.dot(e, initial_mean))
            if sim >= min_similarity:
                valid_embs.append(e)
                valid_weights.append(w * max(0.1, sim))
            else:
                logger.warning(f"Discarding outlier embedding with low similarity ({sim:.3f}) to centroid.")

        if not valid_embs:
            # If all rejected, fall back to unfiltered mean
            valid_embs = norm_embs
            valid_weights = w_list

        w_arr = np.array(valid_weights, dtype=np.float32)[:, np.newaxis]
        weighted_sum = np.sum(np.array(valid_embs) * w_arr, axis=0)
        return cls.l2_normalize(weighted_sum)

    @classmethod
    def interpolate_identities(
        cls,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Performs Spherical Linear Interpolation (Slerp) between two identity vectors.

        Args:
            emb_a: First 512-D identity embedding.
            emb_b: Second 512-D identity embedding.
            alpha: Interpolation factor (0.0 = 100% Identity A, 1.0 = 100% Identity B).

        Returns:
            Normalized 512-D blended identity embedding.
        """
        v0 = cls.l2_normalize(emb_a.flatten().astype(np.float32))
        v1 = cls.l2_normalize(emb_b.flatten().astype(np.float32))

        dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))

        # If vectors are virtually identical, linear interpolation is sufficient
        if dot > 0.9995:
            res = (1.0 - alpha) * v0 + alpha * v1
            return cls.l2_normalize(res)

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        if abs(sin_theta_0) < 1e-6:
            res = (1.0 - alpha) * v0 + alpha * v1
            return cls.l2_normalize(res)

        theta = theta_0 * alpha
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return cls.l2_normalize(s0 * v0 + s1 * v1)

    @classmethod
    def create_fused_target(
        cls,
        target_id: str,
        display_name: str,
        category: str,
        fused_embedding: np.ndarray,
        reference_image: np.ndarray,
        output_base_dir: str = "faces",
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> TargetFace:
        """
        Persists a fused identity target into the target database directory structure.
        """
        dest_dir = os.path.join(output_base_dir, category, target_id)
        os.makedirs(dest_dir, exist_ok=True)

        ref_path = os.path.join(dest_dir, "reference.jpg")
        cv2.imwrite(ref_path, reference_image)

        emb_path = os.path.join(dest_dir, "face.npy")
        np.save(emb_path, fused_embedding.astype(np.float32))

        meta_path = os.path.join(dest_dir, "metadata.json")
        meta = {
            "name": display_name,
            "category": category,
            "target_id": target_id,
            "is_fused": True,
            "source_details": source_metadata or {},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        meta_obj = TargetMetadata(
            person_id=target_id,
            display_name=display_name,
            category=category,
            source="fused",
            notes=str(source_metadata or ""),
        )

        return TargetFace(
            target_id=target_id,
            display_name=display_name,
            category=category,
            reference_image_path=ref_path,
            reference_image=reference_image,
            embedding=fused_embedding,
            metadata=meta_obj,
        )
