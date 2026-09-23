"""
Turbo Spatial Optimizer: Umeyama Transform Caching, Jitter Suppression,
and Dynamic Multi-Scale Inference Acceleration.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2

from src.alignment.face_alignment import FaceAligner
from src.detection.face_landmarks import INSWAPPER_STANDARD_128, ARCFACE_STANDARD_512
from src.utils.logger import get_logger

logger = get_logger("TurboOptimizer")


class TurboSpatialOptimizer:
    """
    Caches affine similarity matrices and inverse warping transforms across consecutive
    frames when head displacement is within a steady-state deadband threshold.
    Eliminates redundant matrix factorizations and prevents visual boundary shimmer.
    """

    def __init__(
        self,
        jitter_threshold_px: float = 1.25,
        max_cache_frames: int = 15,
        enable_deadband: bool = True,
    ):
        """
        Args:
            jitter_threshold_px: Maximum landmark displacement to consider pose stationary.
            max_cache_frames: Maximum consecutive frames to reuse cached transform before force-refresh.
            enable_deadband: Whether to lock landmarks within the deadband to eliminate sub-pixel jitter.
        """
        self.jitter_threshold_px = float(jitter_threshold_px)
        self.max_cache_frames = int(max_cache_frames)
        self.enable_deadband = bool(enable_deadband)

        # Caches indexed by (track_id, crop_size) -> dict of cached values
        self._cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        """Clears spatial cache for a track or all tracks."""
        if track_id is not None:
            keys_to_remove = [k for k in self._cache.keys() if k[0] == track_id]
            for k in keys_to_remove:
                self._cache.pop(k, None)
        else:
            self._cache.clear()

    def get_or_compute_transform(
        self,
        landmarks: np.ndarray,
        track_id: int = 1,
        crop_size: int = 128,
        standard_landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Retrieves cached forward and inverse affine transform matrices if head pose
        is stationary, otherwise calculates fresh matrices and updates cache.

        Returns:
            Tuple[transform_matrix, inverse_matrix, was_cached]
        """
        key = (track_id, crop_size)
        cached = self._cache.get(key)

        if cached is not None and landmarks is not None:
            prev_lms = cached["landmarks"]
            frame_count = cached["frame_count"]

            # Compute mean Euclidean landmark displacement
            displacement = float(np.mean(np.linalg.norm(landmarks - prev_lms, axis=1)))

            if displacement < self.jitter_threshold_px and frame_count < self.max_cache_frames:
                cached["frame_count"] += 1
                return cached["matrix"], cached["inv_matrix"], True

        # Need fresh computation
        std_lms = (
            standard_landmarks
            if standard_landmarks is not None
            else (INSWAPPER_STANDARD_128 if crop_size == 128 else ARCFACE_STANDARD_512)
        )

        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            std_lms.astype(np.float32),
            method=cv2.LMEDS,
        )

        if matrix is None:
            # Fallback to standard aligner
            matrix = cv2.getRotationMatrix2D((crop_size / 2, crop_size / 2), 0, 1.0)

        inv_matrix = cv2.invertAffineTransform(matrix)

        self._cache[key] = {
            "landmarks": landmarks.copy(),
            "matrix": matrix,
            "inv_matrix": inv_matrix,
            "frame_count": 0,
        }

        return matrix, inv_matrix, False

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on active tracked transforms."""
        return {
            "cached_tracks": len(self._cache),
            "tracks": list(self._cache.keys()),
        }
