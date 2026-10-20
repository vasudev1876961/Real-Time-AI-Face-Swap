"""
Temporal Identity Latent Coherence Smoother (Phase 17).
Provides geodesic embedding stabilization and temporal micro-jitter suppression
to eliminate frame-to-frame high-frequency identity flicker in real-time video.
"""

from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("TemporalCoherence")


class TemporalIdentitySmoother:
    """
    Stabilizes facial identity projections across video streams by filtering
    high-frequency latent variations on the ArcFace unit hypersphere
    and dampening pixel-level crop jitter during slow head motion.
    """

    def __init__(
        self,
        default_smoothing_factor: float = 0.85,
        default_jitter_suppression: float = 0.40,
    ):
        self.smoothing_factor: float = float(np.clip(default_smoothing_factor, 0.0, 0.98))
        self.jitter_suppression: float = float(np.clip(default_jitter_suppression, 0.0, 1.0))

        # Track histories: track_id -> smoothed_embedding
        self._embedding_history: Dict[int, np.ndarray] = {}
        # Track histories: track_id -> previous_swapped_crop
        self._crop_history: Dict[int, np.ndarray] = {}
        # Track histories: track_id -> previous_landmarks
        self._landmark_history: Dict[int, np.ndarray] = {}

    def reset(self) -> None:
        """Clears temporal state across all tracks."""
        self._embedding_history.clear()
        self._crop_history.clear()
        self._crop_history.clear()
        self._landmark_history.clear()

    def set_config(
        self,
        smoothing_factor: Optional[float] = None,
        jitter_suppression: Optional[float] = None,
    ) -> None:
        """Dynamically tunes coherence filter parameters."""
        if smoothing_factor is not None:
            self.smoothing_factor = float(np.clip(smoothing_factor, 0.0, 0.98))
        if jitter_suppression is not None:
            self.jitter_suppression = float(np.clip(jitter_suppression, 0.0, 1.0))

    def smooth_embedding(
        self,
        embedding: np.ndarray,
        track_id: int = 1,
        factor: Optional[float] = None,
    ) -> np.ndarray:
        """
        Applies geodesic spherical exponential moving average to facial identity embedding.
        Guarantees the resulting vector stays strictly on the unit hypersphere (L2 norm = 1.0).
        """
        if embedding is None:
            return embedding

        alpha = self.smoothing_factor if factor is None else float(np.clip(factor, 0.0, 0.98))
        emb_flat = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb_flat)
        if norm > 1e-6:
            emb_flat = emb_flat / norm

        prev_emb = self._embedding_history.get(track_id)
        if prev_emb is None or alpha <= 0.01:
            self._embedding_history[track_id] = emb_flat
            return emb_flat

        # Spherical geodesic EMA
        smoothed = alpha * prev_emb + (1.0 - alpha) * emb_flat
        s_norm = np.linalg.norm(smoothed)
        if s_norm > 1e-6:
            smoothed = smoothed / s_norm
        else:
            smoothed = emb_flat

        self._embedding_history[track_id] = smoothed
        return smoothed

    def stabilize_crop(
        self,
        swapped_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        track_id: int = 1,
    ) -> np.ndarray:
        """
        Suppresses high-frequency pixel micro-jitter in the swapped crop.
        When head movement is slow/stationary, performs temporal integration
        to yield ultra-smooth skin textures and eliminate temporal sparkle.
        """
        if swapped_crop is None or self.jitter_suppression <= 0.01:
            return swapped_crop

        prev_crop = self._crop_history.get(track_id)
        prev_lms = self._landmark_history.get(track_id)

        # Store current landmarks for next iteration
        if landmarks is not None:
            self._landmark_history[track_id] = landmarks.copy()

        if prev_crop is None or prev_crop.shape != swapped_crop.shape:
            self._crop_history[track_id] = swapped_crop.copy()
            return swapped_crop

        # Calculate landmark velocity if available
        velocity = 0.0
        if landmarks is not None and prev_lms is not None and len(landmarks) == len(prev_lms):
            diffs = np.linalg.norm(landmarks[:, :2] - prev_lms[:, :2], axis=1)
            velocity = float(np.mean(diffs))

        # Dynamic motion-adaptive weight
        # High velocity (> 3.0 px) reduces temporal weight to prevent ghosting/motion trail
        # Low velocity (< 1.0 px) enables maximal temporal stabilization
        motion_attenuation = float(np.exp(-velocity / 2.0))
        effective_weight = self.jitter_suppression * 0.45 * motion_attenuation

        if effective_weight > 0.01:
            stabilized = cv2.addWeighted(
                prev_crop, effective_weight,
                swapped_crop, (1.0 - effective_weight),
                0.0
            )
        else:
            stabilized = swapped_crop

        self._crop_history[track_id] = stabilized.copy()
        return stabilized

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns diagnostic telemetry for temporal identity coherence."""
        return {
            "smoothing_factor": self.smoothing_factor,
            "jitter_suppression": self.jitter_suppression,
            "tracked_embeddings_count": len(self._embedding_history),
            "tracked_crops_count": len(self._crop_history),
        }
