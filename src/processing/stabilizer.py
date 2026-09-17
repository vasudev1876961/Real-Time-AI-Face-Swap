"""
Temporal Motion and Anti-Jitter Stabilization Subsystem.
Smooths affine geometric transformations and clamps temporal luminance jitter
to eliminate sub-pixel facial vibration and video strobing during real-time swapping.
"""

from typing import Optional, Dict, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Stabilizer")


class TrackStabilizerState:
    """Maintains running temporal geometric and photometric statistics for one face track."""

    def __init__(self):
        self.smoothed_matrix: Optional[np.ndarray] = None
        self.smoothed_inv_matrix: Optional[np.ndarray] = None
        self.smoothed_luminance: Optional[float] = None
        self.last_update_time: float = 0.0


class TemporalMotionStabilizer:
    """
    Stabilizes affine warp matrices and photometric luminance across consecutive frames.
    Suppresses micro-jitter when the subject is still while dynamically responding
    instantly to fast head turns and movements.
    """

    def __init__(
        self,
        motion_alpha: float = 0.60,
        luminance_alpha: float = 0.75,
        velocity_threshold: float = 4.0,
    ):
        """
        Args:
            motion_alpha: EMA weight for current matrix [0.1, 1.0]. Lower = smoother.
            luminance_alpha: EMA weight for luminance coherence [0.1, 1.0].
            velocity_threshold: Pixel displacement threshold above which smoothing yields to immediate movement.
        """
        self.motion_alpha = float(np.clip(motion_alpha, 0.1, 1.0))
        self.luminance_alpha = float(np.clip(luminance_alpha, 0.1, 1.0))
        self.velocity_threshold = float(velocity_threshold)
        self._states: Dict[int, TrackStabilizerState] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        """Resets stabilizer state for a specific track or all tracks."""
        if track_id is not None:
            self._states.pop(track_id, None)
        else:
            self._states.clear()

    def stabilize_transform(
        self,
        matrix: np.ndarray,
        inv_matrix: np.ndarray,
        track_id: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies velocity-adaptive exponential smoothing to forward and inverse affine matrices.
        Returns stabilized (matrix, inv_matrix).
        """
        if matrix is None or inv_matrix is None:
            return matrix, inv_matrix

        if track_id not in self._states:
            self._states[track_id] = TrackStabilizerState()

        state = self._states[track_id]

        if state.smoothed_matrix is None or state.smoothed_inv_matrix is None:
            state.smoothed_matrix = matrix.copy().astype(np.float64)
            state.smoothed_inv_matrix = inv_matrix.copy().astype(np.float64)
            return matrix, inv_matrix

        # Calculate translation velocity (Euclidean shift of origin [0, 0] under matrix)
        current_trans = matrix[:, 2]
        prev_trans = state.smoothed_matrix[:, 2]
        displacement = float(np.linalg.norm(current_trans - prev_trans))

        # Dynamically scale alpha: fast movement gets higher alpha (instant response)
        if displacement > self.velocity_threshold:
            # Shift towards 1.0 proportionally to velocity to eliminate motion drag
            adapt_factor = min(1.0, (displacement - self.velocity_threshold) / (self.velocity_threshold * 2.0))
            effective_alpha = self.motion_alpha + (1.0 - self.motion_alpha) * adapt_factor
        else:
            effective_alpha = self.motion_alpha

        # EMA blend on matrices
        smoothed_m = effective_alpha * matrix.astype(np.float64) + (1.0 - effective_alpha) * state.smoothed_matrix
        smoothed_inv = effective_alpha * inv_matrix.astype(np.float64) + (1.0 - effective_alpha) * state.smoothed_inv_matrix

        state.smoothed_matrix = smoothed_m
        state.smoothed_inv_matrix = smoothed_inv

        return smoothed_m.astype(np.float32), smoothed_inv.astype(np.float32)

    def stabilize_luminance(
        self,
        crop: np.ndarray,
        track_id: int = 1,
    ) -> np.ndarray:
        """
        Clamps sudden single-frame brightness flashes/pops to preserve photometric continuity.
        """
        if crop is None or crop.size == 0:
            return crop

        if track_id not in self._states:
            self._states[track_id] = TrackStabilizerState()

        state = self._states[track_id]

        # Convert to YCrCb to measure mean luminance
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        current_lum = float(np.mean(ycrcb[:, :, 0]))

        if state.smoothed_luminance is None:
            state.smoothed_luminance = current_lum
            return crop

        # Update smoothed luminance
        target_lum = (
            self.luminance_alpha * current_lum + (1.0 - self.luminance_alpha) * state.smoothed_luminance
        )
        state.smoothed_luminance = target_lum

        # If current frame luminance deviates by more than 8% due to camera jitter, clamp gently
        lum_ratio = target_lum / max(current_lum, 1.0)
        if abs(lum_ratio - 1.0) > 0.05:
            # Gentle adjustment factor
            adj = 1.0 + (lum_ratio - 1.0) * 0.70
            ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0].astype(np.float32) * adj, 0, 255).astype(np.uint8)
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        return crop
