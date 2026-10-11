"""
Directional Motion Blur Harmonization Engine (Phase 16).
Tracks head velocity trajectory across frames and synthesizes directional
shutter motion blur kernels on swapped facial crops to eliminate floating sticker jitter.
"""

from typing import Optional, Tuple, Dict, Any
import math
import time
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("MotionBlur")


class MotionBlurHarmonizer:
    """
    Solves the discrepancy between high-speed shutter camera movement blur and artificially
    crisp neural face swap crops. Synthesizes a trajectory-aligned directional convolution
    blur kernel proportional to facial velocity vector (vx, vy).
    """

    def __init__(
        self,
        default_threshold: float = 2.5,
        default_sensitivity: float = 0.50,
        max_kernel_size: int = 15,
        velocity_inertia: float = 0.60,
    ):
        """
        Args:
            default_threshold: Velocity magnitude threshold in pixels/frame before blur activates [0.5, 10.0].
            default_sensitivity: Scale factor mapping excess velocity to kernel length [0.1, 1.0].
            max_kernel_size: Maximum directional kernel diameter (must be odd).
            velocity_inertia: Temporal momentum smoothing factor [0.0, 0.95].
        """
        self.default_threshold = float(max(0.5, default_threshold))
        self.default_sensitivity = float(np.clip(default_sensitivity, 0.1, 1.0))
        self.max_kernel_size = max(3, int(max_kernel_size) | 1)
        self.velocity_inertia = float(np.clip(velocity_inertia, 0.0, 0.95))

        self._track_states: Dict[int, Dict[str, Any]] = {}
        self._last_metrics: Dict[str, Any] = {
            "velocity": 0.0,
            "angle_deg": 0.0,
            "blur_applied": False,
            "kernel_length": 0.0,
        }

    def track_motion(
        self,
        landmarks: Optional[np.ndarray] = None,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        track_id: int = 1,
    ) -> Dict[str, float]:
        """
        Tracks facial centroid displacement and computes smoothed velocity vector (vx, vy).

        Returns:
            Dict containing:
                vx: Horizontal velocity (px/frame)
                vy: Vertical velocity (px/frame)
                velocity: Euclidean velocity magnitude
                angle_rad: Trajectory angle in radians
                angle_deg: Trajectory angle in degrees
        """
        # Determine current centroid
        if landmarks is not None and len(landmarks) >= 5:
            pts = np.asarray(landmarks, dtype=np.float32)
            curr_x = float(np.mean(pts[:, 0]))
            curr_y = float(np.mean(pts[:, 1]))
        elif face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            curr_x = float(x1 + x2) / 2.0
            curr_y = float(y1 + y2) / 2.0
        else:
            return {
                "vx": 0.0,
                "vy": 0.0,
                "velocity": 0.0,
                "angle_rad": 0.0,
                "angle_deg": 0.0,
            }

        prev = self._track_states.get(track_id)
        if prev is None:
            self._track_states[track_id] = {
                "x": curr_x,
                "y": curr_y,
                "vx": 0.0,
                "vy": 0.0,
            }
            return {
                "vx": 0.0,
                "vy": 0.0,
                "velocity": 0.0,
                "angle_rad": 0.0,
                "angle_deg": 0.0,
            }

        # Raw displacement
        raw_vx = curr_x - prev["x"]
        raw_vy = curr_y - prev["y"]

        # Inertia filter
        vx = (1.0 - self.velocity_inertia) * raw_vx + self.velocity_inertia * prev["vx"]
        vy = (1.0 - self.velocity_inertia) * raw_vy + self.velocity_inertia * prev["vy"]

        speed = math.sqrt(vx * vx + vy * vy)
        angle_rad = math.atan2(vy, vx) if speed > 1e-4 else 0.0
        angle_deg = math.degrees(angle_rad)

        # Update state
        self._track_states[track_id] = {
            "x": curr_x,
            "y": curr_y,
            "vx": vx,
            "vy": vy,
        }

        return {
            "vx": vx,
            "vy": vy,
            "velocity": speed,
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
        }

    def create_directional_kernel(
        self,
        length: float,
        angle_rad: float,
        max_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Constructs a normalized 2D linear directional blur convolution kernel.

        Args:
            length: Effective length of blur streak in pixels.
            angle_rad: Angle of streak in radians.
            max_size: Maximum kernel dimension.

        Returns:
            Normalized 2D float32 kernel summing to 1.0.
        """
        limit = self.max_kernel_size if max_size is None else max(3, int(max_size) | 1)
        ksize = max(3, min(limit, int(math.ceil(length * 1.5)) | 1))

        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2

        # Half-streak coordinates along angle
        half_len = length / 2.0
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        pt1 = (int(round(center - half_len * cos_a)), int(round(center - half_len * sin_a)))
        pt2 = (int(round(center + half_len * cos_a)), int(round(center + half_len * sin_a)))

        # Draw anti-aliased line
        cv2.line(kernel, pt1, pt2, 1.0, thickness=1, lineType=cv2.LINE_AA)

        k_sum = float(np.sum(kernel))
        if k_sum > 1e-5:
            kernel /= k_sum
        else:
            kernel[center, center] = 1.0

        return kernel

    def harmonize_motion_blur(
        self,
        swapped_crop: np.ndarray,
        track_id: int = 1,
        landmarks: Optional[np.ndarray] = None,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        threshold: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Dynamically applies shutter motion blur to the swapped face crop if movement
        exceeds threshold, seamlessly harmonizing it with the moving background.

        Args:
            swapped_crop: Neural swapped face crop (H, W, 3).
            track_id: Face tracking ID.
            landmarks: Optional face landmarks.
            face_bbox: Optional face bounding box.
            threshold: Speed threshold.
            sensitivity: Motion-to-blur multiplier.

        Returns:
            Tuple[harmonized_crop, telemetry_metrics].
        """
        if swapped_crop is None or swapped_crop.size == 0:
            return swapped_crop, self._last_metrics

        thresh = self.default_threshold if threshold is None else float(max(0.1, threshold))
        sens = self.default_sensitivity if sensitivity is None else float(np.clip(sensitivity, 0.05, 1.0))

        # Track motion
        motion = self.track_motion(landmarks=landmarks, face_bbox=face_bbox, track_id=track_id)
        speed = motion["velocity"]
        angle_rad = motion["angle_rad"]

        if speed < thresh:
            self._last_metrics = {
                "velocity": round(speed, 2),
                "angle_deg": round(motion["angle_deg"], 1),
                "blur_applied": False,
                "kernel_length": 0.0,
            }
            return swapped_crop.copy(), self._last_metrics

        # Compute blur kernel length proportional to excess speed
        excess_speed = speed - thresh
        blur_len = min(float(self.max_kernel_size), excess_speed * sens * 1.8)

        if blur_len < 1.0:
            self._last_metrics = {
                "velocity": round(speed, 2),
                "angle_deg": round(motion["angle_deg"], 1),
                "blur_applied": False,
                "kernel_length": 0.0,
            }
            return swapped_crop.copy(), self._last_metrics

        # Synthesize kernel and convolve
        kernel = self.create_directional_kernel(blur_len, angle_rad)
        blurred = cv2.filter2D(swapped_crop, -1, kernel, borderType=cv2.BORDER_REPLICATE)

        # Smooth cross-fade based on excess speed to eliminate popping
        blend_factor = min(1.0, excess_speed / (thresh * 2.0))
        output = cv2.addWeighted(blurred, blend_factor, swapped_crop, 1.0 - blend_factor, 0)

        self._last_metrics = {
            "velocity": round(speed, 2),
            "angle_deg": round(motion["angle_deg"], 1),
            "blur_applied": True,
            "kernel_length": round(blur_len, 2),
        }

        return output, self._last_metrics

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns consolidated motion blur telemetry."""
        return dict(self._last_metrics)
