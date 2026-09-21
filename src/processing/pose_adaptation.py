"""
3D Head Pose-Adaptive Boundary Clamping & Profile Falloff Engine (Phase 9).
Eliminates 2D affine warping distortions, background pixel leakage, and stretched
synthetic textures when the subject turns their head into profile or steep tilt angles.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("PoseAdaptation")


class PoseAdaptationEngine:
    """
    Computes 3D head pose (Yaw, Pitch, Roll) and dynamically governs boundary clamping
    and graceful blend falloff at extreme profile angles.
    """

    def __init__(
        self,
        profile_threshold_deg: float = 45.0,
        max_turn_deg: float = 60.0,
        pitch_threshold_deg: float = 35.0,
    ):
        """
        Args:
            profile_threshold_deg: Yaw angle at which graceful blend falloff begins.
            max_turn_deg: Yaw angle at which swap blends completely to real face (alpha = 0).
            pitch_threshold_deg: Pitch angle above which vertical clamping is applied.
        """
        self.profile_threshold_deg = profile_threshold_deg
        self.max_turn_deg = max(max_turn_deg, profile_threshold_deg + 5.0)
        self.pitch_threshold_deg = pitch_threshold_deg

    def estimate_head_pose_3d(
        self,
        landmarks: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """
        Estimates full 3D head pose (Yaw, Pitch, Roll) in degrees from 5 standard landmarks.

        Args:
            landmarks: 5 standard facial landmarks [[x, y], ...].

        Returns:
            Tuple of (yaw_deg, pitch_deg, roll_deg).
            Yaw: Negative = turned left, Positive = turned right.
            Pitch: Negative = looking up, Positive = looking down.
            Roll: Tilt angle.
        """
        if landmarks is None or len(landmarks) < 5:
            return 0.0, 0.0, 0.0

        pts = np.asarray(landmarks[:5], dtype=np.float32)
        left_eye, right_eye, nose, left_mouth, right_mouth = pts[:5]

        # 1. Roll Angle (tilt in 2D image plane)
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        roll = float(np.degrees(np.arctan2(eye_dy, max(abs(eye_dx), 1e-4))))

        # 2. Yaw Angle (horizontal turn)
        eye_mid = (left_eye + right_eye) / 2.0
        eye_dist = max(float(np.linalg.norm(right_eye - left_eye)), 1e-4)
        nose_dx = nose[0] - eye_mid[0]
        yaw = float(np.clip((nose_dx / eye_dist) * 90.0, -90.0, 90.0))

        # 3. Pitch Angle (vertical tilt)
        nose_dy = nose[1] - eye_mid[1]
        pitch = float(np.clip(((nose_dy / eye_dist) - 0.55) * 90.0, -90.0, 90.0))

        return round(yaw, 2), round(pitch, 2), round(roll, 2)

    def compute_profile_blend_falloff(self, yaw: float, pitch: float = 0.0) -> float:
        """
        Calculates dynamic blend factor [0.0, 1.0] based on head pose.
        When head turns beyond profile_threshold_deg, the factor gracefully ramps down
        to 0.0 at max_turn_deg, seamlessly transitioning back to authentic face rather than
        displaying distorted, stretched synthetic affine textures.

        Returns:
            Pose falloff factor in [0.0, 1.0].
        """
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)

        # Yaw falloff calculation
        if abs_yaw <= self.profile_threshold_deg:
            yaw_alpha = 1.0
        elif abs_yaw >= self.max_turn_deg:
            yaw_alpha = 0.0
        else:
            # Smooth cosine ramp
            prog = (abs_yaw - self.profile_threshold_deg) / (self.max_turn_deg - self.profile_threshold_deg)
            yaw_alpha = float(0.5 * (1.0 + np.cos(np.pi * prog)))

        # Pitch falloff calculation
        if abs_pitch <= self.pitch_threshold_deg:
            pitch_alpha = 1.0
        elif abs_pitch >= self.pitch_threshold_deg + 20.0:
            pitch_alpha = 0.0
        else:
            prog = (abs_pitch - self.pitch_threshold_deg) / 20.0
            pitch_alpha = float(0.5 * (1.0 + np.cos(np.pi * prog)))

        return float(np.clip(yaw_alpha * pitch_alpha, 0.0, 1.0))

    def clamp_profile_contour(
        self,
        contour: np.ndarray,
        yaw: float,
        crop_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Contracts contour points on the turned-away cheek boundary to prevent
        the mask from stretching into ear, hair, or background when turned sideways.

        Args:
            contour: (N, 2) integer or float array of contour coordinates.
            yaw: Head yaw angle in degrees.
            crop_shape: (H, W) of crop.

        Returns:
            Contour with profile cheek boundaries clamped.
        """
        if abs(yaw) < 12.0 or len(contour) == 0:
            return contour

        h, w = crop_shape[:2]
        center_x = w * 0.50
        pts = contour.copy().astype(np.float32)

        # Normalize yaw factor [-1.0, 1.0]
        norm_yaw = float(np.clip(yaw / 45.0, -1.0, 1.0))

        if norm_yaw > 0:
            # Turned right: left cheek is turned away, pull left edge inwards
            left_mask = pts[:, 0] < center_x
            shrink_factor = abs(norm_yaw) * 0.28
            pts[left_mask, 0] += (center_x - pts[left_mask, 0]) * shrink_factor
        else:
            # Turned left: right cheek is turned away, pull right edge inwards
            right_mask = pts[:, 0] > center_x
            shrink_factor = abs(norm_yaw) * 0.28
            pts[right_mask, 0] -= (pts[right_mask, 0] - center_x) * shrink_factor

        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        return np.round(pts).astype(np.int32)

    def adapt_mask(
        self,
        mask: np.ndarray,
        yaw: float,
        pitch: float,
    ) -> np.ndarray:
        """
        Applies pose-adaptive directional feathering and erosion to the binary/soft mask.

        Args:
            mask: (H, W) float32 mask [0.0, 1.0].
            yaw: Yaw angle in degrees.
            pitch: Pitch angle in degrees.

        Returns:
            Pose-adapted (H, W) float32 mask.
        """
        if mask is None or mask.size == 0 or (abs(yaw) < 15.0 and abs(pitch) < 15.0):
            return mask

        h, w = mask.shape[:2]
        adapted = mask.copy()

        # Asymmetric erosion along turned boundary
        norm_yaw = float(np.clip(yaw / 45.0, -1.0, 1.0))
        if abs(norm_yaw) > 0.35:
            shift_px = int(abs(norm_yaw) * w * 0.05)
            if shift_px > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (shift_px * 2 + 1, 1))
                adapted_u8 = (adapted * 255).astype(np.uint8)
                eroded = cv2.erode(adapted_u8, kernel, iterations=1)
                adapted = eroded.astype(np.float32) / 255.0

        return adapted
