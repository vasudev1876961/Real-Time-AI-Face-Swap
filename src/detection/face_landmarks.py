"""
Standard Face Landmark Coordinates, Normalization, and Convex Hull Mask Utilities.
"""

from typing import Tuple, Optional, Union
import cv2
import numpy as np

# Standard ArcFace / InsightFace 5-point landmark coordinates for 112x112 aligned face crop:
# [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
ARCFACE_STANDARD_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Standard INSwapper 128x128 landmark template (112x112 standard centered with 8px margin on all sides)
INSWAPPER_STANDARD_128 = ARCFACE_STANDARD_112 + 8.0

# Standard ArcFace coordinates scaled to 512x512
ARCFACE_STANDARD_512 = np.array(
    [
        [175.061, 236.326],
        [336.145, 235.435],
        [256.115, 327.939],
        [190.019, 422.242],
        [323.336, 421.504],
    ],
    dtype=np.float32,
)


class FaceLandmarks:
    """Helper utilities for landmark operations and geometric calculations."""

    @staticmethod
    def calculate_interocular_distance(kps: np.ndarray) -> float:
        """Calculates distance between the two eyes (kps[0] and kps[1])."""
        if kps is None or len(kps) < 2:
            return 0.0
        return float(np.linalg.norm(kps[0] - kps[1]))

    @staticmethod
    def calculate_head_pose_angles(kps: np.ndarray) -> Tuple[float, float]:
        """
        Estimates yaw and pitch angles roughly from 5 keypoints.
        Returns (approx_yaw_deg, approx_pitch_deg).
        """
        if kps is None or len(kps) < 5:
            return 0.0, 0.0

        left_eye, right_eye, nose, left_mouth, right_mouth = kps[:5]
        eye_center = (left_eye + right_eye) / 2.0
        eye_dist = max(np.linalg.norm(right_eye - left_eye), 1e-5)
        nose_offset_x = nose[0] - eye_center[0]
        yaw = float(np.clip((nose_offset_x / eye_dist) * 90.0, -90.0, 90.0))

        nose_dist_y = nose[1] - eye_center[1]
        pitch = float(np.clip(((nose_dist_y / eye_dist) - 0.55) * 90.0, -90.0, 90.0))
        return yaw, pitch

    @staticmethod
    def smooth_landmarks_ema(
        current: Optional[np.ndarray] = None,
        previous: Optional[np.ndarray] = None,
        alpha: float = 0.65,
        current_kps: Optional[np.ndarray] = None,
        prev_kps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Applies Exponential Moving Average smoothing on keypoints."""
        curr = current if current is not None else current_kps
        prev = previous if previous is not None else prev_kps
        if prev is None or curr is None:
            return curr
        return alpha * curr + (1.0 - alpha) * prev


def generate_face_mask_from_landmarks(
    arg1: Union[np.ndarray, Tuple[int, int]],
    arg2: Union[np.ndarray, Tuple[int, int]],
    mask_type: str = "convex_hull",
    blur_kernel: int = 15,
) -> np.ndarray:
    """
    Generates a single-channel 8-bit soft facial mask from 5 keypoints.
    Flexible signature: accepts (landmarks, frame_shape) or (frame_shape, landmarks).
    """
    if isinstance(arg1, tuple) or (isinstance(arg1, (list, np.ndarray)) and len(arg1) == 2 and not isinstance(arg1[0], (list, np.ndarray))):
        h, w = int(arg1[0]), int(arg1[1])
        landmarks = np.asarray(arg2, dtype=np.float32)
    else:
        landmarks = np.asarray(arg1, dtype=np.float32)
        h, w = int(arg2[0]), int(arg2[1])

    mask = np.zeros((h, w), dtype=np.uint8)

    if landmarks is None or len(landmarks) < 3:
        return mask

    pts = landmarks[:5].astype(np.int32)
    if len(pts) >= 5:
        left_eye, right_eye, nose, left_mouth, right_mouth = pts[:5]
        eye_center = (left_eye + right_eye) / 2.0
        forehead_pt = eye_center - (nose - eye_center) * 0.85
        chin_pt = (left_mouth + right_mouth) / 2.0 + ((left_mouth + right_mouth) / 2.0 - nose) * 0.85
        left_cheek = left_eye + (left_mouth - left_eye) * 0.5 - (right_eye - left_eye) * 0.35
        right_cheek = right_eye + (right_mouth - right_eye) * 0.5 + (right_eye - left_eye) * 0.35

        hull_pts = np.array(
            [
                forehead_pt,
                right_eye + [10, -10],
                right_cheek,
                right_mouth + [10, 10],
                chin_pt,
                left_mouth + [-10, 10],
                left_cheek,
                left_eye + [-10, -10],
            ],
            dtype=np.int32,
        )
    else:
        hull_pts = pts

    hull = cv2.convexHull(hull_pts)
    cv2.fillConvexPoly(mask, hull, 255)

    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

    return mask


def extract_convex_hull_mask(
    arg1: Union[np.ndarray, Tuple[int, int]],
    arg2: Union[np.ndarray, Tuple[int, int]],
    blur_kernel: int = 15,
) -> np.ndarray:
    """Wrapper supporting both (landmarks, shape) and (shape, landmarks) calls."""
    return generate_face_mask_from_landmarks(arg1, arg2, mask_type="convex_hull", blur_kernel=blur_kernel)
