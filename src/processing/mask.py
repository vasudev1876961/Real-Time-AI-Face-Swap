"""
Face Mask Generation with Morphological Operations and Edge Feathering.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.core.config_loader import ProcessingConfig
from src.utils.logger import get_logger

logger = get_logger("FaceMask")


def create_face_mask(
    crop_shape: Tuple[int, int],
    mask_type: str = "elliptical",
    blur_kernel_size: int = 15,
    feather_factor: float = 0.5,
    erosion_pixels: int = 2,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generates a feathered single-channel float32 mask [0.0, 1.0] for aligned face crop (128x128).
    """
    h, w = crop_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    # Create clean inner facial mask (stays below hairline and inside jaw/temple boundaries)
    if landmarks is not None and len(landmarks) >= 5:
        # Landmarks in crop space (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
        pts = landmarks[:5].astype(np.float32)
        eye_center = (pts[0] + pts[1]) / 2.0
        eye_dist = max(float(np.linalg.norm(pts[1] - pts[0])), 10.0)
        
        forehead = eye_center - (pts[2] - eye_center) * 0.55
        chin = (pts[3] + pts[4]) / 2.0 + ((pts[3] + pts[4]) / 2.0 - pts[2]) * 0.65
        left_temple = pts[0] + np.array([-eye_dist * 0.45, -eye_dist * 0.25], dtype=np.float32)
        right_temple = pts[1] + np.array([eye_dist * 0.45, -eye_dist * 0.25], dtype=np.float32)
        left_jaw = pts[3] + np.array([-eye_dist * 0.35, eye_dist * 0.2], dtype=np.float32)
        right_jaw = pts[4] + np.array([eye_dist * 0.35, eye_dist * 0.2], dtype=np.float32)
        
        poly = np.array([
            forehead,
            right_temple,
            right_jaw,
            chin,
            left_jaw,
            left_temple,
        ], dtype=np.int32)
        hull = cv2.convexHull(poly)
        cv2.fillConvexPoly(binary_mask, hull, 255)
    else:
        # High quality anatomical oval tailored for 128x128 INSwapper standard crop
        center = (int(w * 0.50), int(h * 0.54))
        axes = (int(w * 0.36), int(h * 0.39))
        cv2.ellipse(binary_mask, center, axes, 0, 0, 360, 255, -1)

    # Apply morphological erosion to pull boundary inwards
    if erosion_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_pixels * 2 + 1, erosion_pixels * 2 + 1))
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    # Soft Gaussian feathering
    ksize = max(5, blur_kernel_size | 1)
    sigma = max(1.0, (ksize / 3.0) * max(0.2, feather_factor))
    blurred = cv2.GaussianBlur(binary_mask.astype(np.float32), (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # Normalize to [0.0, 1.0]
    mask_normalized = np.clip(blurred / 255.0, 0.0, 1.0)
    return mask_normalized


class FaceMaskGenerator:
    """Configurable Mask Generation Engine."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()

    def generate_mask(
        self,
        crop_shape: Tuple[int, int] = (128, 128),
        landmarks: Optional[np.ndarray] = None,
        feather_override: Optional[float] = None,
    ) -> np.ndarray:
        feather = feather_override if feather_override is not None else self.config.mask_feather
        return create_face_mask(
            crop_shape=crop_shape,
            mask_type=self.config.mask_type,
            blur_kernel_size=self.config.mask_blur,
            feather_factor=feather,
            erosion_pixels=self.config.mask_erosion,
            landmarks=landmarks,
        )
