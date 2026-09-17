"""
Occlusion-Aware Facial Masking and Foreground Object Detection.
Identifies hands, glasses, microphones, coffee cups, and foreign objects
crossing the face boundary, carving them out so swapped faces blend behind them.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.utils.logger import get_logger

logger = get_logger("Occlusion")


class OcclusionDetector:
    """
    Detects occluding foreground objects (hands, glasses, cups, microphones)
    inside the facial region using multi-color-space skin profiling and
    structural gradient disparity, carving out occlusions with feathered borders.
    """

    def __init__(self, sensitivity: float = 0.50, blur_radius: int = 7):
        """
        Args:
            sensitivity: Detection sensitivity [0.0, 1.0]. Higher values detect subtle occlusions.
            blur_radius: Feathering blur radius for smooth occlusion boundaries.
        """
        self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1

    def detect_occlusion_mask(
        self,
        aligned_crop: np.ndarray,
        base_mask: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes an occlusion probability matte (values in [0.0, 1.0]).
        1.0 means fully occluded by a foreground object; 0.0 means unoccluded face.
        """
        if aligned_crop is None or aligned_crop.size == 0 or base_mask is None:
            return np.zeros_like(base_mask, dtype=np.float32)

        h, w = aligned_crop.shape[:2]
        float_mask = (
            base_mask.astype(np.float32) / 255.0 if base_mask.dtype == np.uint8 else base_mask.copy()
        )

        # 1. Sample candidate skin patches at forehead and upper bridge
        lms = landmarks if landmarks is not None else INSWAPPER_STANDARD_128
        scale_x, scale_y = w / 128.0, h / 128.0
        scaled_lms = lms * np.array([scale_x, scale_y], dtype=np.float32)

        pts_left_eye = scaled_lms[0]
        pts_right_eye = scaled_lms[1]
        pts_nose = scaled_lms[2]

        eye_center = (pts_left_eye + pts_right_eye) / 2.0
        forehead_pt = eye_center - (pts_nose - eye_center) * 0.75
        forehead_left = forehead_pt + np.array([-w * 0.12, 0.0])
        forehead_right = forehead_pt + np.array([w * 0.12, 0.0])
        upper_nose = eye_center + (pts_nose - eye_center) * 0.25

        sample_pts = [forehead_pt, forehead_left, forehead_right, upper_nose]
        sample_pixels = []
        r = max(2, int(min(h, w) * 0.04))

        for pt in sample_pts:
            px, py = int(np.clip(pt[0], r, w - r - 1)), int(np.clip(pt[1], r, h - r - 1))
            patch = aligned_crop[py - r : py + r + 1, px - r : px + r + 1]
            if patch.size > 0:
                p_ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
                mean_p_y = np.mean(p_ycrcb[:, :, 0])
                if 40 <= mean_p_y <= 245:
                    sample_pixels.append(patch.reshape(-1, 3))

        if sample_pixels:
            skin_samples_bgr = np.vstack(sample_pixels)
            samples_ycrcb = cv2.cvtColor(skin_samples_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
            mean_cr = float(np.mean(samples_ycrcb[:, 1]))
            mean_cb = float(np.mean(samples_ycrcb[:, 2]))
            mean_y = float(np.mean(samples_ycrcb[:, 0]))
            std_cr = max(float(np.std(samples_ycrcb[:, 1])), 5.0)
            std_cb = max(float(np.std(samples_ycrcb[:, 2])), 5.0)
            std_y = max(float(np.std(samples_ycrcb[:, 0])), 15.0)
        else:
            mean_y, mean_cr, mean_cb = 145.0, 152.0, 110.0
            std_y, std_cr, std_cb = 25.0, 8.0, 8.0

        ycrcb = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2YCrCb)

        # Sensitivity tuning factor: higher sensitivity tightens threshold (more occlusions caught)
        thresh_multiplier = 3.2 - (self.sensitivity * 1.5)
        lum_thresh = 3.0 - (self.sensitivity * 1.5)

        # Mahalanobis-like chrominance distance from skin profile
        dist_cr = np.abs(ycrcb[:, :, 1].astype(np.float32) - mean_cr) / std_cr
        dist_cb = np.abs(ycrcb[:, :, 2].astype(np.float32) - mean_cb) / std_cb
        chroma_dist = np.sqrt(dist_cr**2 + dist_cb**2)

        lum = ycrcb[:, :, 0].astype(np.float32)
        lum_dist = np.abs(lum - mean_y) / std_y

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        bio_skin = (cr >= 125) & (cr <= 180) & (cb >= 75) & (cb <= 140) & (ycrcb[:, :, 0] >= 42)

        candidate_non_skin = (
            (chroma_dist > thresh_multiplier) |
            (lum_dist > lum_thresh) |
            (~bio_skin) |
            (ycrcb[:, :, 0] < 42)
        ) & (float_mask > 0.25)

        # 3. High-Frequency Edge Disparity for structural objects (mug rims, glasses, phones)
        gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140).astype(np.float32) / 255.0
        edge_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        structural_occlusion = candidate_non_skin | ((edge_dilated > 0.5) & (chroma_dist > 1.8) & (float_mask > 0.4))

        # 4. Morphological filtering (remove isolated salt-and-pepper noise, connect contiguous objects)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occl_uint8 = (structural_occlusion.astype(np.uint8) * 255)
        opened = cv2.morphologyEx(occl_uint8, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # 5. Smooth feathered occlusion matte
        occlusion_float = closed.astype(np.float32) / 255.0
        feathered = cv2.GaussianBlur(occlusion_float, (self.blur_radius, self.blur_radius), 0)

        return np.clip(feathered, 0.0, 1.0)

    def refine_mask_with_occlusion(
        self,
        base_mask: np.ndarray,
        aligned_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Subtracts detected occlusions from the base face mask.
        Returns the refined single-channel mask in the same dtype as base_mask.
        """
        if self.sensitivity <= 0.01:
            return base_mask

        occlusion_matte = self.detect_occlusion_mask(aligned_crop, base_mask, landmarks)

        orig_dtype = base_mask.dtype
        mask_f = base_mask.astype(np.float32) / 255.0 if orig_dtype == np.uint8 else base_mask.astype(np.float32)

        # Subtract occlusion matte from face mask
        refined = np.clip(mask_f * (1.0 - occlusion_matte), 0.0, 1.0)

        if orig_dtype == np.uint8:
            return (refined * 255.0).astype(np.uint8)
        return refined
