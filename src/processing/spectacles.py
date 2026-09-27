"""
Spectacles, Eyewear & Lens Glare Preservation Engine.
Identifies physical eyeglass frames (metal wire, dark acetate rims, bridge bars)
and specular lens reflections crossing the facial plane, preserving them so authentic
glasses sit cleanly on top of synthetic swapped faces.
"""

from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.utils.logger import get_logger

logger = get_logger("SpectaclesPreservation")


class SpectaclesPreservationEngine:
    """
    Detects and preserves physical spectacles, bridge bars, orbital frame rims,
    and lens specular glints to prevent facial swappers from painting over glasses.
    """

    def __init__(
        self,
        default_strength: float = 0.75,
        temporal_alpha: float = 0.70,
    ):
        """
        Args:
            default_strength: Opacity/strength of spectacles preservation [0.0, 1.0].
            temporal_alpha: EMA smoothing factor to suppress frame-to-frame shimmer.
        """
        self.default_strength = float(np.clip(default_strength, 0.0, 1.0))
        self.temporal_alpha = float(np.clip(temporal_alpha, 0.1, 1.0))
        self._history: Dict[int, np.ndarray] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        """Resets temporal smoothing history."""
        if track_id is not None:
            self._history.pop(track_id, None)
        else:
            self._history.clear()

    def detect_spectacles_mask(
        self,
        aligned_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        strength: Optional[float] = None,
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates a single-channel floating-point matte [0.0, 1.0] of detected
        spectacles (frames, bridge, and specular lens reflections).
        1.0 means full spectacles preservation (carve out of face swap).
        """
        if aligned_crop is None or aligned_crop.size == 0:
            return np.zeros((128, 128), dtype=np.float32)

        h, w = aligned_crop.shape[:2]
        effective_strength = self.default_strength if strength is None else float(np.clip(strength, 0.0, 1.0))

        if effective_strength <= 0.01:
            return np.zeros((h, w), dtype=np.float32)

        lms = landmarks if landmarks is not None else INSWAPPER_STANDARD_128
        scale_x, scale_y = w / 128.0, h / 128.0
        scaled_lms = lms * np.array([scale_x, scale_y], dtype=np.float32)

        pts_left_eye = scaled_lms[0]
        pts_right_eye = scaled_lms[1]
        pts_nose = scaled_lms[2]

        iod = max(float(np.linalg.norm(pts_right_eye - pts_left_eye)), 10.0)
        eye_center = (pts_left_eye + pts_right_eye) / 2.0

        # Define Region of Interest (ROI) for spectacles:
        # Encloses left temple, left orbital, nasal bridge, right orbital, and right temple
        spec_roi_mask = np.zeros((h, w), dtype=np.uint8)

        r_x = int(iod * 0.44)
        r_y = int(iod * 0.32)

        # Left lens ellipse
        cv2.ellipse(
            spec_roi_mask,
            (int(pts_left_eye[0]), int(pts_left_eye[1])),
            (r_x, r_y),
            0, 0, 360, 255, -1
        )
        # Right lens ellipse
        cv2.ellipse(
            spec_roi_mask,
            (int(pts_right_eye[0]), int(pts_right_eye[1])),
            (r_x, r_y),
            0, 0, 360, 255, -1
        )
        # Nasal bridge corridor connecting the two lenses
        bridge_pts = np.array([
            pts_left_eye + np.array([0, -r_y * 0.35], dtype=np.float32),
            pts_right_eye + np.array([0, -r_y * 0.35], dtype=np.float32),
            pts_right_eye + np.array([0, r_y * 0.25], dtype=np.float32),
            pts_left_eye + np.array([0, r_y * 0.25], dtype=np.float32),
        ], dtype=np.int32)
        cv2.fillPoly(spec_roi_mask, [bridge_pts], 255)

        # 1. Structural Edge & High-Contrast Ridge Detection (Frame Rims & Bridge Bar)
        gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Morphological gradient reveals thin dark/light boundaries of frames
        kernel_ridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(blurred_gray, cv2.MORPH_GRADIENT, kernel_ridge)

        # Frame boundaries have strong gradients within the ROI
        frame_edges = cv2.Canny(blurred_gray, 40, 110).astype(np.float32) / 255.0

        # 2. Nasal Bridge Structure: High-contrast bar between eye corners
        inner_left_x = int(pts_left_eye[0] + iod * 0.15)
        inner_right_x = int(pts_right_eye[0] - iod * 0.15)
        bridge_y_top = int(max(0, eye_center[1] - iod * 0.22))
        bridge_y_bot = int(min(h, eye_center[1] + iod * 0.15))

        bridge_structure = np.zeros((h, w), dtype=np.float32)
        if inner_right_x > inner_left_x and bridge_y_bot > bridge_y_top:
            bridge_patch = gray[bridge_y_top:bridge_y_bot, inner_left_x:inner_right_x]
            if bridge_patch.size > 0:
                p_mean = float(np.mean(bridge_patch))
                p_std = float(np.std(bridge_patch))
                # If there is a distinct bridge bar, local variance is elevated or mean is dark
                if p_std > 12.0 or p_mean < 75.0:
                    sobel_h = np.abs(cv2.Sobel(bridge_patch, cv2.CV_32F, 0, 1, ksize=3))
                    bridge_structure[bridge_y_top:bridge_y_bot, inner_left_x:inner_right_x] = np.clip(
                        sobel_h / 255.0 * 2.0, 0.0, 1.0
                    )

        # 3. Specular Reflection & Lens Glare Extraction
        hsv = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        # Glare on glass lenses: High brightness (Val > 215) and Low saturation (Sat < 45)
        specular_glare = ((val > 215) & (sat < 45)).astype(np.float32)

        # Dark frames (acetate / black rims): Val < 45 within spectacles ROI
        dark_frames = ((val < 45) & (spec_roi_mask > 0)).astype(np.float32)

        # 4. Synthesize Spectacles Mask
        raw_spectacles = (
            frame_edges * 0.70 +
            bridge_structure * 0.85 +
            dark_frames * 0.90 +
            specular_glare * 0.65
        ) * (spec_roi_mask.astype(np.float32) / 255.0)

        raw_spectacles = np.clip(raw_spectacles, 0.0, 1.0)

        # Filter out minor noise with gentle threshold and dilation
        active_spectacles = (raw_spectacles > 0.28).astype(np.uint8) * 255
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(active_spectacles, kernel_clean, iterations=1)

        # Feather boundaries
        feathered = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, (5, 5), 0)
        final_matte = np.clip(feathered * effective_strength, 0.0, 1.0)

        # 5. Temporal EMA Smoothing
        if track_id is not None:
            prev = self._history.get(track_id)
            if prev is not None and prev.shape == final_matte.shape:
                final_matte = self.temporal_alpha * final_matte + (1.0 - self.temporal_alpha) * prev
            self._history[track_id] = final_matte.copy()

        return final_matte

    def preserve_spectacles_on_swapped(
        self,
        original_crop: np.ndarray,
        swapped_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        strength: Optional[float] = None,
        track_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blends authentic spectacles from original_crop over swapped_crop.
        Returns:
            (composite_crop, spectacles_matte)
        """
        if original_crop is None or swapped_crop is None:
            return swapped_crop, np.zeros((128, 128), dtype=np.float32)

        spec_matte = self.detect_spectacles_mask(
            aligned_crop=original_crop,
            landmarks=landmarks,
            strength=strength,
            track_id=track_id,
        )

        if np.max(spec_matte) < 0.05:
            return swapped_crop, spec_matte

        # Composite: swapped * (1 - matte) + original * matte
        alpha_3d = np.repeat(spec_matte[:, :, np.newaxis], 3, axis=2)
        composite = (
            swapped_crop.astype(np.float32) * (1.0 - alpha_3d) +
            original_crop.astype(np.float32) * alpha_3d
        )
        return np.clip(composite, 0.0, 255.0).astype(np.uint8), spec_matte
