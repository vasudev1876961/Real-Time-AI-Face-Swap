"""
Ocular Gaze, Corneal Catchlights & Natural Blink Realism Subsystem (Phase 9).
Eliminates synthetic 'dead eyes / unblinking stare' and distorted blink artifacts
by synchronizing eyelid closures, injecting corneal specular highlights, and preserving
authentic iris contrast from the camera subject feed.
"""

from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.utils.logger import get_logger

logger = get_logger("EyeGaze")


class EyeGazePreserver:
    """
    Ocular fidelity engine responsible for natural blink synchronization,
    corneal specular catchlight injection, and gaze vector alignment.
    Works seamlessly across 128x128 crop coordinates and 512x512 neural super-resolution crops.
    """

    def __init__(
        self,
        default_strength: float = 0.70,
        blink_threshold: float = 0.28,
        catchlight_strength: float = 0.65,
    ):
        """
        Args:
            default_strength: Blending strength for eye preservation [0.0, 1.0].
            blink_threshold: Threshold below which an eye is classified as closed/blinking.
            catchlight_strength: Intensity factor for corneal highlight injection.
        """
        self.default_strength = float(np.clip(default_strength, 0.0, 1.0))
        self.blink_threshold = blink_threshold
        self.catchlight_strength = float(np.clip(catchlight_strength, 0.0, 1.0))

    def compute_ocular_masks(
        self,
        crop_shape: Tuple[int, int],
        landmarks: Optional[np.ndarray] = None,
        feather: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates feathered elliptical masks for left, right, and combined ocular regions.

        Args:
            crop_shape: (H, W) of crop.
            landmarks: 5 standard ArcFace landmarks [[x, y], ...].
            feather: Feathering blur radius.

        Returns:
            Tuple of (left_mask, right_mask, combined_mask) as float32 in [0.0, 1.0].
        """
        h, w = crop_shape[:2]
        left_mask = np.zeros((h, w), dtype=np.float32)
        right_mask = np.zeros((h, w), dtype=np.float32)

        if landmarks is not None and len(landmarks) >= 2:
            pts = np.array(landmarks, dtype=np.float32).copy()
            if pts.max() <= 1.05:
                pts[:, 0] *= w
                pts[:, 1] *= h
            elif (pts[:, 0].max() <= 135.0 and w > 200) or (pts[:, 1].max() <= 135.0 and h > 200):
                pts[:, 0] *= float(w) / 128.0
                pts[:, 1] *= float(h) / 128.0

            left_eye = pts[0]
            right_eye = pts[1]
            iod = float(np.linalg.norm(right_eye - left_eye))
            radius_x = int(max(5, iod * 0.24))
            radius_y = int(max(3, iod * 0.16))

            left_center = (int(left_eye[0]), int(left_eye[1]))
            right_center = (int(right_eye[0]), int(right_eye[1]))
        else:
            # Standard geometry fallback scaled to (h, w)
            radius_x = int(w * 0.12)
            radius_y = int(h * 0.07)
            left_center = (int(w * 0.36), int(h * 0.40))
            right_center = (int(w * 0.64), int(h * 0.40))

        # Render elliptical ocular boundaries
        cv2.ellipse(left_mask, left_center, (radius_x, radius_y), 0.0, 0.0, 360.0, 1.0, -1)
        cv2.ellipse(right_mask, right_center, (radius_x, radius_y), 0.0, 0.0, 360.0, 1.0, -1)

        ksize = feather * 2 + 1
        left_mask = cv2.GaussianBlur(left_mask, (ksize, ksize), feather * 0.5)
        right_mask = cv2.GaussianBlur(right_mask, (ksize, ksize), feather * 0.5)

        left_mask = np.clip(left_mask, 0.0, 1.0)
        right_mask = np.clip(right_mask, 0.0, 1.0)
        combined_mask = np.clip(left_mask + right_mask, 0.0, 1.0)

        return left_mask, right_mask, combined_mask

    def detect_blink_state(
        self,
        original_crop: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> Tuple[bool, bool, float, float]:
        """
        Measures vertical ocular gradient energy and luminance disparity
        to determine whether either eye is closed or squinting (blinking).
        In open eyes, dark pupil and white sclera create strong vertical gradients;
        in closed eyes, smooth eyelid skin produces low gradient energy.

        Returns:
            Tuple of (is_left_blinking, is_right_blinking, left_openness, right_openness).
        """
        if original_crop is None or original_crop.size == 0:
            return False, False, 1.0, 1.0

        gray = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY)
        # Vertical Sobel gradient measures eyelid edge contrast and pupil boundary
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_energy = np.abs(sobel_y)

        def _evaluate_eye(mask: np.ndarray) -> float:
            bin_mask = (mask > 0.35).astype(np.uint8)
            num_px = int(np.sum(bin_mask))
            if num_px < 8:
                return 1.0
            # Mean vertical gradient in ocular socket
            mean_grad = float(np.sum(grad_energy * bin_mask) / num_px)
            # Std of pixel intensities (pupil vs sclera creates high variance)
            eye_pixels = gray[bin_mask == 1]
            pixel_std = float(np.std(eye_pixels))
            # Composite openness metric: normalized in approx [0.0, 1.0]
            openness = (mean_grad * 0.015) + (pixel_std * 0.01)
            return float(np.clip(openness, 0.0, 1.0))

        left_openness = _evaluate_eye(left_mask)
        right_openness = _evaluate_eye(right_mask)

        is_left_blinking = left_openness < self.blink_threshold
        is_right_blinking = right_openness < self.blink_threshold

        return is_left_blinking, is_right_blinking, left_openness, right_openness

    def extract_corneal_catchlights(
        self,
        original_crop: np.ndarray,
        combined_mask: np.ndarray,
        luminance_threshold: int = 210,
    ) -> np.ndarray:
        """
        Extracts micro-specular corneal highlights ('catchlights') from original camera eyes.
        These specular sparkles give life to eyes and capture ambient lighting direction.

        Returns:
            (H, W, 3) float32 corneal highlight layer [0.0, 255.0].
        """
        h, w = original_crop.shape[:2]
        gray = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY)

        # High-pass threshold inside the ocular boundary
        _, spec_thresh = cv2.threshold(gray, luminance_threshold, 255, cv2.THRESH_BINARY)
        spec_mask = (spec_thresh.astype(np.float32) / 255.0) * combined_mask

        # Small morphological opening to isolate tiny point specular reflections
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        spec_mask_u8 = (spec_mask * 255).astype(np.uint8)
        spec_filtered = cv2.morphologyEx(spec_mask_u8, cv2.MORPH_OPEN, kernel).astype(np.float32) / 255.0

        # Create 3-channel catchlight overlay
        spec_3ch = np.repeat(spec_filtered[:, :, np.newaxis], 3, axis=2)
        catchlights = (original_crop.astype(np.float32) - 180.0) * 1.5
        catchlights = np.clip(catchlights, 0.0, 255.0) * spec_3ch

        return catchlights

    def preserve_eye_realism(
        self,
        original_crop: np.ndarray,
        swapped_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        strength: Optional[float] = None,
        sync_blinks: bool = True,
        inject_catchlights: bool = True,
    ) -> np.ndarray:
        """
        Full eye preservation and enhancement pipeline:
        1. Segments left/right ocular boundaries.
        2. Detects blinking/squinting states.
        3. If blinking, softly blends authentic eyelid skin & lash textures from original crop.
        4. If open, extracts and injects corneal specular catchlights and restores iris depth.

        Args:
            original_crop: Original aligned face crop (128x128 or 512x512).
            swapped_crop: Swapped face crop before eye refinement.
            landmarks: 5 standard ArcFace landmarks.
            strength: Blending strength override [0.0, 1.0].
            sync_blinks: Whether to synchronize blink closures.
            inject_catchlights: Whether to transfer room specular highlights to eyes.

        Returns:
            Refined swapped face crop with natural eye realism.
        """
        if original_crop is None or swapped_crop is None:
            return swapped_crop

        eff_strength = self.default_strength if strength is None else float(np.clip(strength, 0.0, 1.0))
        if eff_strength <= 0.01:
            return swapped_crop

        # Ensure spatial match
        h, w = swapped_crop.shape[:2]
        sh, sw = original_crop.shape[:2]
        if (sw, sh) != (w, h):
            orig_matched = cv2.resize(original_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            orig_matched = original_crop

        left_mask, right_mask, combined_mask = self.compute_ocular_masks(
            crop_shape=(h, w),
            landmarks=landmarks,
            feather=max(3, int(w * 0.035)),
        )

        left_blink, right_blink, left_open, right_open = self.detect_blink_state(
            orig_matched, left_mask, right_mask
        )

        result = swapped_crop.copy().astype(np.float32)

        # 1. Blink Synchronization: restore closed eyelid skin if subject blinked
        if sync_blinks and (left_blink or right_blink):
            blink_mask = np.zeros((h, w), dtype=np.float32)
            if left_blink:
                # Modulate mask by closure degree
                weight = 1.0 - (left_open / max(self.blink_threshold, 1e-4))
                blink_mask += left_mask * float(np.clip(weight, 0.5, 1.0))
            if right_blink:
                weight = 1.0 - (right_open / max(self.blink_threshold, 1e-4))
                blink_mask += right_mask * float(np.clip(weight, 0.5, 1.0))

            blink_mask = np.clip(blink_mask, 0.0, 1.0) * eff_strength
            blink_3ch = np.repeat(blink_mask[:, :, np.newaxis], 3, axis=2)

            # Smoothly composite original eyelid texture over swap
            result = result * (1.0 - blink_3ch) + orig_matched.astype(np.float32) * blink_3ch

        # 2. Open Eye Enhancement & Corneal Specular Catchlight Transfer
        if inject_catchlights and not (left_blink and right_blink):
            open_mask = combined_mask.copy()
            if left_blink:
                open_mask = np.maximum(0.0, open_mask - left_mask)
            if right_blink:
                open_mask = np.maximum(0.0, open_mask - right_mask)

            catchlights = self.extract_corneal_catchlights(orig_matched, open_mask)
            # Additive blend with saturation preservation
            catchlight_weight = self.catchlight_strength * eff_strength
            result = np.clip(result + catchlights * catchlight_weight, 0.0, 255.0)

            # Soft iris depth recovery: preserve 20% of authentic pupil contrast
            iris_weight = (open_mask[:, :, np.newaxis] * 0.25 * eff_strength)
            result = result * (1.0 - iris_weight) + orig_matched.astype(np.float32) * iris_weight

        return np.clip(result, 0, 255).astype(np.uint8)
