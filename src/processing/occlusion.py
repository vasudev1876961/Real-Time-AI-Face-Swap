"""
Occlusion-Aware Facial Masking and Foreground Object Detection.
Identifies hands, fingers, glasses, microphones, coffee cups, and foreign objects
crossing the face boundary, carving them out so swapped faces blend behind them.
Includes landmark-guided facial feature protection (eyes, brows, mouth),
spectacles preservation, skin-on-skin hand/finger carving, and temporal multi-frame matte stabilization.
"""

from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.processing.spectacles import SpectaclesPreservationEngine
from src.utils.logger import get_logger

logger = get_logger("Occlusion")


class TemporalOcclusionStabilizer:
    """
    Maintains temporal exponential moving average (EMA) consistency for occlusion
    mattes across consecutive frames to prevent edge shimmer and flickering.
    """

    def __init__(self, alpha: float = 0.65):
        """
        Args:
            alpha: Weight for the current frame [0.1, 1.0]. Lower = smoother transitions.
        """
        self.alpha = float(np.clip(alpha, 0.1, 1.0))
        self._track_mattes: Dict[int, np.ndarray] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        """Resets temporal matte history for a track or all tracks."""
        if track_id is not None:
            self._track_mattes.pop(track_id, None)
        else:
            self._history_clear(track_id)

    def _history_clear(self, track_id: Optional[int]) -> None:
        if track_id is not None:
            self._track_mattes.pop(track_id, None)
        else:
            self._track_mattes.clear()

    def smooth(self, occlusion_matte: np.ndarray, track_id: int = 1) -> np.ndarray:
        """Applies EMA smoothing to the occlusion probability matte."""
        if occlusion_matte is None or occlusion_matte.size == 0:
            return occlusion_matte

        prev = self._track_mattes.get(track_id)
        if prev is None or prev.shape != occlusion_matte.shape:
            self._track_mattes[track_id] = occlusion_matte.copy().astype(np.float32)
            return occlusion_matte

        smoothed = self.alpha * occlusion_matte.astype(np.float32) + (1.0 - self.alpha) * prev
        self._track_mattes[track_id] = smoothed
        return np.clip(smoothed, 0.0, 1.0)


class SkinOnSkinOcclusionEngine:
    """
    Detects hands, fingers, wrists, and arms occluding the face where pure color
    chrominance methods fail because both the face and foreground object are
    biological human skin.
    Uses directional boundary disparity, shadow penumbra gradient steps, and
    localized high-pass morphological contours within the face mask.
    """

    def __init__(self, sensitivity: float = 0.60):
        self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

    def detect_hand_occlusion(
        self,
        aligned_crop: np.ndarray,
        base_mask: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Detects hand and finger boundaries crossing the face.
        Returns a single-channel floating-point matte [0.0, 1.0].
        """
        if aligned_crop is None or aligned_crop.size == 0 or base_mask is None:
            return np.zeros_like(base_mask, dtype=np.float32)

        h, w = aligned_crop.shape[:2]
        float_mask = (
            base_mask.astype(np.float32) / 255.0 if base_mask.dtype == np.uint8 else base_mask.copy()
        )

        gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)

        # 1. Evaluate directional gradient steps (Sobel magnitude)
        sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize gradient
        grad_norm = np.clip(grad_mag / (np.max(grad_mag) + 1e-5), 0.0, 1.0)

        # 2. Local adaptive thresholding isolates finger shadow contours and edge steps
        thresh_block = max(7, int(min(h, w) * 0.12) | 1)
        adaptive_edges = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_block, 4
        ).astype(np.float32) / 255.0

        # 3. YCrCb Skin verification: only consider skin-like regions
        ycrcb = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        is_skin = (cr >= 125) & (cr <= 180) & (cb >= 75) & (cb <= 140)

        # Fingers produce elongated structures with distinct edge boundaries
        raw_hand_edges = (grad_norm > (0.35 - self.sensitivity * 0.15)) & (adaptive_edges > 0.5) & is_skin & (float_mask > 0.3)

        # Connect contiguous finger contours
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(raw_hand_edges.astype(np.uint8), kernel_connect, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_connect)

        hand_matte = closed.astype(np.float32)
        hand_matte = cv2.GaussianBlur(hand_matte, (5, 5), 0)
        return np.clip(hand_matte * self.sensitivity, 0.0, 1.0)


class OcclusionDetector:
    """
    Detects occluding foreground objects (hands, fingers, glasses, cups, microphones)
    inside the facial region using multi-color-space skin profiling, spectacles preservation,
    skin-on-skin boundary disparity, and structural gradient analysis.
    """

    def __init__(
        self,
        sensitivity: float = 0.50,
        blur_radius: int = 7,
        temporal_alpha: float = 0.70,
        enable_spectacles: bool = True,
        spectacles_strength: float = 0.75,
        enable_hand_carving: bool = True,
        hand_carving_strength: float = 0.60,
        visualize_hud: bool = False,
    ):
        """
        Args:
            sensitivity: Detection sensitivity [0.0, 1.0]. Higher values detect subtle occlusions.
            blur_radius: Feathering blur radius for smooth occlusion boundaries.
            temporal_alpha: Weight for temporal smoothing across frames.
            enable_spectacles: Enable dedicated eyeglass frame and bridge preservation.
            spectacles_strength: Strength of spectacles preservation [0.0, 1.0].
            enable_hand_carving: Enable skin-on-skin hand and finger boundary detection.
            hand_carving_strength: Strength of hand occlusion carving [0.0, 1.0].
            visualize_hud: Produce augmented occlusion debug telemetry overlay.
        """
        self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        self.stabilizer = TemporalOcclusionStabilizer(alpha=temporal_alpha)

        # Phase 12 Advancements: Spectacles & Hand Occlusion
        self.enable_spectacles = enable_spectacles
        self.spectacles_strength = float(np.clip(spectacles_strength, 0.0, 1.0))
        self.spectacles_engine = SpectaclesPreservationEngine(
            default_strength=self.spectacles_strength,
            temporal_alpha=temporal_alpha,
        )

        self.enable_hand_carving = enable_hand_carving
        self.hand_carving_strength = float(np.clip(hand_carving_strength, 0.0, 1.0))
        self.hand_engine = SkinOnSkinOcclusionEngine(sensitivity=self.hand_carving_strength)

        self.visualize_hud = visualize_hud
        self._last_metrics: Dict[str, Any] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        """Resets temporal smoothing history."""
        self.stabilizer.reset(track_id)
        self.spectacles_engine.reset(track_id)

    def compute_feature_protection_mask(
        self,
        shape: Tuple[int, int],
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Creates a soft protection mask over core facial features (eyes, eyebrows, mouth)
        to prevent natural facial variations (dark pupils, lashes, lipstick) from being
        falsely flagged as foreground occlusions.
        """
        h, w = shape[:2]
        protection = np.zeros((h, w), dtype=np.float32)

        lms = landmarks if landmarks is not None else INSWAPPER_STANDARD_128
        scale_x, scale_y = w / 128.0, h / 128.0
        scaled_lms = lms * np.array([scale_x, scale_y], dtype=np.float32)

        left_eye = scaled_lms[0]
        right_eye = scaled_lms[1]
        nose = scaled_lms[2]
        left_mouth = scaled_lms[3]
        right_mouth = scaled_lms[4]

        iod = float(np.linalg.norm(right_eye - left_eye))
        eye_r_x = int(max(4, iod * 0.22))
        eye_r_y = int(max(3, iod * 0.18))

        # 1. Left and Right Eye & Brow Protection
        cv2.ellipse(
            protection,
            (int(left_eye[0]), int(left_eye[1] - eye_r_y * 0.15)),
            (eye_r_x, eye_r_y),
            0, 0, 360, 1.0, -1
        )
        cv2.ellipse(
            protection,
            (int(right_eye[0]), int(right_eye[1] - eye_r_y * 0.15)),
            (eye_r_x, eye_r_y),
            0, 0, 360, 1.0, -1
        )

        # 2. Mouth Protection (inner lip area)
        mouth_center = (left_mouth + right_mouth) / 2.0
        mouth_w = int(max(6, np.linalg.norm(right_mouth - left_mouth) * 0.50))
        mouth_h = int(max(4, iod * 0.16))
        cv2.ellipse(
            protection,
            (int(mouth_center[0]), int(mouth_center[1])),
            (mouth_w, mouth_h),
            0, 0, 360, 1.0, -1
        )

        # 3. Soft Gaussian feathering so protection tapers gently
        k_size = max(5, int(iod * 0.15))
        if k_size % 2 == 0:
            k_size += 1
        protection = cv2.GaussianBlur(protection, (k_size, k_size), 0)
        return np.clip(protection, 0.0, 1.0)

    def detect_occlusion_mask(
        self,
        aligned_crop: np.ndarray,
        base_mask: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
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

        # 3. High-Frequency Edge Disparity for structural objects (mug rims, phones, pens)
        gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140).astype(np.float32) / 255.0
        edge_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        structural_occlusion = candidate_non_skin | ((edge_dilated > 0.5) & (chroma_dist > 1.8) & (float_mask > 0.4))

        # 4. Phase 12 Spectacles Preservation
        spectacles_matte = np.zeros((h, w), dtype=np.float32)
        if self.enable_spectacles:
            spectacles_matte = self.spectacles_engine.detect_spectacles_mask(
                aligned_crop=aligned_crop,
                landmarks=landmarks,
                strength=self.spectacles_strength,
                track_id=track_id,
            )

        # 5. Phase 12 Skin-on-Skin Hand/Finger Occlusion Carving
        hand_matte = np.zeros((h, w), dtype=np.float32)
        if self.enable_hand_carving:
            hand_matte = self.hand_engine.detect_hand_occlusion(
                aligned_crop=aligned_crop,
                base_mask=base_mask,
                landmarks=landmarks,
            )

        # 6. Feature Protection: Protect inner eyes, brows, and mouth unless strong foreign object or spectacles
        protection = self.compute_feature_protection_mask((h, w), landmarks)
        # Objects that have strong non-skin color or high contrast can override protection
        high_confidence_foreign = (chroma_dist > 4.5) | (ycrcb[:, :, 0] < 25) | (spectacles_matte > 0.35)
        suppressed_occlusion = structural_occlusion & ((protection < 0.55) | high_confidence_foreign)

        # Combine general occlusion, hand occlusion, and spectacles
        total_occlusion = suppressed_occlusion.astype(np.float32) + hand_matte * 0.85 + spectacles_matte * 0.95
        total_occlusion = np.clip(total_occlusion, 0.0, 1.0)

        # 7. Morphological filtering (remove isolated salt-and-pepper noise, connect contiguous objects)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occl_uint8 = (total_occlusion * 255.0).astype(np.uint8)
        opened = cv2.morphologyEx(occl_uint8, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # 8. Smooth feathered occlusion matte
        occlusion_float = closed.astype(np.float32) / 255.0
        feathered = cv2.GaussianBlur(occlusion_float, (self.blur_radius, self.blur_radius), 0)
        feathered = np.clip(feathered, 0.0, 1.0)

        # 9. Temporal Smoothing if track_id is provided
        if track_id is not None:
            feathered = self.stabilizer.smooth(feathered, track_id=track_id)

        # Cache metrics for telemetry HUD
        self._last_metrics = {
            "mean_occlusion": float(np.mean(feathered)),
            "max_occlusion": float(np.max(feathered)),
            "spectacles_active": bool(np.max(spectacles_matte) > 0.25),
            "hand_active": bool(np.max(hand_matte) > 0.25),
        }

        return feathered

    def refine_mask_with_occlusion(
        self,
        base_mask: np.ndarray,
        aligned_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Subtracts detected occlusions from the base face mask.
        Returns the refined single-channel mask in the same dtype as base_mask.
        """
        if self.sensitivity <= 0.01 and not self.enable_spectacles and not self.enable_hand_carving:
            return base_mask

        occlusion_matte = self.detect_occlusion_mask(aligned_crop, base_mask, landmarks, track_id=track_id)

        orig_dtype = base_mask.dtype
        mask_f = base_mask.astype(np.float32) / 255.0 if orig_dtype == np.uint8 else base_mask.astype(np.float32)

        # Subtract occlusion matte from face mask
        refined = np.clip(mask_f * (1.0 - occlusion_matte), 0.0, 1.0)

        if orig_dtype == np.uint8:
            return (refined * 255.0).astype(np.uint8)
        return refined

    def visualize_occlusion_hud(
        self,
        frame: np.ndarray,
        occlusion_matte: np.ndarray,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Generates an augmented debug telemetry overlay highlighting occluded zones:
        Neon Emerald for general occlusions/hands, Golden Amber for spectacles frames.
        """
        if frame is None or occlusion_matte is None:
            return frame

        vis = frame.copy()
        h, w = frame.shape[:2]

        if crop_rect is not None:
            x, y, cw, ch = crop_rect
            matte_resized = cv2.resize(occlusion_matte, (cw, ch), interpolation=cv2.INTER_LINEAR)
            roi = vis[y : y + ch, x : x + cw]
            color_tint = np.zeros_like(roi)
            color_tint[:, :] = [30, 235, 30]  # Neon Emerald
            alpha_3d = np.repeat(matte_resized[:, :, np.newaxis], 3, axis=2) * 0.45
            roi_blend = (roi.astype(np.float32) * (1.0 - alpha_3d) + color_tint.astype(np.float32) * alpha_3d)
            vis[y : y + ch, x : x + cw] = np.clip(roi_blend, 0, 255).astype(np.uint8)
        else:
            matte_resized = cv2.resize(occlusion_matte, (w, h), interpolation=cv2.INTER_LINEAR)
            color_tint = np.zeros_like(vis)
            color_tint[:, :] = [30, 235, 30]
            alpha_3d = np.repeat(matte_resized[:, :, np.newaxis], 3, axis=2) * 0.40
            vis = (vis.astype(np.float32) * (1.0 - alpha_3d) + color_tint.astype(np.float32) * alpha_3d)
            vis = np.clip(vis, 0, 255).astype(np.uint8)

        return vis

    def get_last_metrics(self) -> Dict[str, Any]:
        """Returns the latest occlusion detection telemetry summary."""
        return dict(self._last_metrics)
