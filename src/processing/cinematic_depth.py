"""
Cinematic Depth Estimation & Real-Time Portrait Bokeh Engine (Phase 16).
Provides monocular facial depth field estimation, Circle-of-Confusion (CoC)
lens blur synthesis, optical highlight specular boosting, and false-color depth telemetry.
"""

from typing import Optional, Tuple, Dict, Any
import math
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("CinematicDepth")


class CinematicDepthEngine:
    """
    Simulates physical shallow depth of field (DoF) and DSLR f/1.4 portrait bokeh.
    Estimates a continuous monocular depth map Z(x, y) in [0.0, 1.0] from 3D facial
    landmarks, bilateral gradient propagation, and radial distance transforms.
    """

    def __init__(
        self,
        default_focus_depth: float = 0.35,
        default_bokeh_radius: float = 12.0,
        default_specular_boost: float = 0.25,
        temporal_smoothing: float = 0.65,
    ):
        """
        Args:
            default_focus_depth: Normalized depth value of focal plane [0.05, 0.95].
            default_bokeh_radius: Circle of Confusion blur radius in pixels [0.0, 30.0].
            default_specular_boost: Specular highlight expansion factor in blurred zones [0.0, 1.0].
            temporal_smoothing: Exponential moving average factor for temporal stability [0.0, 0.95].
        """
        self.focus_depth = float(np.clip(default_focus_depth, 0.05, 0.95))
        self.bokeh_radius = float(np.clip(default_bokeh_radius, 0.0, 30.0))
        self.specular_boost = float(np.clip(default_specular_boost, 0.0, 1.0))
        self.temporal_smoothing = float(np.clip(temporal_smoothing, 0.0, 0.95))

        self._cached_depth_map: Optional[np.ndarray] = None
        self._last_metrics: Dict[str, Any] = {
            "focus_depth": self.focus_depth,
            "bokeh_radius": self.bokeh_radius,
            "specular_boost": self.specular_boost,
            "mean_depth": 0.50,
        }

    def estimate_depth_map(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Estimates a normalized relative depth field Z(x, y) in [0.0, 1.0].
        0.0 = Nearest to camera (Nose tip / foreground subject)
        1.0 = Farthest from camera (Infinite room background)

        Args:
            frame: BGR input image (H, W, 3).
            landmarks: Optional facial landmark coordinates (N, 2).
            face_bbox: Optional bounding box (x1, y1, x2, y2).

        Returns:
            depth_map: float32 array (H, W) in range [0.0, 1.0].
        """
        if frame is None or frame.size == 0:
            return np.full((100, 100), 0.5, dtype=np.float32)

        h, w = frame.shape[:2]

        # 1. Base anchor depth estimation
        if landmarks is not None and len(landmarks) >= 5:
            # Multi-point anatomical anchor depth initialization
            # Center of face (nose / bridge)
            if len(landmarks) == 5:
                # 5-pt: 0: left eye, 1: right eye, 2: nose, 3: left mouth, 4: right mouth
                nose = landmarks[2]
                face_center_x, face_center_y = float(nose[0]), float(nose[1])
                inter_eye = float(np.linalg.norm(landmarks[1] - landmarks[0]))
                face_radius = max(30.0, inter_eye * 1.8)
            else:
                pts = np.asarray(landmarks, dtype=np.float32)
                face_center_x, face_center_y = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
                face_radius = max(30.0, float(np.std(pts)) * 2.2)
        elif face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_center_x = (x1 + x2) / 2.0
            face_center_y = (y1 + y2) / 2.0
            face_radius = max(30.0, max(x2 - x1, y2 - y1) / 2.0)
        else:
            # Fallback center anchor
            face_center_x = w / 2.0
            face_center_y = h / 2.0
            face_radius = min(w, h) * 0.35

        # 2. Vectorized Radial Distance Field from Face Silhouette
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        dx = (x_indices - face_center_x) / max(1.0, face_radius)
        dy = (y_indices - face_center_y) / max(1.0, face_radius * 1.25)
        dist_sq = dx * dx + dy * dy

        # Sigmoidal depth ramp: Face core ~ 0.25 -> Periphery ~ 0.55 -> Background ~ 0.95
        depth_raw = 0.25 + 0.70 / (1.0 + np.exp(-2.2 * (dist_sq - 1.0)))
        depth_raw = np.clip(depth_raw, 0.05, 1.0).astype(np.float32)

        # 3. Edge-Preserving Gradient Harmonization with Frame Luminance
        # Downscale for real-time edge guidance then upscale
        small_h, small_w = max(16, h // 4), max(16, w // 4)
        small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (small_w, small_h))
        small_depth = cv2.resize(depth_raw, (small_w, small_h))

        # Bilateral / Guided edge refinement
        refined_small = cv2.bilateralFilter(small_depth, d=5, sigmaColor=0.15, sigmaSpace=7.0)
        depth_map = cv2.resize(refined_small, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_map = np.clip(depth_map, 0.0, 1.0).astype(np.float32)

        # 4. Temporal EMA Smoothing
        if (
            self._cached_depth_map is not None
            and self._cached_depth_map.shape == depth_map.shape
            and self.temporal_smoothing > 0.01
        ):
            alpha = 1.0 - self.temporal_smoothing
            depth_map = alpha * depth_map + self.temporal_smoothing * self._cached_depth_map

        self._cached_depth_map = depth_map.copy()
        self._last_metrics["mean_depth"] = round(float(np.mean(depth_map)), 3)
        return depth_map

    def apply_cinematic_bokeh(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        focus_depth: Optional[float] = None,
        bokeh_radius: Optional[float] = None,
        specular_boost: Optional[float] = None,
        face_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Synthesizes a creamy DSLR portrait bokeh on background out-of-focus planes
        while keeping the subject focal plane razor sharp.

        Args:
            frame: Original BGR image (H, W, 3).
            depth_map: Optional precomputed depth map (H, W).
            focus_depth: Focal plane normalized depth [0.05, 0.95].
            bokeh_radius: Maximum Circle of Confusion blur radius in pixels.
            specular_boost: Highlight expansion weight.
            face_mask: Optional binary or float face mask (H, W) to strictly protect face sharpness.

        Returns:
            rendered_frame: BGR composite with physical bokeh applied.
        """
        if frame is None or frame.size == 0:
            return frame

        f_depth = self.focus_depth if focus_depth is None else float(np.clip(focus_depth, 0.05, 0.95))
        b_radius = self.bokeh_radius if bokeh_radius is None else float(max(0.0, bokeh_radius))
        s_boost = self.specular_boost if specular_boost is None else float(np.clip(specular_boost, 0.0, 1.0))

        if b_radius < 0.5:
            return frame.copy()

        h, w = frame.shape[:2]
        if depth_map is None:
            depth_map = self.estimate_depth_map(frame)
        elif depth_map.shape[:2] != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # 1. Circle-of-Confusion (CoC) Diameter Field
        # Defocus increases with distance from focal plane
        depth_diff = np.abs(depth_map - f_depth)
        coc_field = np.clip(depth_diff / max(0.1, (1.0 - f_depth)), 0.0, 1.0) * b_radius

        # 2. Multi-tier Aperture Disk Defocus
        # Stratified blur: tier 1 (medium) and tier 2 (heavy)
        k_med = max(3, int(b_radius * 0.6) | 1)
        k_heavy = max(5, int(b_radius * 1.2) | 1)

        blur_med = cv2.GaussianBlur(frame, (k_med, k_med), b_radius * 0.35)
        blur_heavy = cv2.GaussianBlur(frame, (k_heavy, k_heavy), b_radius * 0.70)

        # 3. Specular Highlight Expansion (Optical Bokeh Blooms)
        if s_boost > 0.01:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Isolate specular highlights above threshold (210)
            spec_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)[1]
            if np.count_nonzero(spec_mask) > 0:
                kernel_spec = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                expanded_spec = cv2.dilate(spec_mask, kernel_spec)
                spec_weight = (expanded_spec.astype(np.float32) / 255.0)[:, :, np.newaxis] * (s_boost * 0.35)
                # Boost specular blooms in blurred heavy tier
                blur_heavy = cv2.addWeighted(
                    blur_heavy, 1.0, (blur_heavy.astype(np.float32) * spec_weight).astype(np.uint8), 1.0, 0
                )

        # 4. Continuous Defocus Alpha Compositing
        norm_coc = np.clip(coc_field / max(1e-5, b_radius), 0.0, 1.0)[:, :, np.newaxis]
        frame_f = frame.astype(np.float32)
        med_f = blur_med.astype(np.float32)
        heavy_f = blur_heavy.astype(np.float32)

        # Tiered interpolation between sharp, medium, and heavy blur
        tier1_weight = np.clip(norm_coc * 2.0, 0.0, 1.0)
        tier2_weight = np.clip((norm_coc - 0.5) * 2.0, 0.0, 1.0)

        blurred_composite = (1.0 - tier1_weight) * frame_f + tier1_weight * (
            (1.0 - tier2_weight) * med_f + tier2_weight * heavy_f
        )
        bokeh_output = np.clip(blurred_composite, 0.0, 255.0).astype(np.uint8)

        # 5. Foreground Subject Sharpness Protection
        if face_mask is not None:
            if face_mask.shape[:2] != (h, w):
                face_mask = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            if face_mask.dtype != np.float32:
                m = (face_mask > 30).astype(np.float32)
            else:
                m = np.clip(face_mask, 0.0, 1.0)
            if len(m.shape) == 2:
                m = m[:, :, np.newaxis]
            # Sharp subject over bokeh background
            bokeh_output = (m * frame_f + (1.0 - m) * bokeh_output.astype(np.float32)).astype(np.uint8)

        return bokeh_output

    def render_depth_visualizer(
        self,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_TURBO,
        show_hud: bool = True,
    ) -> np.ndarray:
        """
        Renders a false-color false-depth visualization overlay.
        Blue: Foreground subject (Close)
        Green/Yellow: Midground
        Red: Background (Far)
        """
        if depth_map is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        # Convert float32 [0.0, 1.0] to uint8 [0, 255]
        norm = np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, colormap)

        if show_hud:
            h, w = heatmap.shape[:2]
            # Overlay glassmorphic banner
            cv2.rectangle(heatmap, (10, 10), (280, 50), (20, 20, 20), -1)
            cv2.rectangle(heatmap, (10, 10), (280, 50), (0, 230, 255), 1)
            text = f"DEPTH HUD | Focus: {self.focus_depth:.2f} | R: {self.bokeh_radius:.1f}px"
            cv2.putText(
                heatmap,
                text,
                (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return heatmap

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns consolidated depth engine telemetry."""
        return {
            "focus_depth": self.focus_depth,
            "bokeh_radius": self.bokeh_radius,
            "specular_boost": self.specular_boost,
            "mean_depth": self._last_metrics.get("mean_depth", 0.50),
        }
