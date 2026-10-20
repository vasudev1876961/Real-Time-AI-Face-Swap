"""
Virtual Portrait Studio Lighting & Edge Rim Light Engine (Phase 17).
Simulates 3-point portrait studio illumination, directional edge rim lighting,
and subtle background separation contrast for Hollywood cinematic depth.
"""

from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("StudioLighting")


class StudioLightingBackdropEngine:
    """
    Renders professional 3-point studio lighting, high-angle hair rim kicks,
    and background luminance attenuation around the transformed facial subject.
    """

    COLOR_PRESETS = {
        "warm_gold": (120, 195, 255),       # BGR: Golden amber studio spotlight
        "cool_cyan": (255, 225, 110),       # BGR: Cool daylight / sci-fi rim
        "neon_magenta": (220, 80, 235),     # BGR: Cyberpunk / synthwave neon
        "neutral_white": (245, 245, 245),   # BGR: Balanced 5500K studio strobe
        "sunset_orange": (80, 150, 255),    # BGR: Deep golden hour rim
    }

    def __init__(
        self,
        default_rim_intensity: float = 0.50,
        default_rim_thickness: int = 3,
        default_rim_color: str = "warm_gold",
        default_backdrop_dim: float = 0.15,
    ):
        self.rim_intensity: float = float(np.clip(default_rim_intensity, 0.0, 1.0))
        self.rim_thickness: int = max(1, min(10, int(default_rim_thickness)))
        self.rim_color_name: str = default_rim_color
        self.backdrop_dim: float = float(np.clip(default_backdrop_dim, 0.0, 0.60))

        self._last_metrics: Dict[str, Any] = {
            "applied": False,
            "rim_intensity": self.rim_intensity,
            "rim_color": self.rim_color_name,
            "backdrop_dim": self.backdrop_dim,
        }

    def set_config(
        self,
        rim_intensity: Optional[float] = None,
        rim_thickness: Optional[int] = None,
        rim_color: Optional[str] = None,
        backdrop_dim: Optional[float] = None,
    ) -> None:
        """Dynamically tunes studio lighting parameters."""
        if rim_intensity is not None:
            self.rim_intensity = float(np.clip(rim_intensity, 0.0, 1.0))
        if rim_thickness is not None:
            self.rim_thickness = max(1, min(10, int(rim_thickness)))
        if rim_color is not None:
            clean_col = rim_color.strip().lower()
            if clean_col in self.COLOR_PRESETS or clean_col == "ambient_adaptive":
                self.rim_color_name = clean_col
            else:
                logger.warning(f"Unknown studio rim color '{rim_color}', defaulting to warm_gold.")
                self.rim_color_name = "warm_gold"
        if backdrop_dim is not None:
            self.backdrop_dim = float(np.clip(backdrop_dim, 0.0, 0.60))

    def _resolve_color_bgr(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the BGR 3-element float array for the active rim light color."""
        if self.rim_color_name == "ambient_adaptive":
            if mask is not None and frame is not None:
                bg_mask = (mask < 0.2).astype(np.uint8)
                if cv2.countNonZero(bg_mask) > 100:
                    mean_bg = cv2.mean(frame, mask=bg_mask)[:3]
                    # Brighten and slightly saturate ambient color
                    col = np.array(mean_bg, dtype=np.float32) * 1.5
                    return np.clip(col, 50.0, 255.0)
            return np.array([200.0, 200.0, 200.0], dtype=np.float32)

        preset = self.COLOR_PRESETS.get(self.rim_color_name, self.COLOR_PRESETS["warm_gold"])
        return np.array(preset, dtype=np.float32)

    def apply_studio_lighting(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
        intensity: Optional[float] = None,
        thickness: Optional[int] = None,
        color_name: Optional[str] = None,
        backdrop_dim: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies studio rim lighting and backdrop attenuation to frame.
        Can be applied either on the aligned crop or on full frame with mask.
        """
        eff_intensity = self.rim_intensity if intensity is None else float(np.clip(intensity, 0.0, 1.0))
        eff_thickness = self.rim_thickness if thickness is None else max(1, min(10, int(thickness)))
        eff_dim = self.backdrop_dim if backdrop_dim is None else float(np.clip(backdrop_dim, 0.0, 0.60))
        
        if color_name is not None:
            self.set_config(rim_color=color_name)

        if frame is None or frame.size == 0 or (eff_intensity <= 0.01 and eff_dim <= 0.01):
            return frame, {"applied": False}

        h, w = frame.shape[:2]
        output = frame.copy()

        # If no explicit mask is provided, generate a smooth foreground boundary from frame center / landmarks
        if mask is None:
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
            if landmarks is not None and len(landmarks) >= 5:
                pts = np.int32(landmarks[:, :2])
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(mask_u8, hull, 255)
            else:
                # Ellipse around center
                center = (w // 2, h // 2)
                axes = (int(w * 0.35), int(h * 0.45))
                cv2.ellipse(mask_u8, center, axes, 0, 0, 360, 255, -1)
            mask_float = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (21, 21), 0)
        else:
            if mask.dtype != np.float32:
                mask_float = mask.astype(np.float32) / (255.0 if mask.max() > 1.0 else 1.0)
            else:
                mask_float = mask
            if mask_float.shape[:2] != (h, w):
                mask_float = cv2.resize(mask_float, (w, h))

        # 1. Background Attenuation & Isolation
        if eff_dim > 0.01:
            bg_factor = 1.0 - (eff_dim * (1.0 - np.clip(mask_float, 0.0, 1.0)))
            if output.ndim == 3:
                bg_factor = bg_factor[:, :, np.newaxis]
            output = np.clip(output.astype(np.float32) * bg_factor, 0.0, 255.0).astype(np.uint8)

        # 2. Edge Rim Light Extraction & Synthesis
        if eff_intensity > 0.01:
            bin_mask = (mask_float > 0.35).astype(np.uint8) * 255
            ksize = max(3, eff_thickness * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            dilated = cv2.dilate(bin_mask, kernel, iterations=1)
            eroded = cv2.erode(bin_mask, kernel, iterations=1)
            rim_boundary = cv2.subtract(dilated, eroded)

            # Smooth edge transition
            blur_size = max(3, (ksize // 2) * 2 + 1)
            rim_gradient = cv2.GaussianBlur(rim_boundary.astype(np.float32) / 255.0, (blur_size, blur_size), 0)

            # High-angle directional weighting (favor upper rim & lateral hair highlights)
            y_indices = np.linspace(1.2, 0.7, h, dtype=np.float32)[:, np.newaxis]
            directional_weight = np.repeat(y_indices, w, axis=1)
            rim_weight = np.clip(rim_gradient * directional_weight * eff_intensity, 0.0, 1.0)

            rim_color = self._resolve_color_bgr(frame, mask_float)

            # Screen blending for soft glow: Result = 1 - (1 - A)(1 - B)
            # Scaled additive light kick
            out_float = output.astype(np.float32)
            for c in range(3):
                added_light = rim_weight * rim_color[c]
                out_float[:, :, c] = np.clip(out_float[:, :, c] + added_light * 0.85, 0.0, 255.0)

            output = out_float.astype(np.uint8)

        metrics = {
            "applied": True,
            "rim_intensity": eff_intensity,
            "rim_thickness": eff_thickness,
            "rim_color": self.rim_color_name,
            "backdrop_dim": eff_dim,
        }
        self._last_metrics = metrics
        return output, metrics

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns current studio lighting telemetry."""
        return {
            "rim_intensity": self.rim_intensity,
            "rim_thickness": self.rim_thickness,
            "rim_color": self.rim_color_name,
            "backdrop_dim": self.backdrop_dim,
            "available_colors": list(self.COLOR_PRESETS.keys()) + ["ambient_adaptive"],
            "last_applied": self._last_metrics.get("applied", False),
        }
