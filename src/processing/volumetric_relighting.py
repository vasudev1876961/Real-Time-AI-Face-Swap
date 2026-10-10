"""
Volumetric Relighting & Directional Cast Shadow Harmonization Engine (Phase 15).
Performs 3D key light vector tracking, directional cast shadow transfer,
and physics-grounded subsurface scattering (SSS) epidermal warmth synthesis.
"""

from typing import Optional, Tuple, Dict, Any
import math
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("VolumetricRelighting")


class VolumetricRelightingEngine:
    """
    Directional 3D lighting and shadow harmonization engine.
    Solves illumination discordance by transferring physical room keylighting,
    cast shadows (nose, brow, hair), and epidermal subsurface scattering warmth.
    """

    def __init__(
        self,
        default_shadow_strength: float = 0.50,
        default_warmth_strength: float = 0.45,
        shading_sigma: float = 12.0,
    ):
        """
        Args:
            default_shadow_strength: Transfer strength for directional cast shadows [0.0, 1.0].
            default_warmth_strength: Intensity of Subsurface Scattering (SSS) warmth along terminators [0.0, 1.0].
            shading_sigma: Filter radius for low-frequency illumination fields.
        """
        self.default_shadow_strength = float(np.clip(default_shadow_strength, 0.0, 1.0))
        self.default_warmth_strength = float(np.clip(default_warmth_strength, 0.0, 1.0))
        self.shading_sigma = max(2.0, float(shading_sigma))
        self._last_light_vector = (0.0, 0.0, 1.0)

    def estimate_3d_light_vector(
        self,
        face_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """
        Estimates the normalized 3D primary key light vector [Lx, Ly, Lz] illuminating the subject.
        Lx: Left (+) to Right (-)
        Ly: Top (+) to Bottom (-)
        Lz: Depth (Towards Camera +)
        """
        if face_crop is None or face_crop.size == 0:
            return (0.0, 0.0, 1.0)

        h, w = face_crop.shape[:2]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1. Bilateral / Gaussian smoothing to isolate macro-shading
        ksize = int(self.shading_sigma * 2.5) | 1
        shading = cv2.GaussianBlur(gray, (ksize, ksize), self.shading_sigma)

        # 2. Quadrant and hemisphere luminance differentials
        half_w = w // 2
        half_h = h // 2

        left_lum = np.mean(shading[:, :half_w])
        right_lum = np.mean(shading[:, half_w:])
        top_lum = np.mean(shading[:half_h, :])
        bottom_lum = np.mean(shading[half_h:, :])

        # Directional gradients
        lx = float(left_lum - right_lum) / 128.0
        ly = float(top_lum - bottom_lum) / 128.0

        # Z-component represents overall frontality
        center_crop = shading[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        avg_lum = np.mean(center_crop) / 255.0
        lz = max(0.2, float(avg_lum * 1.5))

        # Normalize to unit vector
        norm = math.sqrt(lx * lx + ly * ly + lz * lz)
        if norm > 1e-6:
            lx, ly, lz = lx / norm, ly / norm, lz / norm
        else:
            lx, ly, lz = 0.0, 0.0, 1.0

        self._last_light_vector = (round(lx, 3), round(ly, 3), round(lz, 3))
        return self._last_light_vector

    def extract_directional_shadow_map(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """
        Extracts directional cast shadow regions by isolating high-gradient luminance attenuation.
        
        Returns:
            (H, W) float32 normalized shadow modulation field in [0.0, 1.0].
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32) / 255.0

        # Macro lighting field
        ksize = int(self.shading_sigma * 3) | 1
        macro = cv2.GaussianBlur(l_chan, (ksize, ksize), self.shading_sigma)

        # Shadow ratio: areas darker than macro average indicate cast shadows
        ratio = np.clip(l_chan / (macro + 1e-4), 0.5, 1.2)
        return ratio

    def inject_subsurface_scattering_warmth(
        self,
        bgr_crop: np.ndarray,
        shading_field: np.ndarray,
        warmth_strength: float = 0.45,
    ) -> np.ndarray:
        """
        Simulates dermal blood backscatter along the shadow terminator boundary
        (the geometric transition zone where directional light meets penumbra shadow).
        Injects warm reddish/peach chromatic energy to eliminate dull gray skin tones.
        """
        if warmth_strength <= 0.01:
            return bgr_crop

        h, w = bgr_crop.shape[:2]
        if shading_field.shape[:2] != (h, w):
            shading_field = cv2.resize(shading_field, (w, h), interpolation=cv2.INTER_LINEAR)

        # Detect the shadow terminator via gradient magnitude of the shading field
        grad_x = cv2.Sobel(shading_field, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(shading_field, cv2.CV_32F, 0, 1, ksize=3)
        terminator_mag = cv2.magnitude(grad_x, grad_y)

        # Normalize and isolate peak terminator band
        max_val = np.max(terminator_mag)
        if max_val > 1e-5:
            terminator_norm = terminator_mag / max_val
        else:
            terminator_norm = terminator_mag

        # Soft band-pass along terminator zone (subsurface scattering penumbra)
        terminator_mask = cv2.GaussianBlur(terminator_norm, (7, 7), 2.0)
        terminator_mask = np.clip(terminator_mask * 2.0, 0.0, 1.0)

        # Transfer warmth in YCrCb color space (Cr channel controls red/peach chrominance)
        ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

        # Warmth injection: boost Cr (reddish) and slightly suppress Cb (cyan/blue)
        warmth_factor = float(warmth_strength) * 18.0
        cr_adapted = cr + (terminator_mask * warmth_factor)
        cb_adapted = cb - (terminator_mask * (warmth_factor * 0.35))

        ycrcb[:, :, 1] = np.clip(cr_adapted, 0.0, 255.0)
        ycrcb[:, :, 2] = np.clip(cb_adapted, 0.0, 255.0)

        result_bgr = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        return result_bgr

    def harmonize_volumetric_lighting(
        self,
        original_crop: np.ndarray,
        swapped_crop: np.ndarray,
        shadow_strength: Optional[float] = None,
        sss_warmth: Optional[float] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes complete volumetric relighting pass:
        1. Estimates 3D key light vector.
        2. Transfers directional cast shadows from original webcam face.
        3. Simulates Subsurface Scattering (SSS) epidermal warmth.
        
        Returns:
            (Harmonized BGR image, Telemetry dictionary)
        """
        if original_crop is None or swapped_crop is None:
            return swapped_crop, {"light_vector": self._last_light_vector, "sss_applied": False}

        eff_shadow = self.default_shadow_strength if shadow_strength is None else float(np.clip(shadow_strength, 0.0, 1.0))
        eff_warmth = self.default_warmth_strength if sss_warmth is None else float(np.clip(sss_warmth, 0.0, 1.0))

        h, w = swapped_crop.shape[:2]
        if original_crop.shape[:2] != (h, w):
            orig_matched = cv2.resize(original_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            orig_matched = original_crop

        # 1. Estimate 3D Light Direction
        light_vec = self.estimate_3d_light_vector(orig_matched, landmarks=landmarks)

        # 2. Extract and Harmonize Directional Cast Shadows
        shadow_map_src = self.extract_directional_shadow_map(orig_matched)
        shadow_map_tgt = self.extract_directional_shadow_map(swapped_crop)

        # Shadow modulation ratio
        shadow_delta = (shadow_map_src - shadow_map_tgt) * eff_shadow
        
        # Apply shadow modulation to target luminance
        lab_tgt = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab_tgt[:, :, 0]
        l_adapted = np.clip(l_chan * (1.0 + shadow_delta), 0.0, 255.0)
        lab_tgt[:, :, 0] = l_adapted

        swapped_lit = cv2.cvtColor(lab_tgt.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # 3. Subsurface Scattering Epidermal Warmth Injection
        swapped_sss = self.inject_subsurface_scattering_warmth(
            swapped_lit,
            shading_field=shadow_map_src,
            warmth_strength=eff_warmth,
        )

        telemetry = {
            "light_vector": light_vec,
            "shadow_strength": eff_shadow,
            "warmth_strength": eff_warmth,
            "sss_applied": (eff_warmth > 0.01),
        }

        return swapped_sss, telemetry
