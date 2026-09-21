"""
Specular-Ambient Lighting & Directional Shading Harmonizer (Phase 9).
Decomposes facial luminance into Specular Highlights, Directional Diffuse Shading,
and Ambient Albedo, transferring physical room keylighting and dynamic skin highlights
onto synthetic swapped faces.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("LightingHarmonizer")


class SpecularAmbientLightingHarmonizer:
    """
    Directional 3D lighting adaptation engine.
    Ensures specular glints (forehead, cheekbones, nose bridge) and directional
    shadow gradients from physical webcam room lighting seamlessly illuminate the synthetic face.
    """

    def __init__(
        self,
        default_strength: float = 0.50,
        specular_threshold: int = 215,
        shading_sigma: float = 14.0,
    ):
        """
        Args:
            default_strength: Overall lighting transfer strength [0.0, 1.0].
            specular_threshold: Luminance threshold for skin specular glints.
            shading_sigma: Gaussian kernel radius for directional shading extraction.
        """
        self.default_strength = float(np.clip(default_strength, 0.0, 1.0))
        self.specular_threshold = specular_threshold
        self.shading_sigma = shading_sigma

    def extract_specular_highlights(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extracts high-intensity specular highlights (glints on forehead, nose bridge, cheekbones).

        Returns:
            (H, W, 3) float32 specular highlight layer.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.specular_threshold, 255, cv2.THRESH_BINARY)
        thresh_f = thresh.astype(np.float32) / 255.0

        if mask is not None:
            if mask.shape[:2] != img.shape[:2]:
                m_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                m_resized = mask
            thresh_f *= m_resized.astype(np.float32)

        # Smooth specular boundaries
        thresh_f = cv2.GaussianBlur(thresh_f, (5, 5), 1.2)
        thresh_3ch = np.repeat(thresh_f[:, :, np.newaxis], 3, axis=2)

        # Specular residual intensity above threshold
        residual = np.maximum(0.0, img.astype(np.float32) - float(self.specular_threshold))
        return residual * thresh_3ch

    def compute_directional_shading_field(
        self,
        img: np.ndarray,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extracts the low-frequency directional illumination field (keylight direction).

        Returns:
            (H, W) float32 normalized luminance shading map.
        """
        sig = sigma or self.shading_sigma
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        ksize = int(sig * 3) | 1
        shading = cv2.GaussianBlur(l_channel, (ksize, ksize), sigmaX=sig, sigmaY=sig)
        return shading

    def harmonize_lighting(
        self,
        original_crop: np.ndarray,
        swapped_crop: np.ndarray,
        mask: Optional[np.ndarray] = None,
        strength: Optional[float] = None,
    ) -> np.ndarray:
        """
        Transfers physical room keylighting, directional shadow gradients, and authentic
        specular glints from original camera face to swapped face.

        Args:
            original_crop: Original aligned face crop from camera.
            swapped_crop: Swapped face crop before lighting harmonization.
            mask: Optional facial mask crop.
            strength: Blending strength override [0.0, 1.0].

        Returns:
            Illumination-harmonized swapped face crop.
        """
        if original_crop is None or swapped_crop is None:
            return swapped_crop

        eff_strength = self.default_strength if strength is None else float(np.clip(strength, 0.0, 1.0))
        if eff_strength <= 0.01:
            return swapped_crop

        h, w = swapped_crop.shape[:2]
        sh, sw = original_crop.shape[:2]
        if (sw, sh) != (w, h):
            orig_matched = cv2.resize(original_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            orig_matched = original_crop

        # 1. Directional Diffuse Shading Field Transfer in LAB space
        tgt_lab = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        shading_src = self.compute_directional_shading_field(orig_matched)
        shading_tgt = self.compute_directional_shading_field(swapped_crop)

        # Ratio of original lighting to target lighting
        shading_ratio = np.clip((shading_src + 1e-3) / (shading_tgt + 1e-3), 0.70, 1.40)

        # Modulate L-channel by directional shading ratio
        l_curr = tgt_lab[:, :, 0]
        l_adapted = l_curr * (1.0 + (shading_ratio - 1.0) * eff_strength)
        tgt_lab[:, :, 0] = np.clip(l_adapted, 0.0, 255.0)

        result_bgr = cv2.cvtColor(tgt_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

        # 2. Specular Highlight Transfer (room keylight glints on skin)
        specular_highlights = self.extract_specular_highlights(orig_matched, mask=mask)
        if np.max(specular_highlights) > 1.0:
            result_bgr = np.clip(result_bgr + specular_highlights * (eff_strength * 0.75), 0.0, 255.0)

        return result_bgr.astype(np.uint8)
