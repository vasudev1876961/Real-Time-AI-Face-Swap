"""
Mouth Cavity & Dental Fidelity Preservation Subsystem.
Prevents synthetic deepfake blurring, 'rubber teeth', and oral cavity discoloration
by detecting open mouth geometry and softly blending crisp authentic teeth, lips highlights,
and oral cavity contrast from the subject camera feed.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.utils.logger import get_logger

logger = get_logger("MouthPreservation")


class OralCavityPreserver:
    """
    Detects and preserves realistic dental structures and natural oral cavity depth.
    Works seamlessly on both standard 128x128 crops and 512x512 neural super-resolution crops.
    """

    def __init__(self, default_strength: float = 0.65):
        self.default_strength = default_strength

    def compute_mouth_region_mask(
        self,
        crop_shape: Tuple[int, int],
        landmarks: Optional[np.ndarray] = None,
        feather: int = 7,
    ) -> np.ndarray:
        """
        Generates a smooth elliptical mask centered over the inter-labial mouth opening.

        Args:
            crop_shape: (H, W) of the crop.
            landmarks: 5 standard ArcFace landmarks [[x, y], ...].
            feather: Feather blur radius.

        Returns:
            (H, W) float32 mask [0.0, 1.0].
        """
        h, w = crop_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        if landmarks is not None and len(landmarks) >= 5:
            pts = np.array(landmarks, dtype=np.float32).copy()
            if pts.max() <= 1.05:
                pts[:, 0] *= w
                pts[:, 1] *= h
            elif (pts[:, 0].max() <= 135.0 and w > 200) or (pts[:, 1].max() <= 135.0 and h > 200):
                # Scale from 128x128 to native resolution
                pts[:, 0] *= float(w) / 128.0
                pts[:, 1] *= float(h) / 128.0

            # Mouth corners: index 3 (left), index 4 (right)
            m_left = pts[3]
            m_right = pts[4]
            center_x = int((m_left[0] + m_right[0]) / 2.0)
            center_y = int((m_left[1] + m_right[1]) / 2.0 + (h * 0.015))
            mouth_w = float(np.linalg.norm(m_right - m_left))
            radius_x = int(max(6, mouth_w * 0.40))
            radius_y = int(max(4, mouth_w * 0.22))
        else:
            # Fallback to standard 128x128 geometry scaled to (h, w)
            center_x = int(w * 0.50)
            center_y = int(h * 0.78)
            radius_x = int(w * 0.18)
            radius_y = int(h * 0.08)

        # Draw soft inner-mouth ellipse
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (radius_x, radius_y),
            0.0,
            0.0,
            360.0,
            1.0,
            -1,
        )

        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather * 0.6)
        return np.clip(mask, 0.0, 1.0)

    def is_mouth_open(
        self,
        original_crop: np.ndarray,
        mouth_mask: np.ndarray,
        threshold: float = 18.0,
    ) -> Tuple[bool, float]:
        """
        Determines if the mouth is open by measuring luminance variance and edge energy
        inside the oral cavity region (teeth create high frequency contrast).

        Returns:
            (is_open, openness_score [0.0, 1.0])
        """
        if original_crop is None or mouth_mask is None:
            return False, 0.0

        h, w = original_crop.shape[:2]
        m = mouth_mask if mouth_mask.shape[:2] == (h, w) else cv2.resize(mouth_mask, (w, h))

        # Weight gray channel by mouth mask
        gray = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        inner_pixels = gray[m > 0.4]

        if inner_pixels.size < 20:
            return False, 0.0

        std_dev = float(np.std(inner_pixels))
        # High std dev inside mouth indicates contrast between dark cavity and bright teeth
        openness = float(np.clip((std_dev - threshold) / 25.0, 0.0, 1.0))
        return (openness > 0.10, openness)

    def preserve_mouth_fidelity(
        self,
        original_crop: np.ndarray,
        swapped_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        strength: Optional[float] = None,
    ) -> np.ndarray:
        """
        Preserves authentic dental definition and natural oral cavity contrast.

        Args:
            original_crop: Original aligned human face crop.
            swapped_crop: Neural swapped face crop (128x128 or 512x512).
            landmarks: Face landmarks.
            strength: Blending strength [0.0, 1.0].

        Returns:
            Enhanced face crop with authentic teeth and oral cavity.
        """
        if original_crop is None or swapped_crop is None:
            return swapped_crop

        k_str = self.default_strength if strength is None else float(strength)
        if k_str <= 0.02:
            return swapped_crop

        h, w = swapped_crop.shape[:2]
        orig_h, orig_w = original_crop.shape[:2]

        if (orig_w, orig_h) != (w, h):
            orig_matched = cv2.resize(original_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            orig_matched = original_crop

        mouth_mask = self.compute_mouth_region_mask((h, w), landmarks=landmarks)
        is_open, openness = self.is_mouth_open(orig_matched, mouth_mask)

        if not is_open:
            return swapped_crop

        # Calculate dental-specific weighting: isolate bright teeth and dark cavity
        orig_gray = cv2.cvtColor(orig_matched, cv2.COLOR_BGR2GRAY).astype(np.float32)
        swap_gray = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Teeth are bright pixels (> 120) inside the mouth mask
        dental_energy = np.clip((orig_gray - 100.0) / 70.0, 0.0, 1.0)
        # Oral cavity dark regions (< 60)
        cavity_energy = np.clip((70.0 - orig_gray) / 50.0, 0.0, 1.0)
        feature_mask = np.maximum(dental_energy, cavity_energy)

        # Combine mouth region mask with dental/cavity mask
        effective_mask = mouth_mask * feature_mask * openness * k_str
        effective_mask = np.clip(effective_mask, 0.0, 0.85)  # Cap at 85% to ensure smooth transition

        # 3-channel blend
        m_3ch = effective_mask[:, :, np.newaxis]
        blended = (
            orig_matched.astype(np.float32) * m_3ch
            + swapped_crop.astype(np.float32) * (1.0 - m_3ch)
        )
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)
