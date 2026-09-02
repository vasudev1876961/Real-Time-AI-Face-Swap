"""
Seamless Face Blending and Mask Compositing Subsystem.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from src.alignment.face_alignment import warp_face_back
from src.core.config_loader import ProcessingConfig
from src.utils.logger import get_logger

logger = get_logger("FaceBlender")


def blend_face_into_frame(
    original_frame: np.ndarray,
    swapped_crop: np.ndarray,
    mask_crop: np.ndarray,
    inv_affine_matrix: np.ndarray,
    method: str = "alpha",
    strength: float = 1.0,
    seamless_mode: str = "NORMAL_CLONE",
) -> np.ndarray:
    """
    Composites the swapped face crop into the full-resolution video frame.

    Args:
        original_frame: (H, W, 3) original camera frame.
        swapped_crop: (h, w, 3) swapped face crop.
        mask_crop: (h, w) float32 [0.0, 1.0] soft feathered mask.
        inv_affine_matrix: 2x3 matrix mapping crop -> full frame.
        method: "alpha" or "seamless_clone".
        strength: Blending strength [0.0, 1.0].
        seamless_mode: "NORMAL_CLONE" or "MIXED_CLONE".

    Returns:
        (H, W, 3) blended frame uint8.
    """
    h, w = original_frame.shape[:2]

    # 1. Warp swapped crop into full frame coordinates
    warped_swap = cv2.warpAffine(
        swapped_crop,
        inv_affine_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # 2. Warp float32 mask into full frame coordinates
    warped_mask = cv2.warpAffine(
        mask_crop,
        inv_affine_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Multiply mask by strength
    if strength < 1.0:
        warped_mask = warped_mask * strength

    if method == "seamless_clone":
        try:
            # Prepare uint8 mask for Poisson clone
            binary_mask = (warped_mask > 0.3).astype(np.uint8) * 255
            # Find center of mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(c)
                center = (x + cw // 2, y + ch // 2)
                # Clone mode flag
                mode_flag = cv2.MIXED_CLONE if seamless_mode == "MIXED_CLONE" else cv2.NORMAL_CLONE
                cloned = cv2.seamlessClone(warped_swap, original_frame, binary_mask, center, mode_flag)
                return cloned
        except Exception as e:
            logger.debug(f"Seamless cloning failed, falling back to alpha blend: {e}")

    # Default high-performance alpha blending (maintains 30+ FPS)
    mask_3ch = np.repeat(warped_mask[:, :, np.newaxis], 3, axis=2)
    blended = (warped_swap.astype(np.float32) * mask_3ch + original_frame.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(blended, 0, 255).astype(np.uint8)


class FaceBlender:
    """Class wrapper for compositing swapped faces into camera frames."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()

    def blend(
        self,
        original_frame: Optional[np.ndarray] = None,
        swapped_crop: Optional[np.ndarray] = None,
        mask_crop: Optional[np.ndarray] = None,
        inv_matrix: Optional[np.ndarray] = None,
        *,
        target_frame: Optional[np.ndarray] = None,
        inverse_matrix: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        method: Optional[str] = None,
        strength: Optional[float] = None,
    ) -> np.ndarray:
        frame = original_frame if original_frame is not None else target_frame
        inv_mat = inv_matrix if inv_matrix is not None else inverse_matrix
        m_crop = mask_crop if mask_crop is not None else mask
        blend_m = method or self.config.blending_method
        blend_s = strength if strength is not None else self.config.blending_strength

        if frame is None or swapped_crop is None or m_crop is None or inv_mat is None:
            raise ValueError("FaceBlender.blend requires frame, swapped_crop, mask, and inverse_matrix.")

        return blend_face_into_frame(
            original_frame=frame,
            swapped_crop=swapped_crop,
            mask_crop=m_crop,
            inv_affine_matrix=inv_mat,
            method=blend_m,
            strength=blend_s,
            seamless_mode=self.config.seamless_clone_mode,
        )

