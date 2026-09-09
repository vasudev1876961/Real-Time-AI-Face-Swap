"""
Seamless Face Blending and Mask Compositing Subsystem with Accelerated ROI-Bounded Warping.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from src.alignment.face_alignment import warp_face_back
from src.core.config_loader import ProcessingConfig
from src.utils.logger import get_logger

logger = get_logger("FaceBlender")


def compute_crop_roi(
    inv_affine_matrix: np.ndarray,
    crop_shape: Tuple[int, int],
    frame_shape: Tuple[int, int],
    margin_ratio: float = 0.15,
) -> Tuple[int, int, int, int, np.ndarray]:
    """
    Computes the tight bounding box (ROI) in frame coordinates corresponding to the warped crop,
    along with the local inverse affine matrix translated to ROI-relative coordinates.

    Returns:
        (x1, y1, x2, y2, local_inv_affine_matrix)
    """
    ch, cw = crop_shape[:2]
    fh, fw = frame_shape[:2]

    # Transform 4 crop corners: (0,0), (cw,0), (cw,ch), (0,ch)
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(cw), 0.0, 1.0],
            [float(cw), float(ch), 1.0],
            [0.0, float(ch), 1.0],
        ],
        dtype=np.float32,
    ).T  # 3x4

    warped_corners = inv_affine_matrix @ corners  # 2x4
    min_x, max_x = float(np.min(warped_corners[0, :])), float(np.max(warped_corners[0, :]))
    min_y, max_y = float(np.min(warped_corners[1, :])), float(np.max(warped_corners[1, :]))

    span_x = max(max_x - min_x, 10.0)
    span_y = max(max_y - min_y, 10.0)

    margin_x = span_x * margin_ratio
    margin_y = span_y * margin_ratio

    x1 = max(0, int(np.floor(min_x - margin_x)))
    y1 = max(0, int(np.floor(min_y - margin_y)))
    x2 = min(fw, int(np.ceil(max_x + margin_x)))
    y2 = min(fh, int(np.ceil(max_y + margin_y)))

    # Local translation matrix
    local_inv_mat = inv_affine_matrix.copy()
    local_inv_mat[0, 2] -= float(x1)
    local_inv_mat[1, 2] -= float(y1)

    return x1, y1, x2, y2, local_inv_mat


def blend_face_into_frame(
    original_frame: np.ndarray,
    swapped_crop: np.ndarray,
    mask_crop: np.ndarray,
    inv_affine_matrix: np.ndarray,
    method: str = "alpha",
    strength: float = 1.0,
    seamless_mode: str = "NORMAL_CLONE",
    use_roi: bool = True,
) -> np.ndarray:
    """
    Composites the swapped face crop into the full-resolution video frame.
    Uses accelerated ROI-bounded warping by default to provide 5x-10x speedup.

    Args:
        original_frame: (H, W, 3) original camera frame.
        swapped_crop: (h, w, 3) swapped face crop.
        mask_crop: (h, w) float32 [0.0, 1.0] soft feathered mask.
        inv_affine_matrix: 2x3 matrix mapping crop -> full frame.
        method: "alpha" or "seamless_clone".
        strength: Blending strength [0.0, 1.0].
        seamless_mode: "NORMAL_CLONE" or "MIXED_CLONE".
        use_roi: Whether to use accelerated ROI-bounded warping.

    Returns:
        (H, W, 3) blended frame uint8.
    """
    fh, fw = original_frame.shape[:2]

    # Fast ROI-Bounded Warping & Blending
    if use_roi:
        x1, y1, x2, y2, local_mat = compute_crop_roi(
            inv_affine_matrix,
            swapped_crop.shape,
            original_frame.shape,
        )

        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            return original_frame

        # Warp swapped crop and mask strictly within local ROI bounding box
        warped_swap_roi = cv2.warpAffine(
            swapped_crop,
            local_mat,
            (rw, rh),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        warped_mask_roi = cv2.warpAffine(
            mask_crop,
            local_mat,
            (rw, rh),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        if strength < 1.0:
            warped_mask_roi = warped_mask_roi * strength

        frame_roi = original_frame[y1:y2, x1:x2]

        if method == "seamless_clone":
            try:
                binary_mask_roi = (warped_mask_roi > 0.25).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    cx, cy, cw_c, ch_c = cv2.boundingRect(c)
                    center = (cx + cw_c // 2, cy + ch_c // 2)
                    mode_flag = cv2.MIXED_CLONE if seamless_mode == "MIXED_CLONE" else cv2.NORMAL_CLONE
                    cloned_roi = cv2.seamlessClone(warped_swap_roi, frame_roi, binary_mask_roi, center, mode_flag)
                    output_frame = original_frame.copy()
                    output_frame[y1:y2, x1:x2] = cloned_roi
                    return output_frame
            except Exception as e:
                logger.debug(f"ROI seamless cloning failed, falling back to alpha blend: {e}")

        # High-performance vectorized alpha blending in local ROI
        mask_3ch = warped_mask_roi[:, :, np.newaxis]
        blended_roi = (warped_swap_roi.astype(np.float32) * mask_3ch + frame_roi.astype(np.float32) * (1.0 - mask_3ch))
        output_frame = original_frame.copy()
        output_frame[y1:y2, x1:x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)
        return output_frame

    # Full-frame fallback path
    warped_swap = cv2.warpAffine(
        swapped_crop,
        inv_affine_matrix,
        (fw, fh),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )

    warped_mask = cv2.warpAffine(
        mask_crop,
        inv_affine_matrix,
        (fw, fh),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )

    if strength < 1.0:
        warped_mask = warped_mask * strength

    if method == "seamless_clone":
        try:
            binary_mask = (warped_mask > 0.3).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(c)
                center = (x + cw // 2, y + ch // 2)
                mode_flag = cv2.MIXED_CLONE if seamless_mode == "MIXED_CLONE" else cv2.NORMAL_CLONE
                cloned = cv2.seamlessClone(warped_swap, original_frame, binary_mask, center, mode_flag)
                return cloned
        except Exception as e:
            logger.debug(f"Seamless cloning failed, falling back to alpha blend: {e}")

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
        use_roi: Optional[bool] = None,
    ) -> np.ndarray:
        frame = original_frame if original_frame is not None else target_frame
        inv_mat = inv_matrix if inv_matrix is not None else inverse_matrix
        m_crop = mask_crop if mask_crop is not None else mask
        blend_m = method or getattr(self.config, "blending_method", "alpha")
        blend_s = strength if strength is not None else getattr(self.config, "blending_strength", 1.0)
        roi_flag = use_roi if use_roi is not None else getattr(self.config, "use_roi_blending", True)

        if frame is None or swapped_crop is None or m_crop is None or inv_mat is None:
            raise ValueError("FaceBlender.blend requires frame, swapped_crop, mask, and inverse_matrix.")

        return blend_face_into_frame(
            original_frame=frame,
            swapped_crop=swapped_crop,
            mask_crop=m_crop,
            inv_affine_matrix=inv_mat,
            method=blend_m,
            strength=blend_s,
            seamless_mode=getattr(self.config, "seamless_clone_mode", "NORMAL_CLONE"),
            use_roi=roi_flag,
        )
