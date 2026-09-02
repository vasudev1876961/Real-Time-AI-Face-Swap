"""
Face Alignment using Standard 5-Point Umeyama Affine Matrix Formulation.
"""

from typing import Tuple, Optional, Union
import cv2
import numpy as np

from src.detection.face_landmarks import (
    ARCFACE_STANDARD_112,
    ARCFACE_STANDARD_512,
    INSWAPPER_STANDARD_128,
)
from src.utils.logger import get_logger

logger = get_logger("FaceAlignment")


def get_affine_transform(
    src_kps: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the 2x3 affine similarity transform matrix from source 5 keypoints
    to standard ArcFace/INSwapper coordinates, and its inverse transform matrix.

    Returns:
        (affine_matrix_2x3, inverse_affine_matrix_2x3)
    """
    if target_size == (128, 128):
        dst_kps = INSWAPPER_STANDARD_128
    elif target_size == (512, 512):
        dst_kps = ARCFACE_STANDARD_512
    elif target_size == (112, 112):
        dst_kps = ARCFACE_STANDARD_112
    else:
        scale_x = target_size[0] / 112.0
        scale_y = target_size[1] / 112.0
        dst_kps = ARCFACE_STANDARD_112.copy()
        dst_kps[:, 0] *= scale_x
        dst_kps[:, 1] *= scale_y

    src_pts = np.float32(src_kps[:5]).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_kps).reshape(-1, 1, 2)

    transform_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)

    if transform_matrix is None:
        src_tri = np.float32([src_kps[0], src_kps[1], src_kps[2]])
        dst_tri = np.float32([dst_kps[0], dst_kps[1], dst_kps[2]])
        transform_matrix = cv2.getAffineTransform(src_tri, dst_tri)

    inv_matrix = cv2.invertAffineTransform(transform_matrix)
    return transform_matrix, inv_matrix


def align_face_crop(
    frame: np.ndarray,
    landmarks: np.ndarray,
    crop_size: Tuple[int, int] = (112, 112),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns and crops a face from the source frame into standard dimensions.
    Returns (aligned_crop, affine_matrix, inverse_affine_matrix).
    """
    if frame is None or landmarks is None:
        empty = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)
        ident = np.eye(2, 3, dtype=np.float32)
        return empty, ident, ident

    mat, inv_mat = get_affine_transform(landmarks, target_size=crop_size)
    aligned_crop = cv2.warpAffine(
        frame,
        mat,
        (crop_size[0], crop_size[1]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned_crop, mat, inv_mat


def paste_aligned_crop(
    target_frame: np.ndarray,
    aligned_crop: np.ndarray,
    inverse_matrix: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Warps an aligned face crop back onto the target frame using the inverse affine matrix.
    """
    h, w = target_frame.shape[:2]
    crop_h, crop_w = aligned_crop.shape[:2]

    # Warp crop to full frame dimensions
    warped_face = cv2.warpAffine(
        aligned_crop,
        inverse_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    if mask is not None:
        if mask.shape[:2] == (crop_h, crop_w):
            warped_mask = cv2.warpAffine(
                mask,
                inverse_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            warped_mask = mask

        if len(warped_mask.shape) == 2:
            alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        else:
            alpha = warped_mask.astype(np.float32) / 255.0

        blended = (warped_face.astype(np.float32) * alpha + target_frame.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        return blended

    return warped_face


class FaceAligner:
    """Stateful aligner instance for pipeline integration."""

    def __init__(self, target_crop_size: Tuple[int, int] = (128, 128)):
        self.target_crop_size = target_crop_size

    def align(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = crop_size or self.target_crop_size
        return align_face_crop(frame, landmarks, crop_size=size)

    def inverse_warp(
        self,
        target_frame: np.ndarray,
        aligned_crop: np.ndarray,
        inv_mat: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return paste_aligned_crop(target_frame, aligned_crop, inv_mat, mask=mask)


def warp_face_back(
    arg1: np.ndarray,
    arg2: np.ndarray,
    arg3: Union[np.ndarray, Tuple[int, int]],
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Flexible wrapper supporting:
      warp_face_back(aligned_crop, inv_mat, (h, w))
      warp_face_back(target_frame, aligned_crop, inv_mat)
    """
    if isinstance(arg3, (tuple, list)):
        aligned_crop = arg1
        inv_mat = arg2
        target_h, target_w = int(arg3[0]), int(arg3[1])
        target_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        return paste_aligned_crop(target_frame, aligned_crop, inv_mat, mask=mask)
    else:
        target_frame = arg1
        aligned_crop = arg2
        inv_mat = arg3
        return paste_aligned_crop(target_frame, aligned_crop, inv_mat, mask=mask)
