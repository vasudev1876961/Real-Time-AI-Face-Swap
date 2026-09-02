"""
Unit Tests for Masking, Color Correction, Blending, and Post-Processing.
"""

import pytest
import numpy as np
import cv2

from src.processing.mask import create_face_mask, FaceMaskGenerator
from src.processing.color_correction import reinhard_color_transfer, gain_color_match, apply_color_correction
from src.processing.blending import blend_face_into_frame, FaceBlender
from src.processing.postprocess import postprocess_frame, apply_unsharp_mask


def test_face_mask_creation():
    mask = create_face_mask(crop_shape=(128, 128), mask_type="elliptical", blur_kernel_size=15)
    assert mask.shape == (128, 128)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_reinhard_color_transfer():
    src = np.full((100, 100, 3), (120, 150, 200), dtype=np.uint8)
    tgt = np.full((100, 100, 3), (80, 90, 100), dtype=np.uint8)

    corrected = reinhard_color_transfer(src, tgt, blend_ratio=1.0)
    assert corrected.shape == tgt.shape
    assert corrected.dtype == np.uint8
    # Should shift towards source color
    assert np.mean(corrected) > np.mean(tgt)


def test_face_blender():
    orig = np.zeros((480, 640, 3), dtype=np.uint8)
    swap_crop = np.full((128, 128, 3), 255, dtype=np.uint8)
    mask = np.ones((128, 128), dtype=np.float32)
    inv_mat = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]], dtype=np.float32)

    blended = blend_face_into_frame(
        original_frame=orig,
        swapped_crop=swap_crop,
        mask_crop=mask,
        inv_affine_matrix=inv_mat,
        method="alpha",
        strength=1.0,
    )
    assert blended.shape == (480, 640, 3)
    assert blended[150, 150, 0] > 200

    # Test FaceBlender wrapper class
    blender = FaceBlender()
    b1 = blender.blend(original_frame=orig, swapped_crop=swap_crop, mask_crop=mask, inv_matrix=inv_mat)
    assert b1.shape == (480, 640, 3)
    b2 = blender.blend(target_frame=orig, swapped_crop=swap_crop, mask=mask, inverse_matrix=inv_mat)
    assert b2.shape == (480, 640, 3)


def test_postprocess_sharpening():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 200
    sharpened = postprocess_frame(img, sharpen_amount=0.5)
    assert sharpened.shape == img.shape
    assert sharpened.dtype == np.uint8
