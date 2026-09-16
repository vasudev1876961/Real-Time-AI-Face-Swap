"""
Unit Tests for ROI-Bounded Inverse Warping, Parity, and Seamless Compositing.
"""

import pytest
import numpy as np
import cv2

from src.processing.blending import (
    blend_face_into_frame,
    FaceBlender,
    compute_crop_roi,
)


def test_compute_crop_roi():
    # 2x3 identity with translation (100, 150)
    inv_mat = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 150.0]], dtype=np.float32)
    crop_shape = (128, 128, 3)
    frame_shape = (720, 1280, 3)

    x1, y1, x2, y2, local_mat = compute_crop_roi(inv_mat, crop_shape, frame_shape, margin_ratio=0.1)
    assert 0 <= x1 < x2 <= 1280
    assert 0 <= y1 < y2 <= 720
    # Inside bounds
    assert x1 <= 100
    assert y1 <= 150
    assert x2 >= 100 + 128
    assert y2 >= 150 + 128
    # Local matrix translation should be relative to (x1, y1)
    assert local_mat[0, 2] == 100.0 - float(x1)
    assert local_mat[1, 2] == 150.0 - float(y1)


def test_roi_and_full_frame_parity():
    orig = np.full((720, 1280, 3), 40, dtype=np.uint8)
    swap = np.full((128, 128, 3), 220, dtype=np.uint8)
    mask = np.full((128, 128), 0.8, dtype=np.float32)

    inv_mat = np.array([[1.5, 0.0, 400.0], [0.0, 1.5, 200.0]], dtype=np.float32)

    res_roi = blend_face_into_frame(orig, swap, mask, inv_mat, method="alpha", use_roi=True)
    res_full = blend_face_into_frame(orig, swap, mask, inv_mat, method="alpha", use_roi=False)

    assert res_roi.shape == (720, 1280, 3)
    assert res_full.shape == (720, 1280, 3)

    # Core swapped face pixels should match closely between ROI and full frame
    diff = np.abs(res_roi.astype(np.int32) - res_full.astype(np.int32))
    assert np.max(diff) <= 2


def test_multiband_pyramid_blending():
    from src.processing.blending import pyramid_blend

    fg = np.full((100, 100, 3), 200, dtype=np.uint8)
    bg = np.full((100, 100, 3), 50, dtype=np.uint8)
    mask = np.full((100, 100), 0.5, dtype=np.float32)

    blended = pyramid_blend(fg, bg, mask, levels=3)
    assert blended.shape == (100, 100, 3)
    assert blended.dtype == np.uint8
    # Blended value should be smoothly centered around 125
    assert 100 <= blended[50, 50, 0] <= 150


def test_512_super_resolution_scaling_blending():
    orig = np.full((720, 1280, 3), 40, dtype=np.uint8)
    # 512x512 super-resolution face crop
    swap_512 = np.full((512, 512, 3), 220, dtype=np.uint8)
    mask_128 = np.full((128, 128), 1.0, dtype=np.float32)

    # 128x128 standard inverse matrix
    inv_mat_128 = np.array([[1.0, 0.0, 300.0], [0.0, 1.0, 200.0]], dtype=np.float32)

    res = blend_face_into_frame(orig, swap_512, mask_128, inv_mat_128, method="multiband", use_roi=True)
    assert res.shape == (720, 1280, 3)
    assert res.dtype == np.uint8

    # Center of face (300 + 64, 200 + 64) = (364, 264) should have face pixels
    assert res[264, 364, 0] > 150



def test_face_at_frame_boundary():
    # Face placed partly outside frame boundaries
    orig = np.zeros((480, 640, 3), dtype=np.uint8)
    swap = np.full((128, 128, 3), 180, dtype=np.uint8)
    mask = np.ones((128, 128), dtype=np.float32)

    # Placed at (-40, -40)
    inv_mat_top_left = np.array([[1.0, 0.0, -40.0], [0.0, 1.0, -40.0]], dtype=np.float32)
    res = blend_face_into_frame(orig, swap, mask, inv_mat_top_left, use_roi=True)
    assert res.shape == (480, 640, 3)
    assert res[10, 10, 0] > 100

    # Placed at (600, 450)
    inv_mat_bottom_right = np.array([[1.0, 0.0, 600.0], [0.0, 1.0, 450.0]], dtype=np.float32)
    res2 = blend_face_into_frame(orig, swap, mask, inv_mat_bottom_right, use_roi=True)
    assert res2.shape == (480, 640, 3)
    assert res2[460, 620, 0] > 100
