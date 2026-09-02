"""
Unit Tests for 5-Point Umeyama Face Alignment and Inverse Warping.
"""

import pytest
import numpy as np
import cv2

from src.alignment.face_alignment import (
    FaceAligner,
    get_affine_transform,
    align_face_crop,
    warp_face_back,
)


def test_affine_transform_computation():
    src_kps = np.array([[100, 120], [180, 120], [140, 160], [110, 200], [170, 200]], dtype=np.float32)
    mat, inv_mat = get_affine_transform(src_kps, target_size=(112, 112))
    assert mat.shape == (2, 3)
    assert inv_mat.shape == (2, 3)


def test_align_and_warp_back():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 60, (255, 200, 180), -1)

    src_kps = np.array([[290, 210], [350, 210], [320, 240], [300, 270], [340, 270]], dtype=np.float32)
    aligned, mat, inv_mat = align_face_crop(frame, src_kps, crop_size=(112, 112))

    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8

    warped_back = warp_face_back(aligned, inv_mat, (480, 640))
    assert warped_back.shape == (480, 640, 3)
    assert warped_back.dtype == np.uint8
