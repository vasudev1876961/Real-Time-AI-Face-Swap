"""
Unit Tests for Masking, Color Correction, Blending, and Post-Processing.
"""

import pytest
import numpy as np
import cv2

from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.processing.mask import (
    create_face_mask,
    FaceMaskGenerator,
    feather_mask_distance_transform,
    get_anatomical_facial_contour,
)
from src.processing.color_correction import (
    reinhard_color_transfer,
    gain_color_match,
    apply_color_correction,
    TemporalColorStabilizer,
)
from src.processing.blending import blend_face_into_frame, FaceBlender, compute_crop_roi
from src.processing.postprocess import postprocess_frame, apply_unsharp_mask


def test_face_mask_creation_all_types():
    mask_types = ["smooth_hull", "pose_adaptive", "distance_transform", "convex_hull", "elliptical"]
    for mtype in mask_types:
        mask = create_face_mask(
            crop_shape=(128, 128),
            mask_type=mtype,
            blur_kernel_size=15,
            feather_factor=0.6,
            landmarks=INSWAPPER_STANDARD_128,
            yaw=15.0,
            pitch=-5.0,
        )
        assert mask.shape == (128, 128), f"Shape mismatch for {mtype}"
        assert mask.dtype == np.float32, f"Dtype mismatch for {mtype}"
        assert mask.min() >= 0.0, f"Negative values in mask for {mtype}"
        assert mask.max() <= 1.0, f"Values > 1.0 in mask for {mtype}"
        # Core center should be high confidence replacement
        assert mask[64, 64] > 0.8, f"Center of mask not active for {mtype}"
        # Outer corners should be 0.0
        assert mask[0, 0] == 0.0, f"Corner of mask not zero for {mtype}"


def test_distance_transform_feathering():
    binary = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(binary, (50, 50), 30, 255, -1)

    feathered = feather_mask_distance_transform(binary, radius=8.0, falloff="smoothstep")
    assert feathered.shape == (100, 100)
    assert feathered.min() == 0.0
    assert feathered.max() == 1.0
    # Inside center must be 1.0
    assert feathered[50, 50] == 1.0
    # Far outside must be 0.0
    assert feathered[5, 5] == 0.0
    # Transition boundary must have smooth gradient
    assert 0.0 < feathered[50, 20] < 1.0


def test_mask_generator_caching():
    generator = FaceMaskGenerator()
    generator.clear_cache()

    m1 = generator.generate_mask(crop_shape=(128, 128), mask_type_override="smooth_hull")
    assert len(generator._cache) == 1

    # Second call with same parameters should hit cache
    m2 = generator.generate_mask(crop_shape=(128, 128), mask_type_override="smooth_hull")
    assert np.array_equal(m1, m2)
    assert len(generator._cache) == 1

    # Call with pose change in pose_adaptive mode
    m3 = generator.generate_mask(crop_shape=(128, 128), mask_type_override="pose_adaptive", yaw=25.0)
    assert len(generator._cache) == 2


def test_reinhard_color_transfer_mask_weighted():
    src = np.full((128, 128, 3), (120, 150, 200), dtype=np.uint8)
    tgt = np.full((128, 128, 3), (80, 90, 100), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[30:90, 30:90] = 1.0

    corrected = reinhard_color_transfer(src, tgt, blend_ratio=1.0, mask=mask)
    assert corrected.shape == tgt.shape
    assert corrected.dtype == np.uint8
    assert np.mean(corrected) > np.mean(tgt)


def test_temporal_color_stabilizer():
    stabilizer = TemporalColorStabilizer(alpha=0.5)
    s1, o1 = stabilizer.update(np.array([1.0, 1.0, 1.0]), np.array([10.0, 10.0, 10.0]))
    assert np.allclose(s1, [1.0, 1.0, 1.0])
    assert np.allclose(o1, [10.0, 10.0, 10.0])

    s2, o2 = stabilizer.update(np.array([2.0, 2.0, 2.0]), np.array([20.0, 20.0, 20.0]))
    # EMA with alpha=0.5: 0.5*2.0 + 0.5*1.0 = 1.5
    assert np.allclose(s2, [1.5, 1.5, 1.5])
    assert np.allclose(o2, [15.0, 15.0, 15.0])


def test_face_blender_roi_and_full():
    orig = np.zeros((480, 640, 3), dtype=np.uint8)
    swap_crop = np.full((128, 128, 3), 255, dtype=np.uint8)
    mask = np.ones((128, 128), dtype=np.float32)
    inv_mat = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]], dtype=np.float32)

    # ROI Blending (default)
    blended_roi = blend_face_into_frame(
        original_frame=orig,
        swapped_crop=swap_crop,
        mask_crop=mask,
        inv_affine_matrix=inv_mat,
        use_roi=True,
    )
    assert blended_roi.shape == (480, 640, 3)
    assert blended_roi[150, 150, 0] > 200

    # Full frame blending
    blended_full = blend_face_into_frame(
        original_frame=orig,
        swapped_crop=swap_crop,
        mask_crop=mask,
        inv_affine_matrix=inv_mat,
        use_roi=False,
    )
    assert blended_full.shape == (480, 640, 3)
    assert blended_full[150, 150, 0] > 200

    # Test FaceBlender wrapper class
    blender = FaceBlender()
    b1 = blender.blend(original_frame=orig, swapped_crop=swap_crop, mask_crop=mask, inv_matrix=inv_mat)
    assert b1.shape == (480, 640, 3)


def test_postprocess_sharpening():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 200
    sharpened = postprocess_frame(img, sharpen_amount=0.5)
    assert sharpened.shape == img.shape
    assert sharpened.dtype == np.uint8
