"""
Unit Tests for Occlusion-Aware Facial Masking.
"""

import pytest
import numpy as np
import cv2

from src.processing.occlusion import OcclusionDetector
from src.processing.mask import FaceMaskGenerator
from src.detection.face_landmarks import INSWAPPER_STANDARD_128


def test_occlusion_detector_clean_skin():
    detector = OcclusionDetector(sensitivity=0.50)

    # Synthetic warm human skin crop (BGR: ~180, 200, 220 in BGR is pinkish skin)
    # Natural skin in BGR: B=130, G=150, R=200
    skin_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    skin_crop[:, :] = [130, 150, 200]

    base_mask = np.ones((128, 128), dtype=np.float32)

    # Detect occlusion on clean skin
    occl_matte = detector.detect_occlusion_mask(skin_crop, base_mask, INSWAPPER_STANDARD_128)
    assert occl_matte.shape == (128, 128)
    # Clean skin should have minimal to no occlusion
    assert np.mean(occl_matte) < 0.15


def test_occlusion_detector_with_foreign_object():
    detector = OcclusionDetector(sensitivity=0.60)

    # Base skin crop
    skin_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    skin_crop[:, :] = [130, 150, 200]

    # Paint a dark foreground object across the lower half (e.g. black microphone or coffee cup)
    # centered around [70:110, 40:90]
    skin_crop[70:110, 40:90] = [15, 15, 20]

    base_mask = np.ones((128, 128), dtype=np.float32)

    occl_matte = detector.detect_occlusion_mask(skin_crop, base_mask, INSWAPPER_STANDARD_128)
    assert occl_matte.shape == (128, 128)
    # Occlusion should be detected strongly at the object location
    assert np.mean(occl_matte[75:105, 45:85]) > 0.50

    # Refine mask
    refined = detector.refine_mask_with_occlusion(base_mask, skin_crop, INSWAPPER_STANDARD_128)
    assert refined.shape == (128, 128)
    # Mask value at object should be significantly attenuated
    assert np.mean(refined[75:105, 45:85]) < 0.50


def test_mask_generator_occlusion_integration():
    generator = FaceMaskGenerator()
    detector = OcclusionDetector(sensitivity=0.60)

    skin_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    skin_crop[:, :] = [130, 150, 200]
    skin_crop[60:90, 40:80] = [10, 10, 15]  # Foreign object

    mask = generator.generate_mask(
        crop_shape=(128, 128),
        landmarks=INSWAPPER_STANDARD_128,
        aligned_crop=skin_crop,
        occlusion_detector=detector,
        track_id=1,
    )
    assert mask.shape == (128, 128)
    assert mask.dtype in [np.uint8, np.float32]


def test_facial_feature_protection_mask():
    detector = OcclusionDetector()
    prot_mask = detector.compute_feature_protection_mask((128, 128), INSWAPPER_STANDARD_128)
    assert prot_mask.shape == (128, 128)
    assert prot_mask.dtype == np.float32
    # Protection should be high near eyes (around y=51, x=46 and x=81)
    eye_l_y, eye_l_x = int(INSWAPPER_STANDARD_128[0, 1]), int(INSWAPPER_STANDARD_128[0, 0])
    assert prot_mask[eye_l_y, eye_l_x] > 0.50


def test_temporal_occlusion_stabilizer():
    from src.processing.occlusion import TemporalOcclusionStabilizer

    stabilizer = TemporalOcclusionStabilizer(alpha=0.50)
    m1 = np.full((128, 128), 0.2, dtype=np.float32)
    m2 = np.full((128, 128), 0.8, dtype=np.float32)

    res1 = stabilizer.smooth(m1, track_id=1)
    np.testing.assert_allclose(res1, m1)

    res2 = stabilizer.smooth(m2, track_id=1)
    # With alpha=0.5, 0.5*0.8 + 0.5*0.2 = 0.5
    np.testing.assert_allclose(res2, np.full((128, 128), 0.5, dtype=np.float32))

    # Reset test
    stabilizer.reset(track_id=1)
    res3 = stabilizer.smooth(m2, track_id=1)
    np.testing.assert_allclose(res3, m2)
