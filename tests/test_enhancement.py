"""
Unit and Integration Tests for Face Enhancement Subsystem.
Tests AdaptiveFidelityEnhancer, FaceEnhancer, and landmark-guided ocular/oral detail restoration.
"""

import numpy as np
import pytest

from src.processing.enhancement import AdaptiveFidelityEnhancer, FaceEnhancer
from src.core.config_loader import SingleModelConfig


def test_adaptive_enhancer_basic():
    enhancer = AdaptiveFidelityEnhancer(default_strength=0.50)
    fake_crop = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)

    enhanced = enhancer.enhance(fake_crop, strength=0.50)
    assert enhanced.shape == fake_crop.shape
    assert enhanced.dtype == np.uint8


def test_adaptive_enhancer_zero_strength():
    enhancer = AdaptiveFidelityEnhancer(default_strength=0.0)
    fake_crop = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)

    enhanced = enhancer.enhance(fake_crop, strength=0.0)
    np.testing.assert_array_equal(enhanced, fake_crop)


def test_adaptive_enhancer_with_landmarks():
    enhancer = AdaptiveFidelityEnhancer(default_strength=0.60)
    fake_crop = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)

    # 5 standard facial landmarks in pixel coordinates
    landmarks = np.array([
        [38.0, 52.0],   # left eye
        [89.0, 52.0],   # right eye
        [64.0, 78.0],   # nose
        [43.0, 102.0],  # left mouth corner
        [85.0, 102.0],  # right mouth corner
    ], dtype=np.float32)

    enhanced = enhancer.enhance(fake_crop, strength=0.60, landmarks=landmarks)
    assert enhanced.shape == fake_crop.shape
    assert enhanced.dtype == np.uint8
    assert not np.array_equal(enhanced, fake_crop)


def test_face_enhancer_fallback():
    cfg = SingleModelConfig(model_path="models/enhancement/non_existent.onnx")
    enhancer = FaceEnhancer(model_config=cfg, default_strength=0.40)

    assert not enhancer.has_neural_model()

    fake_crop = np.full((128, 128, 3), 120, dtype=np.uint8)
    enhanced = enhancer.enhance(fake_crop, strength=0.40)
    assert enhanced.shape == (128, 128, 3)


def test_skin_texture_injection():
    from src.processing.enhancement import inject_original_skin_texture

    orig = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
    swap = np.full((128, 128, 3), 130, dtype=np.uint8)

    injected = inject_original_skin_texture(orig, swap, amount=0.50)
    assert injected.shape == (128, 128, 3)
    assert injected.dtype == np.uint8
    # Should not be identical to flat swap because texture was injected
    assert not np.array_equal(injected, swap)


def test_ocular_catchlight_restoration():
    from src.processing.enhancement import restore_ocular_specular_catchlights

    fake_face = np.full((128, 128, 3), 100, dtype=np.uint8)
    # create eye bright spots
    fake_face[50:54, 36:40] = 210
    fake_face[50:54, 86:90] = 210

    landmarks = np.array([
        [38.0, 52.0],
        [88.0, 52.0],
        [64.0, 75.0],
        [45.0, 100.0],
        [85.0, 100.0],
    ], dtype=np.float32)

    enhanced = restore_ocular_specular_catchlights(fake_face, landmarks, boost=0.60)
    assert enhanced.shape == fake_face.shape
    assert enhanced.dtype == np.uint8


def test_dental_clarity_protection():
    from src.processing.enhancement import protect_dental_clarity

    fake_face = np.full((128, 128, 3), 120, dtype=np.uint8)
    landmarks = np.array([
        [38.0, 52.0],
        [88.0, 52.0],
        [64.0, 75.0],
        [45.0, 100.0],
        [85.0, 100.0],
    ], dtype=np.float32)

    protected = protect_dental_clarity(fake_face, landmarks, clarity_boost=0.50)
    assert protected.shape == fake_face.shape
    assert protected.dtype == np.uint8


def test_gfpgan_neural_enhancement_if_available():
    import os
    model_path = "models/enhancement/face_enhancer.onnx"
    if os.path.isfile(model_path):
        cfg = SingleModelConfig(model_path=model_path, model_type="gfpgan_onnx")
        enhancer = FaceEnhancer(model_config=cfg, default_strength=0.60)
        assert enhancer.has_neural_model()

        fake_crop = np.random.randint(60, 200, (128, 128, 3), dtype=np.uint8)
        # Test 512x512 native super-resolution
        enhanced_512 = enhancer.enhance(fake_crop, strength=0.60, keep_native_resolution=True)
        assert enhanced_512.shape == (512, 512, 3)
        assert enhanced_512.dtype == np.uint8

