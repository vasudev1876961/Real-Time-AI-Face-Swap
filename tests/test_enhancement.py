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
