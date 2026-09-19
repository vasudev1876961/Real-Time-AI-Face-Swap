"""
Unit Tests for Facial Landmark Detection, Geometric Utilities, and Alignment Templates.
"""

import pytest
import numpy as np

from src.detection.face_landmarks import (
    ARCFACE_STANDARD_112,
    INSWAPPER_STANDARD_128,
    ARCFACE_STANDARD_512,
    FaceLandmarks,
    extract_convex_hull_mask,
    generate_face_mask_from_landmarks,
)
from src.landmarks.landmark_detector import LandmarkDetector


def test_standard_landmark_templates():
    assert ARCFACE_STANDARD_112.shape == (5, 2)
    assert INSWAPPER_STANDARD_128.shape == (5, 2)
    assert ARCFACE_STANDARD_512.shape == (5, 2)

    # 128 template has 8px horizontal offset from 112 template
    np.testing.assert_allclose(INSWAPPER_STANDARD_128[:, 0], ARCFACE_STANDARD_112[:, 0] + 8.0)


def test_interocular_distance():
    iod = FaceLandmarks.calculate_interocular_distance(ARCFACE_STANDARD_112)
    expected = float(np.linalg.norm(ARCFACE_STANDARD_112[1] - ARCFACE_STANDARD_112[0]))
    assert abs(iod - expected) < 1e-4

    # None and empty handling
    assert FaceLandmarks.calculate_interocular_distance(None) == 0.0
    assert FaceLandmarks.calculate_interocular_distance(np.zeros((1, 2))) == 0.0


def test_head_pose_angles():
    # Frontal template should have yaw near 0
    yaw, pitch = FaceLandmarks.calculate_head_pose_angles(INSWAPPER_STANDARD_128)
    assert abs(yaw) < 15.0
    assert abs(pitch) < 30.0

    # None handling
    assert FaceLandmarks.calculate_head_pose_angles(None) == (0.0, 0.0)


def test_generate_face_mask_from_landmarks():
    mask = generate_face_mask_from_landmarks((128, 128), INSWAPPER_STANDARD_128)
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    assert mask.max() == 255


def test_landmark_detector_adapter():
    detector = LandmarkDetector()
    assert detector is not None

    # Test smoothing via adapter
    kps1 = np.ones((5, 2), dtype=np.float32) * 10.0
    kps2 = np.ones((5, 2), dtype=np.float32) * 20.0
    smoothed = detector.smooth_landmarks(kps2, kps1, alpha=0.5)
    np.testing.assert_allclose(smoothed, np.full((5, 2), 15.0))
