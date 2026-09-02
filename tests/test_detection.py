"""
Unit Tests for Face Detection, Landmarks, and Convex Hull Generation.
"""

import pytest
import numpy as np
import cv2

from src.detection.face_detector import FaceData, FaceDetector
from src.detection.face_landmarks import FaceLandmarks, extract_convex_hull_mask, ARCFACE_STANDARD_112


def test_face_data_properties():
    kps = np.zeros((5, 2), dtype=np.float32)
    face = FaceData(bbox=(100, 150, 300, 450), score=0.95, landmarks=kps)
    assert face.width == 200
    assert face.height == 300
    assert face.area == 60000
    assert face.center == (200.0, 300.0)


def test_face_detector_initialization():
    detector = FaceDetector()
    assert detector.is_ready() is True
    # Test on empty image
    assert detector.detect(None) == []
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detector.detect(empty_img) == []


def test_face_landmarks_smoothing():
    kps1 = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]], dtype=np.float32)
    kps2 = np.array([[12, 22], [32, 42], [52, 62], [72, 82], [92, 102]], dtype=np.float32)

    smoothed = FaceLandmarks.smooth_landmarks_ema(kps2, kps1, alpha=0.5)
    expected = (kps2 + kps1) / 2.0
    np.testing.assert_allclose(smoothed, expected, rtol=1e-5)

    # Test keyword arguments with both naming conventions
    smoothed_kw1 = FaceLandmarks.smooth_landmarks_ema(current=kps2, previous=kps1, alpha=0.5)
    np.testing.assert_allclose(smoothed_kw1, expected, rtol=1e-5)

    smoothed_kw2 = FaceLandmarks.smooth_landmarks_ema(current_kps=kps2, prev_kps=kps1, alpha=0.5)
    np.testing.assert_allclose(smoothed_kw2, expected, rtol=1e-5)


def test_convex_hull_mask_generation():
    kps = np.array([[50, 40], [90, 40], [70, 60], [55, 80], [85, 80]], dtype=np.float32)
    mask = extract_convex_hull_mask((128, 128), kps)
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask[60, 70] == 255  # Center region should be inside mask
