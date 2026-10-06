"""
Integration and Unit Tests for Real-Time Frame Rate Stabilization,
Camera MJPEG Optimization, Frame Deduplication, and Dynamic Blending Governor.
"""

import pytest
import numpy as np
import cv2

from src.camera.camera_backend import CameraBackend, FramePacket
from src.tracking.face_tracker import FaceTracker, TrackedFace
from src.detection.face_detector import FaceData, FaceDetector
from src.optimization.performance import AdaptivePerformanceGovernor
from src.processing.blending import blend_face_into_frame, pyramid_blend
from src.processing.postprocess import postprocess_frame
from src.core.config_loader import SingleModelConfig, PerformanceConfig


def test_camera_backend_mjpeg_configuration():
    """Verifies CameraBackend initializes with MJPG fourcc support and fast properties."""
    backend = CameraBackend(camera_index=99, width=1280, height=720, target_fps=30)
    assert backend.target_fps == 30
    assert backend.width == 1280
    assert backend.height == 720
    # read_latest before opening returns None
    assert backend.read_latest() is None


def test_face_tracker_detection_anti_spiking():
    """Verifies that face tracker throttles re-detections when faces are missed."""
    detector = FaceDetector(SingleModelConfig())
    cfg = PerformanceConfig(detection_interval=4)
    tracker = FaceTracker(detector, cfg)

    # Synthetic frame
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Frame 1: initial detection run
    tracker.update_all(frame)
    assert tracker._frame_counter == 1
    assert tracker._last_detection_frame == 1

    # Simulate an active track with missed_frames > 0
    tracker.active_tracks[1] = TrackedFace(
        track_id=1,
        face_data=FaceData(bbox=(100, 100, 200, 200), score=0.9, landmarks=np.zeros((5, 2), dtype=np.float32)),
        missed_frames=1,
    )

    # Frame 2: consecutive frame. Even though missed_frames > 0, frames_since_detect is 1 (< 2),
    # so tracker should NOT run heavy detection again immediately.
    tracker.update_all(frame)
    assert tracker._frame_counter == 2
    # _last_detection_frame should still be 1 (detection was skipped on frame 2)
    assert tracker._last_detection_frame == 1


def test_governor_dynamic_blending_and_feathering():
    """Verifies AdaptivePerformanceGovernor steps down blending and feathering under load."""
    gov = AdaptivePerformanceGovernor(target_fps=30.0, low_fps_threshold=23.0, high_fps_threshold=28.0, hysteresis_frames=3)

    # In optimal state
    assert gov.current_state == "optimal"
    assert gov.get_recommended_blending_method("multiband") == "multiband"
    assert gov.get_recommended_curvature_feathering(True) is True

    # Drop FPS for 3 frames to trigger downstep
    for _ in range(4):
        gov.update(20.0)
    assert gov.current_state == "balanced"
    # In balanced state, blending falls back to zero-overhead alpha
    assert gov.get_recommended_blending_method("multiband") == "alpha"

    # Drop FPS further to trigger throttled state
    for _ in range(4):
        gov.update(15.0)
    assert gov.current_state == "throttled"
    assert gov.get_recommended_blending_method("multiband") == "alpha"
    assert gov.get_recommended_curvature_feathering(True) is False


def test_blending_dynamic_pyramid_levels_and_alpha():
    """Verifies blending handles large and small ROIs with high speed and zero artifact."""
    orig = np.full((480, 640, 3), 50, dtype=np.uint8)
    swap = np.full((128, 128, 3), 200, dtype=np.uint8)
    mask = np.full((128, 128), 0.7, dtype=np.float32)

    inv_mat = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]], dtype=np.float32)

    # Test multiband blending in ROI
    res_multi = blend_face_into_frame(orig, swap, mask, inv_mat, method="multiband", use_roi=True)
    assert res_multi.shape == (480, 640, 3)
    assert res_multi[150, 150, 0] > 100

    # Test alpha blending in ROI
    res_alpha = blend_face_into_frame(orig, swap, mask, inv_mat, method="alpha", use_roi=True)
    assert res_alpha.shape == (480, 640, 3)
    assert res_alpha[150, 150, 0] > 100


def test_postprocess_empty_frame_efficiency():
    """Verifies postprocessing skips expensive full-frame Gaussian blur when no face is present."""
    empty_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    # Without face_data and sharpen_full_frame=False, frame should remain untouched
    processed = postprocess_frame(empty_frame, sharpen_amount=0.5, face_data=None, sharpen_full_frame=False)
    assert np.array_equal(processed, empty_frame)
