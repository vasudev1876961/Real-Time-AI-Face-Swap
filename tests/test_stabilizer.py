"""
Unit Tests for Temporal Motion and Anti-Jitter Stabilization.
"""

import pytest
import numpy as np
import cv2

from src.processing.stabilizer import TemporalMotionStabilizer


def test_stabilizer_affine_smoothing():
    stabilizer = TemporalMotionStabilizer(motion_alpha=0.50, velocity_threshold=5.0)

    # Frame 1: Base affine identity matrix with translation
    m1 = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 60.0]], dtype=np.float32)
    inv1 = np.array([[1.0, 0.0, -50.0], [0.0, 1.0, -60.0]], dtype=np.float32)

    sm1, sinv1 = stabilizer.stabilize_transform(m1, inv1, track_id=1)
    np.testing.assert_allclose(sm1, m1)

    # Frame 2: Micro-jitter (translation shifted by 0.8 pixels, within threshold)
    m2 = np.array([[1.0, 0.0, 50.8], [0.0, 1.0, 60.6]], dtype=np.float32)
    inv2 = np.array([[1.0, 0.0, -50.8], [0.0, 1.0, -60.6]], dtype=np.float32)

    sm2, sinv2 = stabilizer.stabilize_transform(m2, inv2, track_id=1)
    # Output should be smoothed between m1 and m2 (translation ~ 50.4, 60.3)
    assert 50.1 < sm2[0, 2] < 50.7
    assert 60.1 < sm2[1, 2] < 60.5


def test_stabilizer_fast_motion_response():
    stabilizer = TemporalMotionStabilizer(motion_alpha=0.50, velocity_threshold=4.0)

    m1 = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 60.0]], dtype=np.float32)
    inv1 = np.array([[1.0, 0.0, -50.0], [0.0, 1.0, -60.0]], dtype=np.float32)
    stabilizer.stabilize_transform(m1, inv1, track_id=1)

    # Frame 2: Large sudden head movement (shifted by 40 pixels)
    m_fast = np.array([[1.0, 0.0, 90.0], [0.0, 1.0, 60.0]], dtype=np.float32)
    inv_fast = np.array([[1.0, 0.0, -90.0], [0.0, 1.0, -60.0]], dtype=np.float32)

    sm_fast, _ = stabilizer.stabilize_transform(m_fast, inv_fast, track_id=1)
    # Fast movement should have dynamic alpha close to 1.0 (minimal drag)
    assert sm_fast[0, 2] > 80.0


def test_stabilizer_luminance_clamping():
    stabilizer = TemporalMotionStabilizer(luminance_alpha=0.70)

    # Frame 1: Stable skin crop
    crop1 = np.full((64, 64, 3), 120, dtype=np.uint8)
    stabilizer.stabilize_luminance(crop1, track_id=1)

    # Frame 2: Sudden flash spike (intensity jumps to 220)
    crop_flash = np.full((64, 64, 3), 220, dtype=np.uint8)
    clamped = stabilizer.stabilize_luminance(crop_flash, track_id=1)

    # Output should have clamped luminance lower than the flash spike
    assert np.mean(clamped) < np.mean(crop_flash)


def test_stabilizer_multi_track_reset():
    stabilizer = TemporalMotionStabilizer()
    m = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float32)
    inv = np.array([[1.0, 0.0, -10.0], [0.0, 1.0, -20.0]], dtype=np.float32)

    stabilizer.stabilize_transform(m, inv, track_id=1)
    stabilizer.stabilize_transform(m, inv, track_id=2)
    assert 1 in stabilizer._states
    assert 2 in stabilizer._states

    stabilizer.reset(track_id=1)
    assert 1 not in stabilizer._states
    assert 2 in stabilizer._states

    stabilizer.reset()
    assert len(stabilizer._states) == 0
