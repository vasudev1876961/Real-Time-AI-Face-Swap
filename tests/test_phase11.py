"""
Unit and Integration Tests for Phase 11:
- Zero-Allocation FrameBufferPool & Deep Blending Recycling
- TurboSpatialOptimizer Stationary Caching & Jitter Suppression
- SpeechExpressionTransferEngine (Phoneme Aperture, Smile Energy, Brow Dynamics)
- HardwareEngineAutotuner Session Options & Provider Profiling
- Web Studio Live MP4 Recording Endpoints & Configuration
"""

import os
import time
import pytest
import numpy as np
import cv2

from src.optimization.buffer_pool import FrameBufferPool, get_buffer_pool
from src.optimization.turbo_pipeline import TurboSpatialOptimizer
from src.processing.expression_transfer import SpeechExpressionTransferEngine
from src.core.device import get_device_manager, HardwareEngineAutotuner, get_hardware_autotuner
from src.processing.blending import FaceBlender, blend_face_into_frame
from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.core.config_loader import AppConfig, ModelsConfig, TargetsConfig


def test_frame_buffer_pool_lifecycle():
    pool = FrameBufferPool(max_buffers_per_spec=4)
    shape = (720, 1280, 3)
    dtype = np.uint8

    # Acquire new buffer
    buf1 = pool.acquire(shape, dtype=dtype, zero_fill=True)
    assert buf1.shape == shape
    assert buf1.dtype == dtype
    assert np.all(buf1 == 0)

    # Release back to pool
    pool.release(buf1)
    stats = pool.get_stats()
    assert stats["total_idle_buffers"] == 1
    assert stats["misses"] == 1

    # Acquire again -> should hit cache
    buf2 = pool.acquire(shape, dtype=dtype)
    stats2 = pool.get_stats()
    assert stats2["hits"] == 1
    assert buf2 is buf1  # Reused identical memory address!

    # Scoped buffer context manager
    with pool.scoped_buffer((128, 128, 3), dtype=np.uint8) as scoped_buf:
        assert scoped_buf.shape == (128, 128, 3)
        scoped_buf[0, 0, 0] = 42

    assert pool.get_stats()["total_idle_buffers"] >= 1


def test_turbo_spatial_optimizer_caching():
    optimizer = TurboSpatialOptimizer(jitter_threshold_px=1.5, max_cache_frames=10)
    base_lms = INSWAPPER_STANDARD_128.copy()

    # Pass 1: Fresh computation
    mat1, inv1, was_cached1 = optimizer.get_or_compute_transform(base_lms, track_id=1, crop_size=128)
    assert not was_cached1
    assert mat1.shape == (2, 3)
    assert inv1.shape == (2, 3)

    # Pass 2: Jitter below threshold (e.g. 0.4px shift) -> should return cached matrix!
    jittered_lms = base_lms + np.random.uniform(-0.4, 0.4, size=base_lms.shape).astype(np.float32)
    mat2, inv2, was_cached2 = optimizer.get_or_compute_transform(jittered_lms, track_id=1, crop_size=128)
    assert was_cached2
    assert np.allclose(mat1, mat2)

    # Pass 3: Significant movement (e.g. 5.0px shift) -> should recompute fresh matrix!
    moved_lms = base_lms + 5.0
    mat3, inv3, was_cached3 = optimizer.get_or_compute_transform(moved_lms, track_id=1, crop_size=128)
    assert not was_cached3

    # Reset
    optimizer.reset(track_id=1)
    assert len(optimizer._cache) == 0


def test_speech_expression_transfer_engine():
    engine = SpeechExpressionTransferEngine(default_strength=0.75)
    orig_crop = np.full((128, 128, 3), 160, dtype=np.uint8)
    swap_crop = np.full((128, 128, 3), 120, dtype=np.uint8)

    # Draw simulated speech aperture (dark mouth hole) in orig
    cv2.ellipse(orig_crop, (64, 95), (18, 10), 0, 0, 360, (20, 20, 20), -1)

    # Transfer expressions
    result = engine.transfer_expressions(
        original_crop=orig_crop,
        swapped_crop=swap_crop,
        landmarks=INSWAPPER_STANDARD_128,
        strength=0.75,
        track_id=1,
    )

    assert result.shape == (128, 128, 3)
    assert result.dtype == np.uint8
    # The mouth region in result should reflect speech aperture
    assert not np.array_equal(result, swap_crop)

    # Zero strength returns swapped crop directly
    no_effect = engine.transfer_expressions(orig_crop, swap_crop, strength=0.0)
    assert np.array_equal(no_effect, swap_crop)

    # Reset state
    engine.reset(track_id=1)
    assert 1 not in engine._states


def test_hardware_engine_autotuner():
    autotuner = get_hardware_autotuner()
    assert autotuner is not None

    opts = autotuner.get_optimized_session_options()
    assert opts is not None

    profile = autotuner.profile_and_autotune()
    assert "active_provider" in profile
    assert "available_providers" in profile
    assert "recommendation" in profile
    assert profile["cpu_cores"] >= 1


def test_pooled_face_blender():
    blender = FaceBlender()
    orig = np.full((240, 320, 3), 100, dtype=np.uint8)
    crop = np.full((128, 128, 3), 200, dtype=np.uint8)
    mask = np.full((128, 128), 1.0, dtype=np.float32)
    inv_mat = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 50.0]], dtype=np.float32)

    # Test blending with auto-acquired pool buffer
    out = blender.blend(
        original_frame=orig,
        swapped_crop=crop,
        mask_crop=mask,
        inv_matrix=inv_mat,
        method="alpha",
    )
    assert out.shape == orig.shape
    assert out.dtype == np.uint8
    # ROI should contain swapped values
    assert np.any(out != 100)


def test_web_studio_recording_endpoints(monkeypatch):
    from fastapi.testclient import TestClient
    from src.camera.camera_manager import CameraManager
    from src.web.app import create_app

    monkeypatch.setattr(CameraManager, "start", lambda self: False)

    app = create_app()
    with TestClient(app) as client:
        # Test pipeline config endpoint (including Phase 11 fields)
        cfg_resp = client.get("/api/pipeline/config")
        assert cfg_resp.status_code == 200
        cfg = cfg_resp.json()
        assert "expression_transfer_strength" in cfg
        assert "enable_expression_transfer" in cfg
        assert "enable_turbo_spatial_caching" in cfg

        # Test update pipeline config
        update_resp = client.post(
            "/api/pipeline/config",
            json={
                "expression_transfer_strength": 0.85,
                "enable_expression_transfer": True,
                "enable_turbo_spatial_caching": True,
            },
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["success"]

        # Test recording status endpoint
        status_resp = client.get("/api/recording/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert "is_recording" in status_data

        # Test list recordings endpoint
        rec_list_resp = client.get("/api/recordings")
        assert rec_list_resp.status_code == 200
        assert "recordings" in rec_list_resp.json()
