"""
Automated Test Suite for Phase 10:
High-Performance Memory Buffer Pooling, Turbo Spatial Transform Caching,
and Interactive Web Studio Live Streaming Engine.
"""

import os
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.optimization.buffer_pool import FrameBufferPool, get_buffer_pool
from src.optimization.turbo_pipeline import TurboSpatialOptimizer
from src.web.app import create_app, get_web_runner
from src.detection.face_landmarks import INSWAPPER_STANDARD_128


# =========================================================================
# 1. Memory Buffer Pool Tests
# =========================================================================

def test_frame_buffer_pool_acquire_and_release():
    pool = FrameBufferPool(max_buffers_per_spec=4)
    shape = (720, 1280, 3)

    # Initial acquire (miss)
    buf1 = pool.acquire(shape, dtype=np.uint8)
    assert buf1.shape == shape
    assert buf1.dtype == np.uint8

    stats1 = pool.get_stats()
    assert stats1["misses"] == 1
    assert stats1["hits"] == 0

    # Release back to pool
    pool.release(buf1)
    stats2 = pool.get_stats()
    assert stats2["total_idle_buffers"] == 1

    # Re-acquire (hit)
    buf2 = pool.acquire(shape, dtype=np.uint8)
    assert buf2.shape == shape
    stats3 = pool.get_stats()
    assert stats3["hits"] == 1
    assert stats3["hit_ratio_percent"] == 50.0

    pool.clear()
    assert pool.get_stats()["total_idle_buffers"] == 0


def test_frame_buffer_pool_scoped_context():
    pool = FrameBufferPool()
    shape = (128, 128, 3)

    with pool.scoped_buffer(shape, dtype=np.float32, zero_fill=True) as b:
        assert b.shape == shape
        assert b.dtype == np.float32
        assert np.all(b == 0)

    # After exiting scope, buffer is returned to pool
    assert pool.get_stats()["total_idle_buffers"] == 1


# =========================================================================
# 2. Turbo Spatial Optimizer Tests
# =========================================================================

def test_turbo_spatial_optimizer_caching_and_displacement():
    optimizer = TurboSpatialOptimizer(jitter_threshold_px=1.5, max_cache_frames=10)
    base_lms = INSWAPPER_STANDARD_128.copy()

    # Frame 1: Miss
    m1, inv1, was_cached = optimizer.get_or_compute_transform(base_lms, track_id=1, crop_size=128)
    assert not was_cached
    assert m1.shape == (2, 3)
    assert inv1.shape == (2, 3)

    # Frame 2: Sub-pixel stationary displacement (0.5px) -> Cache Hit!
    sub_lms = base_lms + 0.3
    m2, inv2, was_cached2 = optimizer.get_or_compute_transform(sub_lms, track_id=1, crop_size=128)
    assert was_cached2
    assert np.allclose(m1, m2)

    # Frame 3: Significant displacement (5.0px shift) -> Cache Miss (fresh recalculation)
    shifted_lms = base_lms + 5.0
    m3, inv3, was_cached3 = optimizer.get_or_compute_transform(shifted_lms, track_id=1, crop_size=128)
    assert not was_cached3

    # Reset
    optimizer.reset(track_id=1)
    stats = optimizer.get_stats()
    assert stats["cached_tracks"] == 0


# =========================================================================
# 3. Web Studio FastAPI Server & REST API Tests
# =========================================================================

@pytest.fixture(scope="module")
def web_client():
    from unittest.mock import patch
    from src.camera.camera_manager import CameraManager
    with patch.object(CameraManager, "start", return_value=False):
        app = create_app()
        with TestClient(app) as client:
            yield client


def test_web_studio_root_html(web_client):
    response = web_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Real-Time AI Face Swap" in response.text


def test_web_studio_targets_api(web_client):
    response = web_client.get("/api/targets")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "targets" in data
    assert isinstance(data["targets"], list)


def test_web_studio_telemetry_api(web_client):
    response = web_client.get("/api/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "fps" in data
    assert "target_fps" in data
    assert "hardware" in data


def test_web_studio_config_update(web_client):
    # Update pipeline parameters
    update_payload = {
        "enhancement_strength": 0.65,
        "eye_realism_strength": 0.85,
        "mouth_preservation_strength": 0.75,
        "color_grading_preset": "cinematic",
    }
    response = web_client.post("/api/pipeline/config", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    # Verify retrieval
    get_res = web_client.get("/api/pipeline/config")
    assert get_res.status_code == 200
    cfg_data = get_res.json()
    assert cfg_data["enhancement_strength"] == 0.65
    assert cfg_data["color_grading_preset"] == "cinematic"


def test_web_studio_source_toggle(web_client):
    response = web_client.post("/api/source/toggle")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "is_synthetic" in data


def test_web_studio_capture_frame(web_client):
    response = web_client.post("/api/capture")
    # Might return 200 if frame is ready, or 503 if pipeline is still starting
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert data["filename"].endswith(".jpg")
