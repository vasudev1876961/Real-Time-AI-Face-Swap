"""
Unit and Integration Tests for Phase 12:
Spectacles & Eyewear Preservation, Skin-on-Skin Hand Carving, and Occlusion HUD.
"""

import pytest
import numpy as np
import cv2

from src.processing.spectacles import SpectaclesPreservationEngine
from src.processing.occlusion import (
    OcclusionDetector,
    SkinOnSkinOcclusionEngine,
    TemporalOcclusionStabilizer,
)
from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.core.config_loader import AppConfig, ProcessingConfig
from src.web.app import create_app
from fastapi.testclient import TestClient


def test_spectacles_preservation_bridge_and_rims():
    engine = SpectaclesPreservationEngine(default_strength=0.85)

    # Base skin crop: 128x128 BGR
    crop = np.zeros((128, 128, 3), dtype=np.uint8)
    crop[:, :] = [130, 150, 200]  # Standard peach/tan skin tone

    # Add dark bridge bar connecting eyes (y=48..54, x=54..74)
    crop[48:54, 54:74] = [15, 15, 20]
    # Add dark left and right frame rims around eyes
    cv2.circle(crop, (46, 51), 16, (20, 20, 25), 2)
    cv2.circle(crop, (81, 51), 16, (20, 20, 25), 2)

    matte = engine.detect_spectacles_mask(crop, INSWAPPER_STANDARD_128, strength=0.85)
    assert matte.shape == (128, 128)
    assert matte.dtype == np.float32

    # Verify bridge bar region has elevated matte value
    bridge_mean = np.mean(matte[48:54, 56:72])
    assert bridge_mean > 0.40

    # Test blending onto synthetic face
    swapped = np.full((128, 128, 3), 100, dtype=np.uint8)
    composite, spec_matte = engine.preserve_spectacles_on_swapped(crop, swapped, INSWAPPER_STANDARD_128)
    assert composite.shape == (128, 128, 3)
    assert composite.dtype == np.uint8
    # The dark bridge should be transferred onto swapped crop
    assert np.mean(composite[48:54, 56:72]) < np.mean(swapped[48:54, 56:72])


def test_spectacles_lens_glare_preservation():
    engine = SpectaclesPreservationEngine(default_strength=0.90)

    crop = np.zeros((128, 128, 3), dtype=np.uint8)
    crop[:, :] = [130, 150, 200]

    # Paint specular glint on left lens (high luminance, low saturation)
    cv2.circle(crop, (44, 48), 5, (250, 252, 255), -1)

    matte = engine.detect_spectacles_mask(crop, INSWAPPER_STANDARD_128)
    assert matte.shape == (128, 128)
    # Lens glare area should be preserved
    assert np.mean(matte[45:51, 41:47]) > 0.30


def test_skin_on_skin_hand_detection():
    hand_engine = SkinOnSkinOcclusionEngine(sensitivity=0.75)

    # Base skin crop: BGR matching biological skin
    crop = np.zeros((128, 128, 3), dtype=np.uint8)
    crop[:, :] = [130, 150, 200]

    base_mask = np.ones((128, 128), dtype=np.float32)

    # Clean skin with zero edges should yield very low hand occlusion
    clean_matte = hand_engine.detect_hand_occlusion(crop, base_mask)
    assert np.mean(clean_matte) < 0.15

    # Simulate finger placed across lower face (horizontal finger bar with edge gradient step)
    # The finger is also skin-colored, but with distinct edge contours and slight shadow
    crop[75:90, 30:100] = [110, 130, 185]  # Slightly darker skin tone
    cv2.line(crop, (30, 75), (100, 75), (70, 85, 120), 2)  # Shadow penumbra
    cv2.line(crop, (30, 90), (100, 90), (70, 85, 120), 2)  # Edge step

    hand_matte = hand_engine.detect_hand_occlusion(crop, base_mask)
    assert hand_matte.shape == (128, 128)
    # Finger edge region should be flagged as skin-on-skin occlusion
    assert np.mean(hand_matte[72:92, 40:90]) > 0.20


def test_feature_protection_override_by_glasses():
    detector = OcclusionDetector(sensitivity=0.60, enable_spectacles=True)

    crop = np.zeros((128, 128, 3), dtype=np.uint8)
    crop[:, :] = [130, 150, 200]

    # Dark spectacles frame passing directly through eye region
    crop[48:54, 30:100] = [10, 10, 15]

    base_mask = np.ones((128, 128), dtype=np.float32)
    matte = detector.detect_occlusion_mask(crop, base_mask, INSWAPPER_STANDARD_128)

    # The glasses bar across eyes should NOT be zeroed out by eye feature protection
    assert np.mean(matte[48:54, 50:80]) > 0.40


def test_occlusion_detector_integrated_phase12():
    detector = OcclusionDetector(
        sensitivity=0.60,
        enable_spectacles=True,
        spectacles_strength=0.80,
        enable_hand_carving=True,
        hand_carving_strength=0.70,
    )

    crop = np.zeros((128, 128, 3), dtype=np.uint8)
    crop[:, :] = [130, 150, 200]

    # Add both foreign object and spectacles bridge
    crop[48:54, 54:74] = [10, 10, 15]  # Spectacles bridge
    crop[90:120, 40:90] = [20, 20, 30]  # Foreign cup/hand

    base_mask = np.ones((128, 128), dtype=np.float32)

    matte = detector.detect_occlusion_mask(crop, base_mask, INSWAPPER_STANDARD_128, track_id=1)
    assert matte.shape == (128, 128)
    assert np.max(matte) > 0.60

    # Refine mask
    refined = detector.refine_mask_with_occlusion(base_mask, crop, INSWAPPER_STANDARD_128, track_id=1)
    assert refined.shape == (128, 128)
    # Occluded areas should have reduced swap weights
    assert np.mean(refined[90:120, 40:90]) < 0.40

    # Check metrics
    metrics = detector.get_last_metrics()
    assert "mean_occlusion" in metrics
    assert "spectacles_active" in metrics
    assert "hand_active" in metrics
    assert metrics["spectacles_active"] is True


def test_occlusion_hud_visualization():
    detector = OcclusionDetector()

    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    matte = np.zeros((128, 128), dtype=np.float32)
    matte[40:90, 40:90] = 0.8  # Occlusion box

    vis_full = detector.visualize_occlusion_hud(frame, matte)
    assert vis_full.shape == frame.shape
    assert not np.array_equal(vis_full, frame)

    # Test with crop_rect (x, y, w, h)
    vis_roi = detector.visualize_occlusion_hud(frame, matte, crop_rect=(50, 50, 100, 100))
    assert vis_roi.shape == frame.shape
    # Outside crop_rect should remain unchanged
    assert np.array_equal(vis_roi[0:40, 0:40], frame[0:40, 0:40])
    # Inside crop_rect should have neon tint
    assert not np.array_equal(vis_roi[50:150, 50:150], frame[50:150, 50:150])


def test_web_api_phase12_config_update():
    from unittest.mock import patch
    from src.camera.camera_manager import CameraManager

    with patch.object(CameraManager, "start", return_value=False):
        app = create_app()
        with TestClient(app) as client:
            # Test getting config includes Phase 12 keys
            res = client.get("/api/pipeline/config")
    assert res.status_code == 200
    cfg = res.json()
    assert "enable_spectacles_preservation" in cfg
    assert "spectacles_preservation_strength" in cfg
    assert "enable_hand_occlusion" in cfg
    assert "hand_occlusion_strength" in cfg
    assert "visualize_occlusion_hud" in cfg

    # Test updating via /api/pipeline/config
    update_payload = {
        "spectacles_preservation_strength": 0.88,
        "hand_occlusion_strength": 0.72,
        "enable_spectacles_preservation": True,
        "enable_hand_occlusion": True,
        "visualize_occlusion_hud": True,
    }
    update_res = client.post("/api/pipeline/config", json=update_payload)
    assert update_res.status_code == 200
    data = update_res.json()
    assert data["success"] is True

    # Test dedicated /api/settings/occlusion endpoint
    occl_res = client.post(
        "/api/settings/occlusion",
        params={
            "sensitivity": 0.65,
            "enable_spectacles": True,
            "spectacles_strength": 0.90,
            "enable_hand_carving": True,
            "hand_strength": 0.75,
            "visualize_hud": True,
        }
    )
    assert occl_res.status_code == 200
    occl_data = occl_res.json()
    assert occl_data["success"] is True
    assert occl_data["occlusion"]["spectacles_strength"] == 0.90
    assert occl_data["occlusion"]["hand_carving_strength"] == 0.75
    assert occl_data["occlusion"]["visualize_hud"] is True
