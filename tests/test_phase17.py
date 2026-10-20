"""
Unit and Integration Tests for Phase 17:
Multi-Subject Identity Binding Matrix, Virtual Portrait Studio Lighting,
Temporal Identity Latent Coherence, and Output Showcase Generator.
"""

import os
import pytest
import numpy as np
import cv2

from src.core.config_loader import AppConfig, ModelsConfig, TargetsConfig, load_all_configs
from src.detection.face_detector import FaceData
from src.targets.target_loader import TargetFace
from src.targets.target_manager import TargetManager
from src.processing.multi_identity import MultiIdentityMatrixManager
from src.processing.studio_lighting import StudioLightingBackdropEngine
from src.processing.temporal_coherence import TemporalIdentitySmoother
from src.processing.showcase_generator import ShowcaseGenerator
from src.pipeline.realtime_pipeline import RealTimePipeline


@pytest.fixture
def dummy_face_a():
    landmarks = np.array([
        [40.0, 45.0], [88.0, 45.0], [64.0, 65.0], [45.0, 90.0], [83.0, 90.0]
    ], dtype=np.float32)
    return FaceData(
        bbox=(20, 20, 108, 108),
        score=0.98,
        landmarks=landmarks,
        track_id=1,
    )


@pytest.fixture
def dummy_face_b():
    landmarks = np.array([
        [140.0, 45.0], [188.0, 45.0], [164.0, 65.0], [145.0, 90.0], [183.0, 90.0]
    ], dtype=np.float32)
    return FaceData(
        bbox=(120, 20, 208, 108),
        score=0.95,
        landmarks=landmarks,
        track_id=2,
    )


from src.targets.target_loader import TargetFace, TargetMetadata


@pytest.fixture
def mock_target_manager():
    tm = TargetManager()
    e1 = np.random.randn(512).astype(np.float32)
    e1 = e1 / np.linalg.norm(e1)
    meta1 = TargetMetadata(person_id="target_alpha", display_name="Alpha Actor", category="actors")
    t1 = TargetFace(
        target_id="target_alpha",
        display_name="Alpha Actor",
        category="actors",
        reference_image_path="",
        reference_image=np.zeros((64, 64, 3), dtype=np.uint8),
        embedding=e1,
        metadata=meta1,
    )

    e2 = np.random.randn(512).astype(np.float32)
    e2 = e2 / np.linalg.norm(e2)
    meta2 = TargetMetadata(person_id="target_beta", display_name="Beta Actor", category="actors")
    t2 = TargetFace(
        target_id="target_beta",
        display_name="Beta Actor",
        category="actors",
        reference_image_path="",
        reference_image=np.zeros((64, 64, 3), dtype=np.uint8),
        embedding=e2,
        metadata=meta2,
    )
    tm._targets = {"target_alpha": t1, "target_beta": t2}
    tm._selected_target_id = "target_alpha"
    return tm


# =============================================================================
# 1. MultiIdentityMatrixManager Tests
# =============================================================================

def test_multi_identity_primary_mode(dummy_face_a, dummy_face_b, mock_target_manager):
    mgr = MultiIdentityMatrixManager(mock_target_manager)
    mgr.set_config(mode="primary")
    default_target = mock_target_manager.get_selected_target()

    bindings = mgr.resolve_bindings([dummy_face_a, dummy_face_b], default_target)
    assert len(bindings) == 2
    assert bindings[0][1] == default_target
    assert bindings[1][1] is None  # Non-primary faces untouched


def test_multi_identity_all_mode(dummy_face_a, dummy_face_b, mock_target_manager):
    mgr = MultiIdentityMatrixManager(mock_target_manager)
    mgr.set_config(mode="all")
    default_target = mock_target_manager.get_selected_target()

    bindings = mgr.resolve_bindings([dummy_face_a, dummy_face_b], default_target)
    assert len(bindings) == 2
    assert bindings[0][1] == default_target
    assert bindings[1][1] == default_target


def test_multi_identity_spatial_lr_mode(dummy_face_a, dummy_face_b, mock_target_manager):
    mgr = MultiIdentityMatrixManager(mock_target_manager)
    mgr.set_config(mode="spatial_lr", spatial_target_ids=["target_alpha", "target_beta"])
    default_target = mock_target_manager.get_selected_target()

    # face_a (x=20) is to the left of face_b (x=120)
    bindings = mgr.resolve_bindings([dummy_face_b, dummy_face_a], default_target)
    binding_dict = {f.track_id: t.person_id for f, t in bindings if t}

    assert binding_dict[1] == "target_alpha"
    assert binding_dict[2] == "target_beta"


def test_multi_identity_custom_matrix_and_persistence(dummy_face_a, dummy_face_b, mock_target_manager):
    mgr = MultiIdentityMatrixManager(mock_target_manager)
    mgr.set_track_mapping(track_id=1, target_id="target_beta")
    mgr.set_config(mode="custom_matrix")

    default_target = mock_target_manager.get_selected_target()
    bindings = mgr.resolve_bindings([dummy_face_a, dummy_face_b], default_target)
    binding_dict = {f.track_id: t.person_id for f, t in bindings if t}

    assert binding_dict[1] == "target_beta"
    assert binding_dict[2] == "target_alpha"  # Falls back to default

    # Verify telemetry
    telemetry = mgr.get_telemetry()
    assert telemetry["mode"] == "custom_matrix"
    assert telemetry["custom_mappings_count"] == 1

    # Reset
    mgr.reset()
    assert len(mgr._track_memory) == 0


# =============================================================================
# 2. StudioLightingBackdropEngine Tests
# =============================================================================

def test_studio_lighting_rim_and_color_presets():
    engine = StudioLightingBackdropEngine(
        default_rim_intensity=0.60,
        default_rim_thickness=4,
        default_rim_color="warm_gold",
        default_backdrop_dim=0.20,
    )

    frame = np.full((128, 128, 3), 100, dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.float32)
    # Circle in center
    cv2.circle(mask, (64, 64), 35, 1.0, -1)

    out, metrics = engine.apply_studio_lighting(frame, mask=mask)
    assert metrics["applied"] is True
    assert metrics["rim_intensity"] == 0.60
    assert metrics["rim_color"] == "warm_gold"

    # Edge rim should be brighter than baseline 100
    # Center and edge should have light added
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_studio_lighting_backdrop_attenuation():
    engine = StudioLightingBackdropEngine(
        default_rim_intensity=0.0,
        default_backdrop_dim=0.30,
    )
    frame = np.full((128, 128, 3), 100, dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.float32)
    # Pure foreground inside circle, pure background outside
    cv2.circle(mask, (64, 64), 30, 1.0, -1)

    out, metrics = engine.apply_studio_lighting(frame, mask=mask)
    # Outside mask (0,0) should be dimmed from 100 to approx 70
    assert out[0, 0, 0] < 100
    # Inside mask (64, 64) should stay close to 100
    assert out[64, 64, 0] >= 95


def test_studio_lighting_ambient_adaptive_and_telemetry():
    engine = StudioLightingBackdropEngine(default_rim_color="ambient_adaptive")
    frame = np.full((128, 128, 3), 150, dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.float32)
    cv2.circle(mask, (64, 64), 30, 1.0, -1)

    out, metrics = engine.apply_studio_lighting(frame, mask=mask)
    assert out is not None

    telemetry = engine.get_telemetry()
    assert "warm_gold" in telemetry["available_colors"]
    assert "ambient_adaptive" in telemetry["available_colors"]


# =============================================================================
# 3. TemporalIdentitySmoother Tests
# =============================================================================

def test_temporal_identity_smoother_geodesic_embedding():
    smoother = TemporalIdentitySmoother(default_smoothing_factor=0.80)
    emb1 = np.random.randn(512).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)

    # First frame
    s1 = smoother.smooth_embedding(emb1, track_id=1)
    np.testing.assert_allclose(np.linalg.norm(s1), 1.0, atol=1e-5)

    # Second frame with slight perturbation
    emb2 = emb1 + 0.1 * np.random.randn(512).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)

    s2 = smoother.smooth_embedding(emb2, track_id=1)
    np.testing.assert_allclose(np.linalg.norm(s2), 1.0, atol=1e-5)
    # s2 should be closer to s1 than emb2 is to s1
    dist_smooth = np.linalg.norm(s2 - s1)
    dist_raw = np.linalg.norm(emb2 - s1)
    assert dist_smooth < dist_raw


def test_temporal_identity_smoother_crop_stabilization():
    smoother = TemporalIdentitySmoother(default_jitter_suppression=0.50)
    c1 = np.full((64, 64, 3), 120, dtype=np.uint8)
    lms = np.array([[20.0, 20.0], [40.0, 20.0], [30.0, 30.0], [25.0, 45.0], [35.0, 45.0]], dtype=np.float32)

    # Frame 1
    out1 = smoother.stabilize_crop(c1, landmarks=lms, track_id=1)
    assert np.array_equal(out1, c1)

    # Frame 2 with noise
    c2 = np.full((64, 64, 3), 140, dtype=np.uint8)
    out2 = smoother.stabilize_crop(c2, landmarks=lms, track_id=1)
    # Because velocity was 0, it should blend c1 and c2
    assert 120 < out2[0, 0, 0] < 140

    telemetry = smoother.get_telemetry()
    assert telemetry["tracked_crops_count"] == 1
    assert telemetry["smoothing_factor"] == 0.80 or telemetry["smoothing_factor"] == 0.85


# =============================================================================
# 4. RealTimePipeline Phase 17 Integration Tests
# =============================================================================

def test_pipeline_phase17_configuration_setters():
    a_cfg, m_cfg, t_cfg = load_all_configs()
    pipeline = RealTimePipeline(a_cfg, m_cfg, t_cfg)

    # Configure Phase 17 Studio Lighting
    pipeline.set_studio_lighting_config(
        enabled=True,
        rim_intensity=0.65,
        rim_thickness=4,
        rim_color="neon_magenta",
        backdrop_dim=0.25,
    )
    assert pipeline.app_config.processing.enable_studio_lighting is True
    assert pipeline.studio_lighting_engine.rim_intensity == 0.65
    assert pipeline.studio_lighting_engine.rim_color_name == "neon_magenta"

    # Configure Phase 17 Temporal Coherence
    pipeline.set_temporal_coherence_config(
        enabled=True,
        smoothing_factor=0.90,
        jitter_suppression=0.50,
    )
    assert pipeline.app_config.processing.enable_temporal_coherence is True
    assert pipeline.temporal_identity_smoother.smoothing_factor == 0.90

    # Configure Phase 17 Multi-Identity Matrix
    pipeline.set_multi_identity_matrix_config(
        mode="spatial_lr",
        spatial_target_ids=["prabhas", "rekha"],
    )
    assert pipeline.app_config.processing.multi_identity_mode == "spatial_lr"
    assert "prabhas" in pipeline.multi_identity_manager.spatial_target_ids

    # Run frame pass and verify Phase 17 telemetry output
    canvas = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.ellipse(canvas, (320, 240), (80, 110), 0, 0, 360, (180, 150, 130), -1)

    result = pipeline.process_frame(canvas)
    assert result is not None
    assert "phase17" in result.metrics_summary
    p17_tel = result.metrics_summary["phase17"]
    assert "studio_lighting" in p17_tel
    assert "multi_identity" in p17_tel
    assert "temporal_coherence" in p17_tel


# =============================================================================
# 5. ShowcaseGenerator Integration Test
# =============================================================================

def test_showcase_generator_passes(tmp_path):
    a_cfg, m_cfg, t_cfg = load_all_configs()
    pipeline = RealTimePipeline(a_cfg, m_cfg, t_cfg)
    generator = ShowcaseGenerator(pipeline=pipeline)

    canvas = np.full((360, 480, 3), 40, dtype=np.uint8)
    cv2.ellipse(canvas, (240, 180), (60, 80), 0, 0, 360, (180, 150, 130), -1)

    out_grid = str(tmp_path / "test_grid.jpg")
    out_split = str(tmp_path / "test_split.jpg")

    grid = generator.generate_comparison_grid(canvas, target_id="prabhas", output_path=out_grid)
    assert grid is not None
    assert os.path.isfile(out_grid)

    split = generator.generate_split_showcase(canvas, target_id="prabhas", output_path=out_split)
    assert split is not None
    assert os.path.isfile(out_split)


# =============================================================================
# 6. Web Studio API Phase 17 Endpoint Tests
# =============================================================================

def test_web_studio_phase17_endpoints():
    from fastapi.testclient import TestClient
    from src.web.app import create_app

    app = create_app()
    client = TestClient(app)

    # 1. Test POST /api/settings/studio_lighting
    resp = client.post("/api/settings/studio_lighting", json={
        "enabled": True,
        "rim_intensity": 0.70,
        "rim_color": "cool_cyan",
        "backdrop_dim": 0.20,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["studio_lighting"]["rim_intensity"] == 0.70
    assert data["studio_lighting"]["rim_color"] == "cool_cyan"

    # 2. Test POST /api/settings/multi_identity
    resp = client.post("/api/settings/multi_identity", json={
        "mode": "spatial_lr",
        "spatial_target_ids": ["prabhas", "rekha"],
    })
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # 3. Test POST /api/settings/temporal_coherence
    resp = client.post("/api/settings/temporal_coherence", json={
        "enabled": True,
        "smoothing_factor": 0.88,
        "jitter_suppression": 0.45,
    })
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # 4. Test GET /api/telemetry/phase17
    resp = client.get("/api/telemetry/phase17")
    assert resp.status_code == 200
    data = resp.json()
    assert "studio_lighting" in data
    assert "multi_identity" in data
    assert "temporal_coherence" in data
