"""
Unit and Integration Tests for Phase 15:
Volumetric Relighting, 3D Directional Key Light Tracking, Subsurface Scattering (SSS) Warmth,
Identity Morphing with Spherical Linear Interpolation (SLERP), Dual-Target Facial Fusion,
and Interactive Live Split-Screen Comparison Modes.
"""

import numpy as np
import cv2
import pytest

from src.processing.identity_morph import (
    slerp,
    smoothstep,
    IdentityMorphEngine,
)
from src.processing.volumetric_relighting import VolumetricRelightingEngine
from src.processing.split_screen import SplitScreenRenderer
from src.targets.target_loader import TargetFace, TargetMetadata
from src.pipeline.realtime_pipeline import RealTimePipeline
from src.core.config_loader import AppConfig, ModelsConfig, TargetsConfig, load_all_configs


def create_mock_target(target_id: str, name: str, embedding: np.ndarray) -> TargetFace:
    """Helper to construct mock TargetFace instances."""
    ref_img = np.full((128, 128, 3), 160, dtype=np.uint8)
    meta = TargetMetadata(person_id=target_id, display_name=name, category="test")
    return TargetFace(
        target_id=target_id,
        display_name=name,
        category="test",
        reference_image_path="",
        reference_image=ref_img,
        embedding=embedding,
        metadata=meta,
    )


# =========================================================================
# 1. Identity Morphing & SLERP Mathematics
# =========================================================================

def test_slerp_interpolation_properties():
    """Validates that SLERP preserves unit norm and smoothly traverses embedding hypersphere."""
    # Create two arbitrary 512-D vectors
    rng = np.random.RandomState(42)
    v0 = rng.randn(512).astype(np.float32)
    v0 /= np.linalg.norm(v0)

    v1 = rng.randn(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    # At t=0.0, output should be identical to v0
    res_0 = slerp(v0, v1, 0.0)
    assert np.allclose(res_0, v0, atol=1e-5)

    # At t=1.0, output should be identical to v1
    res_1 = slerp(v0, v1, 1.0)
    assert np.allclose(res_1, v1, atol=1e-5)

    # At midpoint t=0.5, norm must remain exactly 1.0
    res_mid = slerp(v0, v1, 0.5)
    assert np.isclose(np.linalg.norm(res_mid), 1.0, atol=1e-4)

    # Distance to v0 and v1 should be symmetric
    dot_0 = float(np.dot(res_mid, v0))
    dot_1 = float(np.dot(res_mid, v1))
    assert np.isclose(dot_0, dot_1, atol=1e-3)


def test_slerp_parallel_vectors():
    """Verifies that nearly parallel or identical vectors are handled gracefully without division by zero."""
    v = np.ones(512, dtype=np.float32) / np.sqrt(512)
    res = slerp(v, v, 0.5)
    assert np.isclose(np.linalg.norm(res), 1.0, atol=1e-4)
    assert np.allclose(res, v, atol=1e-4)


def test_smoothstep_easing():
    """Verifies Hermite smoothstep boundary values and inflection."""
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == 0.5
    assert smoothstep(0.1) < 0.1  # Ease in (slower start)
    assert smoothstep(0.9) > 0.9  # Ease out (slower end)


def test_identity_morph_dual_target_fusion():
    """Validates real-time blending of two TargetFace identities into a synthetic composite TargetFace."""
    engine = IdentityMorphEngine()
    emb_a = np.zeros(512, dtype=np.float32)
    emb_a[0] = 1.0
    emb_b = np.zeros(512, dtype=np.float32)
    emb_b[1] = 1.0

    target_a = create_mock_target("id_a", "Actor A", emb_a)
    target_b = create_mock_target("id_b", "Actor B", emb_b)

    fused = engine.fuse_targets(target_a, target_b, ratio=0.5)
    assert fused is not None
    assert "Actor A (50%) + Actor B (50%)" in fused.display_name
    assert np.isclose(np.linalg.norm(fused.embedding), 1.0, atol=1e-4)
    assert fused.embedding[0] > 0.5 and fused.embedding[1] > 0.5


def test_identity_morph_transition_state_machine():
    """Tests temporal morph transition from source to destination target."""
    engine = IdentityMorphEngine(default_duration=0.1)  # fast transition for test
    emb_a = np.zeros(512, dtype=np.float32); emb_a[0] = 1.0
    emb_b = np.zeros(512, dtype=np.float32); emb_b[1] = 1.0

    target_a = create_mock_target("id_a", "Actor A", emb_a)
    target_b = create_mock_target("id_b", "Actor B", emb_b)

    engine.start_morph(target_a, target_b, duration=0.1)
    assert engine.is_morphing is True
    assert engine.source_target.target_id == "id_a"
    assert engine.destination_target.target_id == "id_b"

    tel = engine.get_telemetry()
    assert tel["is_morphing"] is True
    assert tel["source_id"] == "id_a"
    assert tel["destination_id"] == "id_b"

    # Advance step
    step_target = engine.resolve_effective_target(target_a)
    assert step_target is not None

    # Cancel morph
    engine.cancel_morph()
    assert engine.is_morphing is False
    assert engine.resolve_effective_target(target_a).target_id == "id_a"


# =========================================================================
# 2. Volumetric Relighting & Directional Shadows
# =========================================================================

def test_volumetric_relighting_3d_vector_estimation():
    """Validates 3D key light vector estimation from directional face gradients."""
    engine = VolumetricRelightingEngine()
    
    # Create synthetic face crop illuminated heavily from the left
    face_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    face_crop[:, :64] = 220  # Left side bright
    face_crop[:, 64:] = 50   # Right side dark

    lx, ly, lz = engine.estimate_3d_light_vector(face_crop)
    # Light should indicate strong positive X direction (from left)
    assert lx > 0.3
    # Vector must be unit normalized
    norm = np.sqrt(lx * lx + ly * ly + lz * lz)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_subsurface_scattering_warmth_injection():
    """Validates that dermal SSS warmth boosts red/peach chrominance along shadow terminators."""
    engine = VolumetricRelightingEngine()
    swapped_crop = np.full((128, 128, 3), 140, dtype=np.uint8)
    shading_field = np.zeros((128, 128), dtype=np.float32)
    # Sharp shadow terminator at X=64
    shading_field[:, :64] = 1.0
    shading_field[:, 64:] = 0.2

    warm_crop = engine.inject_subsurface_scattering_warmth(
        swapped_crop, shading_field=shading_field, warmth_strength=0.8
    )

    assert warm_crop.shape == swapped_crop.shape
    # YCrCb inspection: Cr (red chrominance, index 1 in YCrCb) should be higher near X=64
    ycrcb_orig = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2YCrCb)
    ycrcb_warm = cv2.cvtColor(warm_crop, cv2.COLOR_BGR2YCrCb)

    cr_orig_mid = np.mean(ycrcb_orig[:, 60:68, 1])
    cr_warm_mid = np.mean(ycrcb_warm[:, 60:68, 1])
    assert cr_warm_mid > cr_orig_mid


def test_harmonize_volumetric_lighting_end_to_end():
    """Validates end-to-end volumetric lighting transfer and returned telemetry."""
    engine = VolumetricRelightingEngine(default_shadow_strength=0.6, default_warmth_strength=0.5)
    orig_crop = np.full((128, 128, 3), 180, dtype=np.uint8)
    orig_crop[64:, :] = 40  # Shadow on bottom half
    swapped_crop = np.full((128, 128, 3), 150, dtype=np.uint8)

    lit_crop, telemetry = engine.harmonize_volumetric_lighting(
        original_crop=orig_crop,
        swapped_crop=swapped_crop,
    )

    assert lit_crop is not None
    assert lit_crop.shape == (128, 128, 3)
    assert "light_vector" in telemetry
    assert telemetry["sss_applied"] is True


# =========================================================================
# 3. Interactive Split-Screen & Comparison Renderer
# =========================================================================

def test_split_screen_modes():
    """Validates all comparison rendering modes: off, vertical split, side by side, difference."""
    orig = np.full((480, 640, 3), 100, dtype=np.uint8)
    trans = np.full((480, 640, 3), 200, dtype=np.uint8)

    # 1. Mode: off
    res_off = SplitScreenRenderer.render(orig, trans, mode="off")
    assert np.array_equal(res_off, trans)

    # 2. Mode: split_vertical
    res_split = SplitScreenRenderer.render(orig, trans, mode="split_vertical", split_position=0.50)
    assert res_split.shape == (480, 640, 3)
    # Left side (X=100) should be original frame
    assert np.allclose(res_split[200, 100], orig[200, 100], atol=5)
    # Right side (X=500) should be transformed frame
    assert np.allclose(res_split[200, 500], trans[200, 500], atol=5)

    # 3. Mode: side_by_side
    res_sbs = SplitScreenRenderer.render(orig, trans, mode="side_by_side")
    assert res_sbs.shape == (480, 640, 3)

    # 4. Mode: difference heatmap
    res_diff = SplitScreenRenderer.render(orig, trans, mode="difference")
    assert res_diff.shape == (480, 640, 3)
    # Different from both pure original and transformed due to heatmap
    assert not np.array_equal(res_diff, trans)


# =========================================================================
# 4. RealTimePipeline Phase 15 End-to-End Integration
# =========================================================================

def test_pipeline_phase15_integration():
    """Validates RealTimePipeline runs cleanly with all Phase 15 features active."""
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    app_cfg.processing.enable_volumetric_relighting = True
    app_cfg.processing.volumetric_shadow_strength = 0.60
    app_cfg.processing.subsurface_scattering_warmth = 0.50
    app_cfg.processing.enable_identity_morphing = True
    app_cfg.processing.split_screen_mode = "split_vertical"
    app_cfg.processing.split_screen_position = 0.50

    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    # Verify initial config
    assert pipeline.relighting_engine.default_shadow_strength == 0.60
    assert pipeline.relighting_engine.default_warmth_strength == 0.50

    # Configure Phase 15 parameters via helper methods
    pipeline.set_volumetric_relighting_config(shadow_strength=0.75, sss_warmth=0.60)
    assert pipeline.relighting_engine.default_shadow_strength == 0.75
    assert pipeline.relighting_engine.default_warmth_strength == 0.60

    pipeline.set_split_screen_config(mode="side_by_side", position=0.45)
    assert pipeline.app_config.processing.split_screen_mode == "side_by_side"
    assert pipeline.app_config.processing.split_screen_position == 0.45

    pipeline.set_identity_morph_config(fusion_ratio=0.35, duration=0.8)
    assert pipeline.dual_target_fusion_ratio == 0.35
    assert pipeline.morph_engine.default_duration == 0.8

    # Process a frame
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    res = pipeline.process_frame(frame)

    assert res is not None
    assert res.rendered_frame.shape == (480, 640, 3)
    assert "morph" in res.metrics_summary
    assert "relighting" in res.metrics_summary
    assert "split_screen" in res.metrics_summary

    # Synchronous frame processing
    res_sync = pipeline.process_frame_sync(frame)
    assert res_sync is not None
    assert "morph" in res_sync.metrics_summary
    assert "relighting" in res_sync.metrics_summary
    assert "split_screen" in res_sync.metrics_summary

    pipeline.stop()


# =========================================================================
# 5. Web Studio REST APIs for Phase 15
# =========================================================================

def test_web_studio_phase15_endpoints():
    """Validates FastAPI Phase 15 endpoints via Starlette TestClient."""
    from fastapi.testclient import TestClient
    from src.web.app import create_app

    app = create_app()
    client = TestClient(app)

    # 1. Pipeline Config GET includes Phase 15
    cfg_resp = client.get("/api/pipeline/config")
    assert cfg_resp.status_code == 200
    cfg_data = cfg_resp.json()
    assert "enable_volumetric_relighting" in cfg_data
    assert "volumetric_shadow_strength" in cfg_data
    assert "subsurface_scattering_warmth" in cfg_data
    assert "enable_identity_morphing" in cfg_data
    assert "split_screen_mode" in cfg_data

    # 2. Update Relighting Settings
    relight_resp = client.post("/api/settings/relighting?shadow_strength=0.7&sss_warmth=0.6")
    assert relight_resp.status_code == 200
    assert relight_resp.json()["success"] is True

    # 3. Update Split Screen Settings
    split_resp = client.post("/api/settings/split_screen?mode=split_vertical&position=0.65")
    assert split_resp.status_code == 200
    assert split_resp.json()["split_screen"]["mode"] == "split_vertical"
    assert split_resp.json()["split_screen"]["position"] == 0.65

    # 4. Telemetry Phase 15 endpoint
    tel_resp = client.get("/api/telemetry/phase15")
    assert tel_resp.status_code == 200
    tel_data = tel_resp.json()
    assert "morph" in tel_data
    assert "relighting" in tel_data
    assert "split_screen" in tel_data
