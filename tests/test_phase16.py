"""
Unit and Integration Tests for Phase 16:
Cinematic Depth-of-Field Portrait Bokeh, Dynamic Micro-Expression Wrinkle Transfer,
and Directional Motion Blur Harmonization.
"""

import math
import numpy as np
import cv2
import pytest
from starlette.testclient import TestClient

from src.processing.cinematic_depth import CinematicDepthEngine
from src.processing.micro_expression import MicroExpressionEngine
from src.processing.motion_blur import MotionBlurHarmonizer
from src.pipeline.realtime_pipeline import RealTimePipeline
from src.core.config_loader import AppConfig, ModelsConfig, TargetsConfig, load_all_configs
from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.web.app import create_app


# =========================================================================
# 1. Cinematic Monocular Depth Estimation & Portrait Bokeh Tests
# =========================================================================

def test_depth_map_estimation_from_landmarks():
    """Validates depth map dimensions, float32 range [0.0, 1.0], and foreground/background gradient."""
    engine = CinematicDepthEngine(default_focus_depth=0.35)
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)

    # 5-point mock landmarks in center of frame
    landmarks = np.array([
        [140.0, 110.0],  # left eye
        [180.0, 110.0],  # right eye
        [160.0, 125.0],  # nose
        [145.0, 145.0],  # left mouth
        [175.0, 145.0],  # right mouth
    ], dtype=np.float32)

    depth_map = engine.estimate_depth_map(frame, landmarks=landmarks)
    assert depth_map.shape == (240, 320)
    assert depth_map.dtype == np.float32
    assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)

    # Subject center (nose tip) should be significantly closer (lower depth) than the far corners
    center_depth = float(depth_map[125, 160])
    corner_depth = float(depth_map[10, 10])
    assert center_depth < corner_depth
    assert center_depth < 0.45
    assert corner_depth > 0.70


def test_depth_map_fallback_without_landmarks():
    """Verifies graceful fallback to center-weighted radial depth when landmarks are missing."""
    engine = CinematicDepthEngine()
    frame = np.full((128, 128, 3), 100, dtype=np.uint8)

    depth_map = engine.estimate_depth_map(frame, landmarks=None, face_bbox=None)
    assert depth_map.shape == (128, 128)
    assert float(depth_map[64, 64]) < float(depth_map[0, 0])


def test_cinematic_bokeh_blur_application():
    """Validates circle-of-confusion bokeh blur on background with subject protection."""
    engine = CinematicDepthEngine(default_focus_depth=0.30, default_bokeh_radius=15.0)

    # Synthesize checkerboard background to have high-frequency edges
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[::10, :] = 255
    frame[:, ::10] = 255

    # Depth map: center is in focus (Z=0.30), edges are far (Z=0.90)
    y, x = np.indices((200, 200), dtype=np.float32)
    dist = np.sqrt((x - 100)**2 + (y - 100)**2) / 100.0
    depth_map = np.clip(0.30 + dist * 0.60, 0.0, 1.0).astype(np.float32)

    face_mask = np.zeros((200, 200), dtype=np.float32)
    cv2.circle(face_mask, (100, 100), 30, 1.0, -1)

    bokeh_frame = engine.apply_cinematic_bokeh(
        frame=frame,
        depth_map=depth_map,
        focus_depth=0.30,
        bokeh_radius=15.0,
        face_mask=face_mask,
    )

    assert bokeh_frame.shape == frame.shape
    assert bokeh_frame.dtype == np.uint8

    # The protected face center should remain identical or very close to original
    center_orig = frame[95:105, 95:105]
    center_bokeh = bokeh_frame[95:105, 95:105]
    assert np.allclose(center_orig, center_bokeh, atol=2)

    # Background corners should be blurred (gradient variance reduced)
    bg_orig_var = float(np.var(frame[:30, :30]))
    bg_bokeh_var = float(np.var(bokeh_frame[:30, :30]))
    assert bg_bokeh_var < bg_orig_var


def test_cinematic_bokeh_zero_radius_passthrough():
    """Verifies that bokeh_radius=0 is a zero-overhead passthrough."""
    engine = CinematicDepthEngine()
    frame = np.full((50, 50, 3), 150, dtype=np.uint8)
    out = engine.apply_cinematic_bokeh(frame, bokeh_radius=0.0)
    assert np.array_equal(out, frame)


def test_depth_visualizer_rendering():
    """Verifies false-color depth visualization overlay generation."""
    engine = CinematicDepthEngine()
    depth_map = np.linspace(0.0, 1.0, 10000, dtype=np.float32).reshape((100, 100))
    vis = engine.render_depth_visualizer(depth_map, show_hud=True)
    assert vis.shape == (100, 100, 3)
    assert vis.dtype == np.uint8


def test_depth_telemetry():
    """Verifies depth engine telemetry dictionary structure."""
    engine = CinematicDepthEngine(default_focus_depth=0.40, default_bokeh_radius=14.0)
    tel = engine.get_telemetry()
    assert "focus_depth" in tel and tel["focus_depth"] == 0.40
    assert "bokeh_radius" in tel and tel["bokeh_radius"] == 14.0
    assert "mean_depth" in tel


# =========================================================================
# 2. Dynamic Micro-Expression Wrinkle & Skin Strain Field Tests
# =========================================================================

def test_expression_strain_neutral_vs_smile():
    """Validates that a wide-mouth smile landmark set triggers elevated smile strain."""
    engine = MicroExpressionEngine(temporal_smoothing=0.0)

    # Neutral 5-point layout
    neutral_lms = np.array([
        [40.0, 50.0],   # left eye
        [80.0, 50.0],   # right eye (inter-eye = 40)
        [60.0, 70.0],   # nose
        [45.0, 95.0],   # left mouth
        [75.0, 95.0],   # right mouth (mouth width = 30; ratio = 30/40 = 0.75)
    ], dtype=np.float32)

    # Broad smile: mouth widens significantly to 48px
    smile_lms = np.array([
        [40.0, 50.0],
        [80.0, 50.0],
        [60.0, 70.0],
        [36.0, 90.0],   # corners pulled outward and upward
        [84.0, 90.0],   # mouth width = 48; ratio = 48/40 = 1.20
    ], dtype=np.float32)

    neutral_strain = engine.calculate_expression_strain(neutral_lms, track_id=1)
    smile_strain = engine.calculate_expression_strain(smile_lms, track_id=2)

    assert neutral_strain["smile_strain"] < 0.20
    assert smile_strain["smile_strain"] > 0.50
    assert smile_strain["composite_strain"] > neutral_strain["composite_strain"]


def test_extract_high_frequency_wrinkles():
    """Validates that Difference-of-Gaussians extracts high-frequency cutaneous detail."""
    engine = MicroExpressionEngine()
    # Crop with synthetic skin wrinkle texture (sinusoidal ridge)
    crop = np.full((128, 128, 3), 140, dtype=np.uint8)
    crop[60:65, 30:100] = 90  # Dark crease line

    detail = engine.extract_high_frequency_wrinkles(crop)
    assert detail.shape == (128, 128)
    assert detail.dtype == np.float32
    assert -1.0 <= np.min(detail) <= 0.0
    assert 0.0 <= np.max(detail) <= 1.0


def test_build_anatomical_zone_masks():
    """Validates creation of smooth spatial masks for nasolabial, forehead, and crow's feet."""
    engine = MicroExpressionEngine()
    masks = engine.build_anatomical_zone_masks(crop_shape=(128, 128))

    assert "nasolabial" in masks
    assert "forehead" in masks
    assert "crow_feet" in masks

    for name, m in masks.items():
        assert m.shape == (128, 128)
        assert np.all(m >= 0.0) and np.all(m <= 1.0)
        assert np.max(m) > 0.8  # Mask activates cleanly


def test_transfer_micro_expressions_active_strain():
    """Tests wrinkle synthesis and transfer onto swapped face under active strain."""
    engine = MicroExpressionEngine(default_strength=0.75, default_sensitivity=0.85, temporal_smoothing=0.0)

    orig_crop = np.full((128, 128, 3), 150, dtype=np.uint8)
    orig_crop[60:64, 40:88] = 70  # Smile crease

    swapped_crop = np.full((128, 128, 3), 150, dtype=np.uint8)

    # Broad smile landmarks
    smile_lms = np.array([
        [38.0, 50.0],
        [78.0, 50.0],
        [58.0, 70.0],
        [30.0, 90.0],
        [86.0, 90.0],
    ], dtype=np.float32)

    enhanced, metrics = engine.transfer_micro_expressions(
        original_crop=orig_crop,
        swapped_crop=swapped_crop,
        landmarks=smile_lms,
        strength=0.75,
        sensitivity=0.85,
        track_id=1,
    )

    assert enhanced.shape == swapped_crop.shape
    assert enhanced.dtype == np.uint8
    assert metrics["active_wrinkles"] is True
    assert metrics["composite_strain"] > 0.30

    # The swapped crop should now contain transferred wrinkle variations
    diff = np.abs(enhanced.astype(np.float32) - swapped_crop.astype(np.float32))
    assert np.max(diff) > 2.0


def test_transfer_micro_expressions_zero_strength():
    """Verifies that strength=0 leaves swapped crop intact."""
    engine = MicroExpressionEngine()
    orig = np.full((64, 64, 3), 100, dtype=np.uint8)
    swapped = np.full((64, 64, 3), 150, dtype=np.uint8)

    enhanced, _ = engine.transfer_micro_expressions(orig, swapped, strength=0.0)
    assert np.array_equal(enhanced, swapped)


# =========================================================================
# 3. Directional Motion Blur Harmonization Tests
# =========================================================================

def test_motion_vector_tracking():
    """Validates velocity magnitude and trajectory angle calculation across consecutive positions."""
    harmonizer = MotionBlurHarmonizer(velocity_inertia=0.0)

    # Frame 1: centroid at (100, 100)
    lms_t0 = np.array([[100.0, 100.0]] * 5, dtype=np.float32)
    m0 = harmonizer.track_motion(landmarks=lms_t0, track_id=1)
    assert m0["velocity"] == 0.0

    # Frame 2: centroid shifts horizontally to (120, 100) -> vx = 20, vy = 0
    lms_t1 = np.array([[120.0, 100.0]] * 5, dtype=np.float32)
    m1 = harmonizer.track_motion(landmarks=lms_t1, track_id=1)
    assert np.isclose(m1["velocity"], 20.0, atol=1e-3)
    assert np.isclose(m1["vx"], 20.0, atol=1e-3)
    assert np.isclose(m1["vy"], 0.0, atol=1e-3)
    assert np.isclose(m1["angle_deg"], 0.0, atol=1.0)


def test_directional_blur_kernel_creation():
    """Validates that directional blur kernel sums to 1.0 and has correct dimensions."""
    harmonizer = MotionBlurHarmonizer()
    kernel = harmonizer.create_directional_kernel(length=7.0, angle_rad=0.0, max_size=15)

    assert kernel.dtype == np.float32
    assert kernel.shape[0] % 2 == 1  # Odd dimension
    assert np.isclose(float(np.sum(kernel)), 1.0, atol=1e-4)

    # At angle 0 (horizontal), central row should have higher weights than top/bottom
    center = kernel.shape[0] // 2
    assert np.sum(kernel[center, :]) > np.sum(kernel[0, :])


def test_motion_blur_harmonization_below_threshold():
    """Verifies that slow movement below velocity threshold is untouched."""
    harmonizer = MotionBlurHarmonizer(default_threshold=5.0, velocity_inertia=0.0)
    crop = np.full((64, 64, 3), 120, dtype=np.uint8)

    # Move only 1.5 pixels (below 5.0 threshold)
    lms0 = np.array([[50.0, 50.0]] * 5, dtype=np.float32)
    harmonizer.track_motion(lms0, track_id=1)
    lms1 = np.array([[51.5, 50.0]] * 5, dtype=np.float32)

    out, metrics = harmonizer.harmonize_motion_blur(
        crop,
        track_id=1,
        landmarks=lms1,
        threshold=5.0,
    )
    assert metrics["blur_applied"] is False
    assert np.array_equal(out, crop)


def test_motion_blur_harmonization_above_threshold():
    """Verifies that fast movement above velocity threshold applies directional blur."""
    harmonizer = MotionBlurHarmonizer(default_threshold=2.0, velocity_inertia=0.0)
    # High-contrast sharp crop
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, 50:] = 255

    # Move rapidly by 15 pixels
    lms0 = np.array([[50.0, 50.0]] * 5, dtype=np.float32)
    harmonizer.track_motion(lms0, track_id=1)
    lms1 = np.array([[65.0, 50.0]] * 5, dtype=np.float32)

    out, metrics = harmonizer.harmonize_motion_blur(
        crop,
        track_id=1,
        landmarks=lms1,
        threshold=2.0,
        sensitivity=0.60,
    )
    assert metrics["blur_applied"] is True
    assert metrics["kernel_length"] > 1.0
    assert out.shape == crop.shape

    # Vertical edge at col 50 should be blurred into neighboring columns
    assert out[50, 48, 0] > 0
    assert out[50, 52, 0] < 255


# =========================================================================
# 4. Pipeline & Web Studio Integration Tests
# =========================================================================

def test_pipeline_phase16_configuration_setters():
    """Validates Phase 16 configuration methods on RealTimePipeline."""
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    # Bokeh setters
    pipeline.set_cinematic_bokeh_config(
        enabled=True,
        bokeh_radius=18.0,
        focus_depth=0.45,
        specular_boost=0.35,
        visualize_depth_hud=True,
    )
    assert pipeline.app_config.processing.enable_cinematic_bokeh is True
    assert pipeline.depth_engine.bokeh_radius == 18.0
    assert pipeline.depth_engine.focus_depth == 0.45
    assert pipeline.depth_engine.specular_boost == 0.35
    assert pipeline.app_config.processing.visualize_depth_hud is True

    # Micro-expression setters
    pipeline.set_micro_expression_config(
        enabled=True,
        strength=0.80,
        sensitivity=0.90,
    )
    assert pipeline.app_config.processing.enable_micro_expressions is True
    assert pipeline.micro_expression_engine.default_strength == 0.80
    assert pipeline.micro_expression_engine.default_sensitivity == 0.90

    # Motion blur setters
    pipeline.set_motion_blur_config(
        enabled=True,
        sensitivity=0.65,
    )
    assert pipeline.app_config.processing.enable_motion_blur_matching is True
    assert pipeline.motion_blur_harmonizer.default_sensitivity == 0.65


def test_pipeline_phase16_execution_and_telemetry():
    """Runs a synthetic frame through RealTimePipeline and inspects Phase 16 telemetry."""
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    res = pipeline.process_frame(frame)

    assert res is not None
    assert "phase16" in res.metrics_summary
    p16 = res.metrics_summary["phase16"]

    assert "depth" in p16
    assert "micro_expression" in p16
    assert "motion_blur" in p16
    assert "bokeh_radius" in p16["depth"]
    assert "smile_strain" in p16["micro_expression"]
    assert "velocity" in p16["motion_blur"]


def test_phase16_fastapi_endpoints():
    """Validates FastAPI REST endpoints for Phase 16."""
    app = create_app()
    client = TestClient(app)

    # 1. POST /api/settings/depth_bokeh
    res_bokeh = client.post(
        "/api/settings/depth_bokeh",
        params={
            "enabled": True,
            "bokeh_radius": 15.0,
            "focus_depth": 0.40,
            "specular_boost": 0.30,
        },
    )
    assert res_bokeh.status_code == 200
    data_bokeh = res_bokeh.json()
    assert data_bokeh["success"] is True
    assert data_bokeh["depth_bokeh"]["enabled"] is True
    assert data_bokeh["depth_bokeh"]["bokeh_radius"] == 15.0

    # 2. POST /api/settings/micro_expression
    res_expr = client.post(
        "/api/settings/micro_expression",
        params={
            "enabled": True,
            "strength": 0.70,
            "sensitivity": 0.80,
        },
    )
    assert res_expr.status_code == 200
    data_expr = res_expr.json()
    assert data_expr["success"] is True
    assert "micro_expression" in data_expr

    # 3. POST /api/settings/motion_blur
    res_mb = client.post(
        "/api/settings/motion_blur",
        params={
            "enabled": True,
            "sensitivity": 0.55,
        },
    )
    assert res_mb.status_code == 200
    data_mb = res_mb.json()
    assert data_mb["success"] is True
    assert "motion_blur" in data_mb

    # 4. GET /api/telemetry/phase16
    res_tel = client.get("/api/telemetry/phase16")
    assert res_tel.status_code == 200
    data_tel = res_tel.json()
    assert "depth_bokeh" in data_tel
    assert "micro_expression" in data_tel
    assert "motion_blur" in data_tel
