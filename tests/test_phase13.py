"""
Unit and Integration Tests for Phase 13:
Anatomical Dense-Mesh Face Masking, Active Edge Snapping, Hairline Carving,
Curvature-Adaptive Feathering, and Real-Time FPS HUD Telemetry.
"""

import numpy as np
import cv2
import pytest

from src.processing.mask_precision import (
    DenseMeshContourProjector,
    AnatomicalContourBuilder,
    ActiveEdgeBoundarySnapper,
    ForeheadHairlineCarver,
    CurvatureAdaptiveFeatherer,
    TemporalMaskStabilizer,
    draw_mask_contour_hud,
)
from src.processing.mask import create_face_mask, FaceMaskGenerator
from src.processing.postprocess import draw_fps_telemetry_hud, postprocess_frame
from src.pipeline.realtime_pipeline import RealTimePipeline, PipelineTimings
from src.core.config_loader import load_all_configs
from src.detection.face_detector import FaceData, MEDIAPIPE_FACE_OVAL_INDICES
from src.detection.face_landmarks import INSWAPPER_STANDARD_128


def test_dense_mesh_contour_projector():
    """Validates 468p MediaPipe face oval projection from frame space to crop space."""
    # Create synthetic full-frame mesh with 468 points
    frame_h, frame_w = 480, 640
    crop_h, crop_w = 128, 128

    # Generate 478 mock landmarks centered around (320, 240)
    mock_mesh = np.zeros((478, 2), dtype=np.float32)
    center = np.array([320.0, 240.0], dtype=np.float32)
    for i in range(478):
        angle = (i / 478.0) * 2.0 * np.pi
        radius = 80.0 + 20.0 * np.sin(angle * 3)
        mock_mesh[i] = center + np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32)

    # Affine matrix mapping (320, 240) -> (64, 64) with scale 0.5
    affine_mat = np.array([
        [0.5, 0.0, 64.0 - 320.0 * 0.5],
        [0.0, 0.5, 64.0 - 240.0 * 0.5],
    ], dtype=np.float32)

    contour = DenseMeshContourProjector.project_mesh_contour(
        mesh_landmarks=mock_mesh,
        affine_mat=affine_mat,
        crop_shape=(crop_h, crop_w),
        subdivisions=2,
    )

    assert contour is not None
    assert len(contour) > 20
    assert np.all(contour[:, 0] >= 0) and np.all(contour[:, 0] < crop_w)
    assert np.all(contour[:, 1] >= 0) and np.all(contour[:, 1] < crop_h)


def test_anatomical_contour_builder_24p():
    """Validates 24-point anatomical contour builder with pose adaptation."""
    crop_shape = (128, 128)
    lms = INSWAPPER_STANDARD_128.copy()

    # Frontal contour
    poly_frontal = AnatomicalContourBuilder.build_contour_24p(
        crop_shape=crop_shape,
        landmarks=lms,
        yaw=0.0,
        pitch=0.0,
    )
    assert poly_frontal is not None
    assert len(poly_frontal) >= 18
    assert np.all(poly_frontal[:, 0] >= 0) and np.all(poly_frontal[:, 0] < 128)
    assert np.all(poly_frontal[:, 1] >= 0) and np.all(poly_frontal[:, 1] < 128)

    # Turned head contour (yaw = 30)
    poly_turned = AnatomicalContourBuilder.build_contour_24p(
        crop_shape=crop_shape,
        landmarks=lms,
        yaw=30.0,
        pitch=-10.0,
    )
    assert poly_turned is not None
    # Turned cheek should differ in coordinates from frontal
    assert not np.array_equal(poly_frontal, poly_turned)


def test_active_edge_boundary_snapper():
    """Validates edge snapping to skin/background gradient boundaries."""
    crop_shape = (128, 128)
    crop = np.full((128, 128, 3), 40, dtype=np.uint8)
    # Draw oval skin tone in center
    cv2.ellipse(crop, (64, 64), (40, 50), 0, 0, 360, (140, 160, 210), -1)

    # Base contour slightly offset from oval
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    pts = np.stack([64 + 43 * np.cos(angles), 64 + 53 * np.sin(angles)], axis=1).astype(np.int32)

    snapper = ActiveEdgeBoundarySnapper(search_radius=5, strength=0.8)
    snapped = snapper.snap_contour_to_edges(pts, crop)

    assert snapped is not None
    assert snapped.shape == pts.shape
    assert np.all(snapped[:, 0] >= 0) and np.all(snapped[:, 0] < 128)


def test_forehead_hairline_carver():
    """Validates bangs and hair silhouette carving in the upper forehead region."""
    crop_shape = (128, 128)
    crop = np.full((128, 128, 3), (140, 160, 210), dtype=np.uint8)  # Skin tone

    # Add dark hair bangs across upper forehead (y: 10..35, x: 40..88)
    crop[10:35, 40:88] = [20, 20, 20]
    # Add texture noise to hair
    noise = np.random.randint(-15, 15, (25, 48, 3), dtype=np.int16)
    crop[10:35, 40:88] = np.clip(crop[10:35, 40:88].astype(np.int16) + noise, 0, 255).astype(np.uint8)

    base_mask = np.full((128, 128), 255, dtype=np.uint8)
    carver = ForeheadHairlineCarver(strength=0.7)
    carved = carver.carve_hairline(base_mask, crop, landmarks=INSWAPPER_STANDARD_128)

    assert carved is not None
    assert carved.shape == (128, 128)
    # The hair region should have reduced mask intensity
    assert np.mean(carved[15:30, 45:80]) < np.mean(base_mask[15:30, 45:80])


def test_curvature_adaptive_featherer():
    """Validates spatially adaptive feathering (tight jawline, soft forehead)."""
    binary_mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(binary_mask, (64, 64), 45, 255, -1)

    feathered = CurvatureAdaptiveFeatherer.create_curvature_feathered_mask(
        binary_mask, base_radius=8.0, eye_y_ratio=0.42, mouth_y_ratio=0.70
    )

    assert feathered is not None
    assert feathered.shape == (128, 128)
    assert feathered.dtype == np.float32
    assert np.min(feathered) >= 0.0 and np.max(feathered) <= 1.0

    # Upper transition (forehead) should have wider transition band than lower (jaw)
    # Gradient magnitude in lower face (jaw) should be steeper (higher max gradient)
    grad_y = np.abs(cv2.Sobel(feathered, cv2.CV_32F, 0, 1, ksize=3))
    forehead_grad = np.max(grad_y[20:50, :])
    jaw_grad = np.max(grad_y[80:115, :])
    assert jaw_grad >= forehead_grad * 0.9  # Jawline is sharp and tight


def test_temporal_mask_stabilizer():
    """Validates multi-frame EMA stabilization of mask mattes."""
    stabilizer = TemporalMaskStabilizer(alpha=0.6)
    mask1 = np.full((64, 64), 0.8, dtype=np.float32)
    mask2 = np.full((64, 64), 0.4, dtype=np.float32)

    res1 = stabilizer.stabilize(mask1, track_id=1)
    assert np.allclose(res1, 0.8)

    res2 = stabilizer.stabilize(mask2, track_id=1)
    expected = 0.6 * 0.4 + 0.4 * 0.8  # 0.56
    assert np.allclose(res2, expected, atol=1e-3)

    stabilizer.reset(track_id=1)
    res3 = stabilizer.stabilize(mask2, track_id=1)
    assert np.allclose(res3, 0.4)


def test_create_face_mask_phase13_full_integration():
    """Validates create_face_mask with dense mesh, edge snapping, and curvature feathering."""
    crop = np.full((128, 128, 3), 150, dtype=np.uint8)

    # 1. Standard anatomical with curvature feathering
    mask = create_face_mask(
        crop_shape=(128, 128),
        mask_type="smooth_hull",
        enable_curvature_feathering=True,
        enable_edge_snapping=True,
        aligned_crop=crop,
    )
    assert mask.shape == (128, 128)
    assert mask.dtype == np.float32
    assert 0.0 <= np.min(mask) <= np.max(mask) <= 1.0

    # 2. Generator caching and stabilization
    gen = FaceMaskGenerator()
    m1 = gen.generate_mask(crop_shape=(128, 128), aligned_crop=crop, track_id=1)
    m2 = gen.generate_mask(crop_shape=(128, 128), aligned_crop=crop, track_id=1)
    assert m1.shape == (128, 128)
    assert m2.shape == (128, 128)


def test_fps_telemetry_hud_rendering():
    """Validates live FPS HUD overlay drawing on video frame."""
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    timings = PipelineTimings()
    timings.tracking_ms = 4.2
    timings.swap_ms = 14.5
    timings.mask_ms = 0.8
    timings.blend_ms = 3.1
    timings.total_ms = 26.2

    hud_frame = draw_fps_telemetry_hud(
        frame=frame,
        fps=36.4,
        latency_ms=26.2,
        timings=timings,
        governor_badge="optimal",
        provider_name="CPUExecutionProvider",
        target_fps=30.0,
        is_swapped=True,
        active_target_name="Prabhas",
    )

    assert hud_frame is not None
    assert hud_frame.shape == (480, 640, 3)
    # The HUD region in top-left should have modified pixel values
    assert not np.array_equal(hud_frame[20:100, 20:300], frame[20:100, 20:300])


def test_realtime_pipeline_phase13_end_to_end():
    """Validates RealTimePipeline execution with Phase 13 mask precision and FPS telemetry."""
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    app_cfg.processing.enable_dense_mesh_mask = True
    app_cfg.processing.enable_edge_snapping = True
    app_cfg.processing.enable_curvature_feathering = True
    app_cfg.processing.show_fps_hud = True

    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    # Synthetic test frame (480x640)
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.ellipse(frame, (320, 240), (80, 110), 0, 0, 360, (140, 165, 210), -1)

    result = pipeline.process_frame(frame)
    assert result is not None
    assert result.rendered_frame.shape == (480, 640, 3)

    telemetry = pipeline.get_fps_telemetry()
    assert "fps" in telemetry
    assert "stage_averages_ms" in telemetry
    assert "governor_state" in telemetry
    assert "active_provider" in telemetry

    pipeline.stop()
