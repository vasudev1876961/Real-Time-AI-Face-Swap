"""
Automated Test Suite for Phase 9:
Ocular Gaze Realism, Blink Synchronization, 3D Pose Adaptation,
and Specular-Ambient Lighting Harmonization.
"""

import numpy as np
import pytest
import cv2

from src.processing.eye_gaze import EyeGazePreserver
from src.processing.pose_adaptation import PoseAdaptationEngine
from src.processing.lighting import SpecularAmbientLightingHarmonizer
from src.detection.face_landmarks import INSWAPPER_STANDARD_128, ARCFACE_STANDARD_512
from src.core.config_loader import AppConfig, ProcessingConfig, load_all_configs
from src.pipeline.realtime_pipeline import RealTimePipeline


# =========================================================================
# 1. Eye Gaze, Blink Detection & Catchlight Tests
# =========================================================================

def test_ocular_mask_generation():
    preserver = EyeGazePreserver()
    left_m, right_m, comb_m = preserver.compute_ocular_masks((128, 128), INSWAPPER_STANDARD_128)

    assert left_m.shape == (128, 128)
    assert right_m.shape == (128, 128)
    assert comb_m.shape == (128, 128)
    assert left_m.dtype == np.float32
    assert 0.0 <= left_m.max() <= 1.0
    assert 0.0 <= right_m.max() <= 1.0

    # Left eye center should be in left half of image (x < 64)
    y_l, x_l = np.unravel_index(np.argmax(left_m), left_m.shape)
    assert x_l < 64

    # Right eye center should be in right half of image (x > 64)
    y_r, x_r = np.unravel_index(np.argmax(right_m), right_m.shape)
    assert x_r > 64


def test_ocular_mask_512_resolution():
    preserver = EyeGazePreserver()
    left_m, right_m, comb_m = preserver.compute_ocular_masks((512, 512), ARCFACE_STANDARD_512)

    assert left_m.shape == (512, 512)
    assert right_m.shape == (512, 512)
    assert comb_m.shape == (512, 512)
    assert np.max(comb_m) > 0.8


def test_blink_state_open_vs_closed():
    preserver = EyeGazePreserver(blink_threshold=0.20)
    left_m, right_m, _ = preserver.compute_ocular_masks((128, 128), INSWAPPER_STANDARD_128)

    # 1. Open Eye Crop: high contrast dark pupil and white sclera (high vertical gradient)
    open_crop = np.full((128, 128, 3), 160, dtype=np.uint8)
    # Draw dark pupils
    cv2.circle(open_crop, (46, 51), 7, (20, 20, 20), -1)
    cv2.circle(open_crop, (81, 51), 7, (20, 20, 20), -1)
    # Draw white sclera border
    cv2.ellipse(open_crop, (46, 51), (12, 7), 0, 0, 360, (240, 240, 240), 2)
    cv2.ellipse(open_crop, (81, 51), (12, 7), 0, 0, 360, (240, 240, 240), 2)

    is_l_blink, is_r_blink, open_l, open_r = preserver.detect_blink_state(open_crop, left_m, right_m)
    assert not is_l_blink
    assert not is_r_blink
    assert open_l > 0.20
    assert open_r > 0.20

    # 2. Closed Eye Crop: smooth uniform eyelid skin tone (low vertical gradient)
    closed_crop = np.full((128, 128, 3), 140, dtype=np.uint8)
    cv2.line(closed_crop, (36, 51), (56, 51), (130, 130, 130), 1)
    cv2.line(closed_crop, (71, 51), (91, 51), (130, 130, 130), 1)

    is_l_blink_c, is_r_blink_c, open_l_c, open_r_c = preserver.detect_blink_state(closed_crop, left_m, right_m)
    assert is_l_blink_c
    assert is_r_blink_c
    assert open_l_c < open_l


def test_corneal_catchlight_extraction():
    preserver = EyeGazePreserver()
    _, _, comb_m = preserver.compute_ocular_masks((128, 128), INSWAPPER_STANDARD_128)

    crop = np.full((128, 128, 3), 100, dtype=np.uint8)
    # Add bright specular catchlight
    cv2.circle(crop, (46, 50), 2, (250, 250, 250), -1)

    catchlights = preserver.extract_corneal_catchlights(crop, comb_m, luminance_threshold=210)
    assert catchlights.shape == (128, 128, 3)
    assert np.max(catchlights) > 0.0


def test_eye_realism_preservation_pipeline():
    preserver = EyeGazePreserver(default_strength=0.80)
    orig = np.full((128, 128, 3), 120, dtype=np.uint8)
    # Closed eyes in original (blink)
    cv2.line(orig, (38, 51), (54, 51), (80, 80, 80), 2)
    cv2.line(orig, (73, 51), (89, 51), (80, 80, 80), 2)

    swap = np.full((128, 128, 3), 160, dtype=np.uint8)
    # Open synthetic eyes in swap
    cv2.circle(swap, (46, 51), 8, (10, 10, 10), -1)
    cv2.circle(swap, (81, 51), 8, (10, 10, 10), -1)

    refined = preserver.preserve_eye_realism(
        original_crop=orig,
        swapped_crop=swap,
        landmarks=INSWAPPER_STANDARD_128,
        strength=0.85,
    )

    assert refined.shape == swap.shape
    assert refined.dtype == np.uint8
    # Should not be identical to swap because closed eyelid was composited
    assert not np.array_equal(refined, swap)


def test_eye_realism_zero_strength():
    preserver = EyeGazePreserver(default_strength=0.0)
    orig = np.full((128, 128, 3), 100, dtype=np.uint8)
    swap = np.full((128, 128, 3), 180, dtype=np.uint8)

    result = preserver.preserve_eye_realism(orig, swap, strength=0.0)
    np.testing.assert_array_equal(result, swap)


# =========================================================================
# 2. 3D Pose Estimation & Adaptation Tests
# =========================================================================

def test_head_pose_estimation_frontal():
    engine = PoseAdaptationEngine()
    yaw, pitch, roll = engine.estimate_head_pose_3d(INSWAPPER_STANDARD_128)

    # Standard aligned template is roughly centered and level
    assert abs(yaw) < 10.0
    assert abs(pitch) < 15.0
    assert abs(roll) < 5.0


def test_head_pose_estimation_turned():
    engine = PoseAdaptationEngine()
    turned_right = INSWAPPER_STANDARD_128.copy()
    # Shift nose tip strongly right
    turned_right[2, 0] += 16.0

    yaw_r, _, _ = engine.estimate_head_pose_3d(turned_right)
    assert yaw_r > 20.0

    turned_left = INSWAPPER_STANDARD_128.copy()
    turned_left[2, 0] -= 16.0
    yaw_l, _, _ = engine.estimate_head_pose_3d(turned_left)
    assert yaw_l < -20.0


def test_profile_blend_falloff_curve():
    engine = PoseAdaptationEngine(profile_threshold_deg=45.0, max_turn_deg=60.0)

    # Normal angles: full swap
    assert engine.compute_profile_blend_falloff(yaw=0.0) == 1.0
    assert engine.compute_profile_blend_falloff(yaw=30.0) == 1.0
    assert engine.compute_profile_blend_falloff(yaw=-40.0) == 1.0

    # Intermediate angle: graceful decay
    mid_falloff = engine.compute_profile_blend_falloff(yaw=52.5)
    assert 0.0 < mid_falloff < 1.0

    # Extreme angle: complete graceful fallback (no distorted polygon)
    assert engine.compute_profile_blend_falloff(yaw=60.0) == 0.0
    assert engine.compute_profile_blend_falloff(yaw=75.0) == 0.0
    assert engine.compute_profile_blend_falloff(yaw=-65.0) == 0.0


def test_clamp_profile_contour():
    engine = PoseAdaptationEngine()
    contour = np.array([
        [20, 20], [108, 20],
        [115, 64], [100, 110],
        [64, 120],
        [28, 110], [15, 64],
    ], dtype=np.int32)

    # Clamping turned right pulls left edge inwards
    clamped = engine.clamp_profile_contour(contour, yaw=35.0, crop_shape=(128, 128))
    assert clamped.shape == contour.shape
    # Left edge point (x=15) should have shifted right towards center
    orig_left_x = contour[6, 0]
    clamped_left_x = clamped[6, 0]
    assert clamped_left_x > orig_left_x


def test_adapt_mask_pose():
    engine = PoseAdaptationEngine()
    mask = np.ones((128, 128), dtype=np.float32)

    # Frontal should not alter mask
    adapted_front = engine.adapt_mask(mask, yaw=0.0, pitch=0.0)
    np.testing.assert_array_equal(adapted_front, mask)

    # High yaw should apply boundary adaptation
    adapted_turn = engine.adapt_mask(mask, yaw=40.0, pitch=0.0)
    assert adapted_turn.shape == mask.shape


# =========================================================================
# 3. Specular-Ambient Lighting Harmonization Tests
# =========================================================================

def test_specular_highlight_extraction():
    harmonizer = SpecularAmbientLightingHarmonizer(specular_threshold=200)
    img = np.full((128, 128, 3), 150, dtype=np.uint8)
    # Bright glint spot on forehead
    cv2.circle(img, (64, 30), 5, (245, 245, 245), -1)

    spec = harmonizer.extract_specular_highlights(img)
    assert spec.shape == (128, 128, 3)
    assert np.max(spec) > 0.0


def test_directional_shading_field():
    harmonizer = SpecularAmbientLightingHarmonizer()
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Left-to-right gradient (strong window light on left)
    for x in range(128):
        img[:, x] = int(255 * (1.0 - x / 128.0))

    shading = harmonizer.compute_directional_shading_field(img)
    assert shading.shape == (128, 128)
    assert shading[:, 10].mean() > shading[:, 118].mean()


def test_lighting_harmonization_transfer():
    harmonizer = SpecularAmbientLightingHarmonizer(default_strength=0.70)
    orig = np.full((128, 128, 3), 200, dtype=np.uint8)  # Brightly lit
    swap = np.full((128, 128, 3), 100, dtype=np.uint8)  # Dimly lit

    harmonized = harmonizer.harmonize_lighting(orig, swap, strength=0.70)
    assert harmonized.shape == swap.shape
    assert harmonized.dtype == np.uint8
    # Average luminance of harmonized should increase towards orig
    assert harmonized.mean() > swap.mean()


# =========================================================================
# 4. Pipeline Integration Tests
# =========================================================================

def test_realtime_pipeline_phase9_initialization():
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    app_cfg.processing.enable_eye_realism = True
    app_cfg.processing.enable_pose_adaptation = True
    app_cfg.processing.enable_specular_lighting = True

    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    assert hasattr(pipeline, "eye_preserver")
    assert hasattr(pipeline, "pose_adapter")
    assert hasattr(pipeline, "lighting_harmonizer")
    assert isinstance(pipeline.eye_preserver, EyeGazePreserver)
    assert isinstance(pipeline.pose_adapter, PoseAdaptationEngine)
    assert isinstance(pipeline.lighting_harmonizer, SpecularAmbientLightingHarmonizer)

    # Process a frame
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    res = pipeline.process_frame(dummy_frame)
    assert res is not None
    assert res.rendered_frame.shape == (480, 640, 3)

    pipeline.stop()
