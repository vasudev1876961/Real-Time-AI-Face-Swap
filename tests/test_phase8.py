"""
Automated Test Suite for Phase 8:
- Studio Color Grading Engine, LUT Caching & Presets
- Multi-Portrait Identity Embedding Fusion & Spherical Interpolation
- Mouth & Oral Cavity Fidelity Preservation
- Resolution-Invariant Color Correction & Async 512px Pipeline Stability
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

from src.processing.color_grading import (
    ColorGradingEngine,
    ColorGradingConfig,
    STUDIO_PRESETS,
)
from src.processing.mouth_preservation import OralCavityPreserver
from src.targets.identity_fusion import IdentityEmbeddingFuser
from src.processing.color_correction import (
    _compute_channel_stats,
    reinhard_color_transfer,
    apply_color_correction,
)
from src.processing.enhancement import inject_original_skin_texture
from src.core.config_loader import load_all_configs
from src.pipeline.realtime_pipeline import RealTimePipeline


# ---------------------------------------------------------------------------
# 1. Color Grading Tests
# ---------------------------------------------------------------------------

def test_color_grading_neutral():
    engine = ColorGradingEngine()
    test_img = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
    graded = engine.apply(test_img)
    # Neutral config should return identical array
    assert np.array_equal(test_img, graded)


def test_color_grading_adjustments():
    engine = ColorGradingEngine()
    test_img = np.full((64, 64, 3), 128, dtype=np.uint8)

    cfg = ColorGradingConfig(
        enabled=True,
        exposure=20.0,
        contrast=1.2,
        saturation=1.2,
        temperature=15.0,
        tint=-5.0,
        gamma=0.9,
    )
    graded = engine.apply(test_img, cfg)
    assert graded.shape == test_img.shape
    assert graded.dtype == np.uint8
    # With exposure +20 and contrast 1.2, value should change
    assert not np.array_equal(test_img, graded)


def test_color_grading_presets():
    engine = ColorGradingEngine()
    test_img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)

    for preset_name in engine.get_preset_names():
        preset_cfg = engine.get_preset(preset_name)
        graded = engine.apply(test_img, preset_cfg)
        assert graded.shape == test_img.shape
        assert graded.dtype == np.uint8


# ---------------------------------------------------------------------------
# 2. Identity Embedding Fusion Tests
# ---------------------------------------------------------------------------

def test_identity_fusion_normalization():
    raw_vec = np.random.randn(512).astype(np.float32)
    norm_vec = IdentityEmbeddingFuser.l2_normalize(raw_vec)
    assert np.isclose(np.linalg.norm(norm_vec), 1.0, atol=1e-5)


def test_identity_fusion_cosine_similarity():
    v1 = IdentityEmbeddingFuser.l2_normalize(np.ones(512, dtype=np.float32))
    v2 = IdentityEmbeddingFuser.l2_normalize(np.ones(512, dtype=np.float32))
    sim = IdentityEmbeddingFuser.cosine_similarity(v1, v2)
    assert np.isclose(sim, 1.0, atol=1e-5)


def test_identity_fusion_multi_samples():
    base = IdentityEmbeddingFuser.l2_normalize(np.random.randn(512).astype(np.float32))
    # Create 2 slightly perturbed variants of base identity
    noise1 = IdentityEmbeddingFuser.l2_normalize(np.random.randn(512).astype(np.float32)) * 0.08
    noise2 = IdentityEmbeddingFuser.l2_normalize(np.random.randn(512).astype(np.float32)) * 0.08
    noisy_1 = IdentityEmbeddingFuser.l2_normalize(base + noise1)
    noisy_2 = IdentityEmbeddingFuser.l2_normalize(base + noise2)
    # Create 1 outlier vector (-base has negative similarity)
    outlier = IdentityEmbeddingFuser.l2_normalize(-base)

    fused = IdentityEmbeddingFuser.fuse_embeddings([noisy_1, noisy_2, outlier])
    assert fused.shape == (512,)
    assert np.isclose(np.linalg.norm(fused), 1.0, atol=1e-5)
    # Fused should correlate highly with the base identity since outlier was rejected
    sim = IdentityEmbeddingFuser.cosine_similarity(fused, base)
    assert sim > 0.85


def test_identity_interpolation_slerp():
    v0 = IdentityEmbeddingFuser.l2_normalize(np.random.randn(512).astype(np.float32))
    v1 = IdentityEmbeddingFuser.l2_normalize(np.random.randn(512).astype(np.float32))

    midpoint = IdentityEmbeddingFuser.interpolate_identities(v0, v1, alpha=0.5)
    assert midpoint.shape == (512,)
    assert np.isclose(np.linalg.norm(midpoint), 1.0, atol=1e-5)

    at_zero = IdentityEmbeddingFuser.interpolate_identities(v0, v1, alpha=0.0)
    assert np.isclose(IdentityEmbeddingFuser.cosine_similarity(at_zero, v0), 1.0, atol=1e-4)

    at_one = IdentityEmbeddingFuser.interpolate_identities(v0, v1, alpha=1.0)
    assert np.isclose(IdentityEmbeddingFuser.cosine_similarity(at_one, v1), 1.0, atol=1e-4)


def test_identity_fusion_create_target():
    fused_emb = IdentityEmbeddingFuser.l2_normalize(np.random.randn(512).astype(np.float32))
    ref_img = np.full((128, 128, 3), 150, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        target = IdentityEmbeddingFuser.create_fused_target(
            target_id="test_fused",
            display_name="Test Fused Identity",
            category="custom",
            fused_embedding=fused_emb,
            reference_image=ref_img,
            output_base_dir=tmpdir,
        )
        assert target.target_id == "test_fused"
        assert os.path.isfile(target.reference_image_path)
        assert os.path.isfile(os.path.join(tmpdir, "custom", "test_fused", "face.npy"))
        assert target.embedding is not None


# ---------------------------------------------------------------------------
# 3. Mouth & Oral Cavity Preservation Tests
# ---------------------------------------------------------------------------

def test_mouth_preservation_mask():
    preserver = OralCavityPreserver()
    mask = preserver.compute_mouth_region_mask((128, 128))
    assert mask.shape == (128, 128)
    assert mask.dtype == np.float32
    assert mask.max() > 0.0
    assert mask.min() >= 0.0

    # Test 512x512 resolution mask scaling
    mask_512 = preserver.compute_mouth_region_mask((512, 512))
    assert mask_512.shape == (512, 512)


def test_mouth_preservation_blending():
    preserver = OralCavityPreserver(default_strength=0.70)
    orig = np.random.randint(50, 220, (128, 128, 3), dtype=np.uint8)
    # Add teeth-like high frequency contrast in oral region
    orig[90:105, 50:78] = 230  # bright teeth
    orig[105:115, 50:78] = 30  # dark cavity

    swap = np.full((128, 128, 3), 120, dtype=np.uint8)
    enhanced = preserver.preserve_mouth_fidelity(orig, swap, strength=0.70)
    assert enhanced.shape == swap.shape
    assert enhanced.dtype == np.uint8


# ---------------------------------------------------------------------------
# 4. Resolution Invariance Tests
# ---------------------------------------------------------------------------

def test_compute_channel_stats_different_resolutions():
    # Target image 512x512, mask 128x128
    tgt_lab = np.random.randint(20, 240, (512, 512, 3), dtype=np.uint8).astype(np.float32)
    mask_128 = np.ones((128, 128), dtype=np.float32)

    mean, std = _compute_channel_stats(tgt_lab, weights=mask_128)
    assert mean.shape == (3,)
    assert std.shape == (3,)


def test_reinhard_color_transfer_different_resolutions():
    src_128 = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    tgt_512 = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    mask_128 = np.ones((128, 128), dtype=np.float32)

    transferred = reinhard_color_transfer(src_128, tgt_512, blend_ratio=0.70, mask=mask_128)
    assert transferred.shape == (512, 512, 3)
    assert transferred.dtype == np.uint8


def test_inject_skin_texture_with_occlusion_matte():
    orig = np.random.randint(60, 200, (128, 128, 3), dtype=np.uint8)
    swap = np.full((128, 128, 3), 120, dtype=np.uint8)
    occ = np.zeros((128, 128), dtype=np.float32)
    occ[40:80, 40:80] = 1.0  # hand / glasses occlusion

    injected = inject_original_skin_texture(
        original_crop=orig,
        swapped_crop=swap,
        amount=0.40,
        occlusion_matte=occ,
    )
    assert injected.shape == (128, 128, 3)
    assert injected.dtype == np.uint8


# ---------------------------------------------------------------------------
# 5. Pipeline Phase 8 Integration
# ---------------------------------------------------------------------------

def test_pipeline_phase8_components():
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    assert hasattr(pipeline, "color_grader")
    assert isinstance(pipeline.color_grader, ColorGradingEngine)
    assert hasattr(pipeline, "mouth_preserver")
    assert isinstance(pipeline.mouth_preserver, OralCavityPreserver)

    # Test config access
    cg_cfg = pipeline._get_active_color_grading_config()
    assert isinstance(cg_cfg, ColorGradingConfig)
