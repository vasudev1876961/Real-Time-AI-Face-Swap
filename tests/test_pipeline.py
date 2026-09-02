"""
Unit and Integration Tests for Real-Time Pipeline Orchestration.
"""

import pytest
import numpy as np
import cv2

from src.pipeline.realtime_pipeline import RealTimePipeline, PipelineResult
from src.core.config_loader import load_all_configs


def test_pipeline_lifecycle():
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    # Test target list & selection
    targets = pipeline.target_manager.list_targets()
    assert isinstance(targets, list)

    if targets:
        first_id = targets[0].target_id
        assert pipeline.select_target(first_id) is True
        assert pipeline.get_selected_target() is not None
        assert pipeline.get_selected_target().target_id == first_id

        # Deselect
        pipeline.select_target(None)
        assert pipeline.get_selected_target() is None


def test_pipeline_frame_processing():
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    # Empty frame test
    res = pipeline.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert isinstance(res, PipelineResult)
    assert res.rendered_frame is not None

    # Synthetic image test
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res2 = pipeline.process_frame(frame)
    assert res2.rendered_frame.shape == (480, 640, 3)
    assert "fps" in res2.metrics_summary
    assert "total_latency_ms" in res2.metrics_summary


def test_tracker_reset():
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)
    pipeline.reset_tracker()
    assert pipeline.tracker.active_track is None


def test_pipeline_face_swapping_end_to_end():
    import os
    import time
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    # Use existing target reference image
    ref_path = "faces/actresses/aruna_irani/reference.jpg"
    if os.path.exists(ref_path):
        img = cv2.imread(ref_path)
        assert img is not None
        assert pipeline.select_target("aruna_irani") is True

        # Process a few frames to allow async worker to produce swapped crop
        for _ in range(5):
            res = pipeline.process_frame(img)
            time.sleep(0.05)

        assert res.rendered_frame.shape == img.shape
        assert res.status_message != "Pipeline Error"

