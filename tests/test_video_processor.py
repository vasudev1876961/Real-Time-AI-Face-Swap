"""
Unit and Integration Tests for Offline Media Processing Engine.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from src.pipeline.video_processor import VideoFileProcessor
from src.pipeline.realtime_pipeline import RealTimePipeline
from src.core.config_loader import load_all_configs


def test_video_processor_process_image():
    test_img = "outputs/captures/capture_2026-08-24_21-47-58_hero_001.jpg"
    if not os.path.isfile(test_img):
        pytest.skip("Reference capture not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_img = os.path.join(tmpdir, "swapped.jpg")
        processor = VideoFileProcessor()

        res = processor.process_image(test_img, out_img, target_id="prabhas", enhance_strength=0.50)
        assert res is not None
        assert os.path.isfile(out_img)
        assert os.path.getsize(out_img) > 0


def test_video_processor_synthetic_video():
    test_img = "outputs/captures/capture_2026-08-24_21-47-58_hero_001.jpg"
    if not os.path.isfile(test_img):
        pytest.skip("Reference capture not found")

    frame = cv2.imread(test_img)
    h, w = frame.shape[:2]

    with tempfile.TemporaryDirectory() as tmpdir:
        in_video = os.path.join(tmpdir, "synth_input.mp4")
        out_video = os.path.join(tmpdir, "synth_output.mp4")

        # Create 5-frame synthetic video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(in_video, fourcc, 15.0, (w, h))
        for _ in range(5):
            writer.write(frame)
        writer.release()

        assert os.path.isfile(in_video)

        processor = VideoFileProcessor()
        frames_tracked = []

        def on_prog(curr, total, fps, eta):
            frames_tracked.append(curr)

        summary = processor.process_video(
            in_video,
            out_video,
            target_id="prabhas",
            enhance_strength=0.40,
            progress_callback=on_prog,
        )

        assert os.path.isfile(out_video)
        assert os.path.getsize(out_video) > 0
        assert summary["total_frames"] == 5
        assert len(frames_tracked) == 5
