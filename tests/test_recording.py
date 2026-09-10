"""
Unit Tests for Asynchronous Recording and Snapshot Capture Subsystem.
"""

import os
import time
import json
import tempfile
import numpy as np
import pytest

from src.recording.video_recorder import ThreadedVideoRecorder
from src.recording.capture import SnapshotCaptureManager


def test_threaded_video_recorder_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "test_record.mp4")
        recorder = ThreadedVideoRecorder(queue_capacity=50)

        assert not recorder.is_recording()

        success = recorder.start_recording(
            vid_path,
            fps=30.0,
            resolution=(320, 240),
            extra_metadata={"test_run": True},
        )
        assert success
        assert recorder.is_recording()

        # Submit 15 synthetic frames
        for _ in range(15):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            recorder.write_frame(frame)
            time.sleep(0.01)

        summary = recorder.stop_recording()
        assert not recorder.is_recording()
        assert os.path.isfile(vid_path)
        assert os.path.getsize(vid_path) > 0

        # Verify JSON metadata
        meta_path = os.path.join(tmpdir, "test_record.json")
        assert os.path.isfile(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data.get("test_run") is True
            assert data.get("frames_written", 0) >= 10


def test_snapshot_capture_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SnapshotCaptureManager(default_captures_dir=tmpdir)
        frame = np.full((100, 100, 3), 180, dtype=np.uint8)

        saved_path = manager.capture(
            frame,
            target_name="Test Hero",
            target_id="test_hero",
            category="custom",
        )
        assert saved_path is not None
        assert os.path.isfile(saved_path)

        # Check metadata sidecar
        meta_path = os.path.splitext(saved_path)[0] + ".json"
        assert os.path.isfile(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data.get("target_id") == "test_hero"
            assert data.get("target_name") == "Test Hero"
            assert data.get("category") == "custom"
