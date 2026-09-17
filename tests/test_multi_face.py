"""
Unit and Integration Tests for Multi-Face Tracking and Swapping.
"""

import pytest
import numpy as np
import cv2

from src.detection.face_detector import FaceData, FaceDetector
from src.tracking.face_tracker import FaceTracker, compute_bbox_iou
from src.pipeline.realtime_pipeline import RealTimePipeline, PipelineResult
from src.core.config_loader import load_all_configs


def test_bbox_iou_computation():
    box1 = (10, 10, 50, 50)
    box2 = (10, 10, 50, 50)
    assert pytest.approx(compute_bbox_iou(box1, box2), 0.001) == 1.0

    box3 = (100, 100, 150, 150)
    assert compute_bbox_iou(box1, box3) == 0.0

    # Partial overlap (20x40 overlap in 40x40 boxes)
    box4 = (30, 10, 70, 50)
    iou = compute_bbox_iou(box1, box4)
    assert 0.2 < iou < 0.4


class MockMultiFaceDetector(FaceDetector):
    def __init__(self, faces):
        self._mock_faces = faces
        self._is_ready = True
        self.backend_type = "mock"

    def detect(self, frame, max_faces=5):
        return self._mock_faces[:max_faces]


def test_multi_face_tracker_association():
    kps1 = np.array([[50, 50], [70, 50], [60, 65], [55, 80], [65, 80]], dtype=np.float32)
    kps2 = np.array([[200, 50], [220, 50], [210, 65], [205, 80], [215, 80]], dtype=np.float32)

    face1 = FaceData(bbox=(40, 40, 80, 90), score=0.95, landmarks=kps1)
    face2 = FaceData(bbox=(190, 40, 230, 90), score=0.92, landmarks=kps2)

    detector = MockMultiFaceDetector([face1, face2])
    tracker = FaceTracker(detector)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Frame 1: initial detection
    tracked = tracker.update_all(frame)
    assert len(tracked) == 2
    id1 = tracked[0].track_id
    id2 = tracked[1].track_id
    assert id1 != id2

    # Frame 2: slight displacement
    kps1_shifted = kps1 + 2.0
    face1_shifted = FaceData(bbox=(42, 42, 82, 92), score=0.95, landmarks=kps1_shifted)
    detector._mock_faces = [face1_shifted, face2]

    tracked2 = tracker.update_all(frame)
    assert len(tracked2) == 2
    # Verify track IDs remained stable
    ids_now = {f.track_id for f in tracked2}
    assert id1 in ids_now
    assert id2 in ids_now


def test_multi_face_pipeline_modes():
    a_cfg, m_cfg, t_cfg = load_all_configs()
    pipeline = RealTimePipeline(a_cfg, m_cfg, t_cfg)

    # Test setting modes
    pipeline.set_multi_face_mode("all")
    assert pipeline.multi_face_mode == "all"

    pipeline.set_multi_face_mode("primary")
    assert pipeline.multi_face_mode == "primary"

    # Test target mapping
    pipeline.set_target_for_track(1, "prabhas")
    pipeline.set_target_for_track(2, "chiranjeevi")
    assert pipeline.target_mappings[1] == "prabhas"
    assert pipeline.target_mappings[2] == "chiranjeevi"

    pipeline.set_target_for_track(1, None)
    assert 1 not in pipeline.target_mappings
