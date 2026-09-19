"""
Spatial Association and Tracking Utility Functions.
"""

from typing import Tuple
from src.tracking.face_tracker import compute_bbox_iou, TrackedFace, FaceTracker

__all__ = ["compute_bbox_iou", "TrackedFace", "FaceTracker"]
