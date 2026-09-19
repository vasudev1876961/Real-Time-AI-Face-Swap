"""
Landmark Detector Adapter and Keypoint Processor.
"""

from typing import Optional, List, Tuple
import numpy as np
from src.detection.face_detector import get_face_detector, FaceData
from src.detection.face_landmarks import FaceLandmarks


class LandmarkDetector:
    """
    Adapter for extracting 5-point facial keypoints from aligned crops or full frames.
    """

    def __init__(self):
        self.detector = get_face_detector()

    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detects 5 facial keypoints for the primary face in image."""
        if image is None or image.size == 0:
            return None
        faces = self.detector.detect(image)
        if not faces:
            return None
        return faces[0].landmarks

    def smooth_landmarks(
        self,
        current: np.ndarray,
        previous: Optional[np.ndarray],
        alpha: float = 0.65,
    ) -> np.ndarray:
        """Applies EMA smoothing between consecutive landmark sets."""
        return FaceLandmarks.smooth_landmarks_ema(current=current, previous=previous, alpha=alpha)


__all__ = ["LandmarkDetector"]
