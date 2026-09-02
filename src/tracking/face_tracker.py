"""
Lightweight and Reliable Face Tracker with Temporal Smoothing and Dynamic Re-Detection.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import cv2
import numpy as np

from src.detection.face_detector import FaceData, FaceDetector
from src.detection.face_landmarks import FaceLandmarks
from src.core.config_loader import PerformanceConfig
from src.utils.logger import get_logger

logger = get_logger("FaceTracker")


@dataclass
class TrackedFace:
    """Represents a temporally tracked face across consecutive video frames."""
    track_id: int
    face_data: FaceData
    missed_frames: int = 0
    total_tracked_frames: int = 1
    smoothed_landmarks: Optional[np.ndarray] = None
    smoothed_bbox: Optional[Tuple[int, int, int, int]] = None


class FaceTracker:
    """
    Coordinates face detection and frame-to-frame tracking to maintain high FPS
    without running heavy detection on every single frame.
    """

    def __init__(
        self,
        detector: FaceDetector,
        config: Optional[PerformanceConfig] = None,
    ):
        self.detector = detector
        self.config = config or PerformanceConfig()
        self.active_track: Optional[TrackedFace] = None
        self._frame_counter = 0
        self._next_track_id = 1
        self._max_missed_frames = 10

    def reset(self) -> None:
        """Resets tracker state completely (e.g. on camera switch)."""
        self.active_track = None
        self._frame_counter = 0
        logger.info("Face tracker state reset.")

    def update(self, frame: np.ndarray) -> Optional[FaceData]:
        """
        Processes a frame, returning the primary tracked face or None.
        Runs full detection when necessary (periodically or when lost).
        """
        if frame is None or frame.size == 0:
            return None

        self._frame_counter += 1
        need_detection = (
            self.active_track is None
            or (self._frame_counter % max(1, self.config.detection_interval) == 0)
            or (self.active_track.missed_frames > 0)
        )

        if need_detection:
            return self._detect_and_associate(frame)
        else:
            return self._track_step(frame)

    def _detect_and_associate(self, frame: np.ndarray) -> Optional[FaceData]:
        """Runs face detection and associates with active track or initiates a new track."""
        faces = self.detector.detect(frame, max_faces=1)

        if not faces:
            if self.active_track is not None:
                self.active_track.missed_frames += 1
                if self.active_track.missed_frames > self._max_missed_frames:
                    self.active_track = None
                    return None
                # Return last known position during temporary occlusion
                return self.active_track.face_data
            return None

        detected_face = faces[0]

        if self.active_track is None:
            # Initiate new track
            track_id = self._next_track_id
            self._next_track_id += 1
            detected_face.track_id = track_id
            self.active_track = TrackedFace(
                track_id=track_id,
                face_data=detected_face,
                missed_frames=0,
                total_tracked_frames=1,
                smoothed_landmarks=detected_face.landmarks.copy() if detected_face.landmarks is not None else None,
                smoothed_bbox=detected_face.bbox,
            )
            return detected_face

        # Smooth existing track
        self.active_track.missed_frames = 0
        self.active_track.total_tracked_frames += 1
        detected_face.track_id = self.active_track.track_id

        if self.config.enable_smoothing and detected_face.landmarks is not None:
            smoothed_kps = FaceLandmarks.smooth_landmarks_ema(
                current=detected_face.landmarks,
                previous=self.active_track.smoothed_landmarks,
                alpha=self.config.ema_alpha,
            )
            self.active_track.smoothed_landmarks = smoothed_kps
            detected_face.landmarks = smoothed_kps
            detected_face.kps = smoothed_kps

        self.active_track.face_data = detected_face
        return detected_face

    def _track_step(self, frame: np.ndarray) -> Optional[FaceData]:
        """
        Lightweight inter-frame tracking step between detections.
        """
        if self.active_track is None or self.active_track.face_data is None:
            return None

        # Return smoothed face data for maximum frame rate
        return self.active_track.face_data
