"""
Lightweight and Reliable Multi-Face Tracker with Spatial Association,
Temporal Smoothing, and Dynamic Re-Detection.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import cv2
import numpy as np

from src.detection.face_detector import FaceData, FaceDetector
from src.detection.face_landmarks import FaceLandmarks
from src.core.config_loader import PerformanceConfig
from src.utils.logger import get_logger

logger = get_logger("FaceTracker")


def compute_bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    box2_area = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union_area = float(box1_area + box2_area - inter_area)
    return inter_area / union_area if union_area > 0 else 0.0


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
    Coordinates multi-face detection and frame-to-frame tracking to maintain high FPS
    while tracking multiple individuals concurrently with stable track IDs and EMA smoothing.
    """

    def __init__(
        self,
        detector: FaceDetector,
        config: Optional[PerformanceConfig] = None,
    ):
        self.detector = detector
        self.config = config or PerformanceConfig()
        self.active_tracks: Dict[int, TrackedFace] = {}
        self.focused_track_id: Optional[int] = None
        self._frame_counter = 0
        self._next_track_id = 1
        self._max_missed_frames = 10

    @property
    def active_track(self) -> Optional[TrackedFace]:
        """
        Backwards-compatible accessor for primary active track.
        Returns the focused track if set, or the largest tracked face.
        """
        if not self.active_tracks:
            return None
        if self.focused_track_id is not None and self.focused_track_id in self.active_tracks:
            return self.active_tracks[self.focused_track_id]
        # Return track with largest face area
        sorted_tracks = sorted(
            self.active_tracks.values(),
            key=lambda t: t.face_data.area if t.face_data else 0,
            reverse=True,
        )
        return sorted_tracks[0] if sorted_tracks else None

    @active_track.setter
    def active_track(self, track: Optional[TrackedFace]) -> None:
        """Backwards-compatible setter for primary track."""
        if track is None:
            self.active_tracks.clear()
            self.focused_track_id = None
        else:
            self.active_tracks[track.track_id] = track
            self.focused_track_id = track.track_id

    def reset(self) -> None:
        """Resets tracker state completely (e.g. on camera switch)."""
        self.active_tracks.clear()
        self.focused_track_id = None
        self._frame_counter = 0
        logger.info("Face tracker state reset.")

    def set_focused_track(self, track_id: Optional[int]) -> None:
        """Designates a specific track ID as primary/focused."""
        self.focused_track_id = track_id

    def get_active_tracks(self) -> List[TrackedFace]:
        """Returns list of currently active tracks."""
        return list(self.active_tracks.values())

    def update(self, frame: np.ndarray) -> Optional[FaceData]:
        """
        Processes a frame and returns the primary tracked face or None.
        Maintains 100% backwards-compatibility with single-face callers.
        """
        faces = self.update_all(frame)
        return faces[0] if faces else None

    def update_all(self, frame: np.ndarray, max_faces: Optional[int] = None) -> List[FaceData]:
        """
        Processes a frame and returns all tracked faces with stable IDs and smoothed landmarks.
        """
        if frame is None or frame.size == 0:
            return []

        self._frame_counter += 1
        cfg_max = getattr(self.config, "max_faces", 5)
        limit_faces = max_faces if max_faces is not None else cfg_max

        # Determine if re-detection is required
        has_active = len(self.active_tracks) > 0
        any_missed = any(t.missed_frames > 0 for t in self.active_tracks.values())
        interval = max(1, getattr(self.config, "detection_interval", 3))
        need_detection = (not has_active) or (self._frame_counter % interval == 0) or any_missed

        if need_detection:
            return self._detect_and_associate(frame, max_faces=limit_faces)
        else:
            return self._track_step_all()

    def _detect_and_associate(self, frame: np.ndarray, max_faces: int = 5) -> List[FaceData]:
        """Runs face detection and associates detected faces with existing active tracks."""
        detected_faces = self.detector.detect(frame, max_faces=max_faces)

        # Build cost / match matrix between existing tracks and detected faces
        existing_ids = list(self.active_tracks.keys())
        matched_tracks: Dict[int, FaceData] = {}
        unmatched_detections: List[FaceData] = []
        used_det_indices = set()

        if existing_ids and detected_faces:
            # Match by IoU or landmark proximity
            for track_id in existing_ids:
                track = self.active_tracks[track_id]
                best_iou = 0.0
                best_idx = -1

                for idx, det in enumerate(detected_faces):
                    if idx in used_det_indices:
                        continue
                    iou = compute_bbox_iou(track.face_data.bbox, det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                # Also test distance if IoU is zero (e.g. fast movement)
                if best_idx == -1 or best_iou < 0.20:
                    min_dist = float("inf")
                    for idx, det in enumerate(detected_faces):
                        if idx in used_det_indices:
                            continue
                        dist = np.linalg.norm(
                            np.array(track.face_data.center) - np.array(det.center)
                        )
                        # Threshold based on face dimension
                        max_allowed_dist = max(track.face_data.width, track.face_data.height) * 1.5
                        if dist < max_allowed_dist and dist < min_dist:
                            min_dist = dist
                            best_idx = idx

                if best_idx != -1:
                    used_det_indices.add(best_idx)
                    matched_tracks[track_id] = detected_faces[best_idx]

        # Remaining detections become new tracks
        for idx, det in enumerate(detected_faces):
            if idx not in used_det_indices:
                unmatched_detections.append(det)

        # Update matched tracks
        for track_id, det in matched_tracks.items():
            track = self.active_tracks[track_id]
            track.missed_frames = 0
            track.total_tracked_frames += 1
            det.track_id = track_id

            if getattr(self.config, "enable_smoothing", True) and det.landmarks is not None:
                smoothed_kps = FaceLandmarks.smooth_landmarks_ema(
                    current=det.landmarks,
                    previous=track.smoothed_landmarks,
                    alpha=getattr(self.config, "ema_alpha", 0.65),
                )
                track.smoothed_landmarks = smoothed_kps
                det.landmarks = smoothed_kps
                det.kps = smoothed_kps

            track.face_data = det

        # Register new tracks for unmatched detections
        for det in unmatched_detections:
            track_id = self._next_track_id
            self._next_track_id += 1
            det.track_id = track_id
            self.active_tracks[track_id] = TrackedFace(
                track_id=track_id,
                face_data=det,
                missed_frames=0,
                total_tracked_frames=1,
                smoothed_landmarks=det.landmarks.copy() if det.landmarks is not None else None,
                smoothed_bbox=det.bbox,
            )

        # Handle unmatched existing tracks (increment missed_frames or purge)
        dead_ids = []
        for track_id, track in self.active_tracks.items():
            if track_id not in matched_tracks and track_id not in [d.track_id for d in unmatched_detections]:
                track.missed_frames += 1
                if track.missed_frames > self._max_missed_frames:
                    dead_ids.append(track_id)

        for track_id in dead_ids:
            del self.active_tracks[track_id]
            if self.focused_track_id == track_id:
                self.focused_track_id = None

        return self._get_sorted_active_faces()

    def _track_step_all(self) -> List[FaceData]:
        """Returns all smoothed tracked faces during inter-detection frames."""
        return self._get_sorted_active_faces()

    def _get_sorted_active_faces(self) -> List[FaceData]:
        """Returns list of active FaceData objects sorted with focused face first, then by area."""
        active_list = [t.face_data for t in self.active_tracks.values() if t.face_data is not None]
        if not active_list:
            return []

        def sort_key(f: FaceData):
            is_focused = 1 if (self.focused_track_id is not None and f.track_id == self.focused_track_id) else 0
            return (is_focused, f.area)

        active_list.sort(key=sort_key, reverse=True)
        return active_list
