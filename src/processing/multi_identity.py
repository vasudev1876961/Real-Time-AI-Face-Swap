"""
Multi-Subject Identity Binding Matrix Manager (Phase 17).
Provides dynamic individual target face assignment for multi-face scenes,
supporting primary focus, spatial left-to-right ordering, sequential queues,
and persistent track-to-target affinity matrices.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.detection.face_detector import FaceData
from src.targets.target_loader import TargetFace
from src.targets.target_manager import TargetManager
from src.utils.logger import get_logger

logger = get_logger("MultiIdentityMatrix")


class MultiIdentityMatrixManager:
    """
    Coordinates multi-subject identity assignments across video tracks.
    Eliminates ambiguity when multiple actors appear simultaneously in camera frame.
    """

    MODES = ["primary", "all", "sequential", "spatial_lr", "custom_matrix"]

    def __init__(self, target_manager: Optional[TargetManager] = None):
        self.target_manager = target_manager
        self.mode: str = "primary"
        self.custom_mappings: Dict[int, str] = {}  # track_id -> target_id
        self.spatial_target_ids: List[str] = []   # Ordered targets for spatial_lr / sequential
        
        # Track persistence cache: track_id -> (target_id, last_seen_timestamp)
        self._track_memory: Dict[int, Tuple[str, float]] = {}
        self.memory_ttl_sec: float = 3.0  # Retain binding for 3s if face temporarily lost

    def reset(self) -> None:
        """Clears volatile track persistence bindings."""
        self._track_memory.clear()

    def set_config(
        self,
        mode: Optional[str] = None,
        custom_mappings: Optional[Dict[int, str]] = None,
        spatial_target_ids: Optional[List[str]] = None,
    ) -> None:
        """Updates configuration for multi-identity binding."""
        if mode is not None:
            clean_mode = mode.strip().lower()
            if clean_mode in self.MODES:
                self.mode = clean_mode
            else:
                logger.warning(f"Unknown multi-identity mode '{mode}'. Keeping '{self.mode}'.")
        if custom_mappings is not None:
            self.custom_mappings = {int(k): str(v) for k, v in custom_mappings.items()}
        if spatial_target_ids is not None:
            self.spatial_target_ids = [str(t) for t in spatial_target_ids if str(t).strip()]

    def set_track_mapping(self, track_id: int, target_id: Optional[str]) -> None:
        """Explicitly associates a track ID with a target ID."""
        if target_id is None:
            self.custom_mappings.pop(track_id, None)
            self._track_memory.pop(track_id, None)
        else:
            self.custom_mappings[track_id] = target_id
            self._track_memory[track_id] = (target_id, time.time())

    def resolve_bindings(
        self,
        detected_faces: List[FaceData],
        default_target: Optional[TargetFace],
    ) -> List[Tuple[FaceData, Optional[TargetFace]]]:
        """
        Resolves the appropriate TargetFace for each detected face in the frame.
        Returns a list of (FaceData, Optional[TargetFace]) pairs.
        """
        if not detected_faces:
            return []

        now = time.time()
        # Clean up stale memory entries older than memory_ttl_sec
        stale_tracks = [
            tid for tid, (_, t_seen) in self._track_memory.items()
            if now - t_seen > self.memory_ttl_sec
        ]
        for tid in stale_tracks:
            self._track_memory.pop(tid, None)

        results: List[Tuple[FaceData, Optional[TargetFace]]] = []

        if self.mode == "primary":
            # Only primary (largest / first) face gets the default target
            for idx, face in enumerate(detected_faces):
                if idx == 0:
                    results.append((face, default_target))
                else:
                    results.append((face, None))
            return results

        if self.mode == "all":
            # Every detected face receives the active default target
            for face in detected_faces:
                results.append((face, default_target))
            return results

        if self.mode == "custom_matrix":
            # Look up explicit mapping or memory, falling back to default
            for face in detected_faces:
                track_id = face.track_id or 1
                assigned_target_id = self.custom_mappings.get(track_id)
                if not assigned_target_id and track_id in self._track_memory:
                    assigned_target_id = self._track_memory[track_id][0]

                target = None
                if assigned_target_id and self.target_manager:
                    target = self.target_manager.get_target_by_id(assigned_target_id)
                if target is None:
                    target = default_target

                if target is not None:
                    self._track_memory[track_id] = (target.target_id, now)

                results.append((face, target))
            return results

        if self.mode == "spatial_lr":
            # Sort faces from left to right by bounding box centroid x
            # and map to spatial_target_ids in order
            sorted_faces = sorted(
                detected_faces,
                key=lambda f: (f.bbox[0] + f.bbox[2]) / 2.0 if f.bbox else 0.0
            )

            available_target_ids = list(self.spatial_target_ids)
            if not available_target_ids and default_target:
                available_target_ids = [default_target.target_id]

            for rank, face in enumerate(sorted_faces):
                track_id = face.track_id or (rank + 1)
                assigned_target_id = None

                # Check if this track already has a stable binding in memory
                if track_id in self._track_memory:
                    assigned_target_id = self._track_memory[track_id][0]
                elif rank < len(available_target_ids):
                    assigned_target_id = available_target_ids[rank]
                elif available_target_ids:
                    assigned_target_id = available_target_ids[-1]

                target = None
                if assigned_target_id and self.target_manager:
                    target = self.target_manager.get_target_by_id(assigned_target_id)
                if target is None:
                    target = default_target

                if target is not None:
                    self._track_memory[track_id] = (target.target_id, now)

                results.append((face, target))

            # Restore original detected order
            face_map = {id(f): t for f, t in results}
            return [(f, face_map.get(id(f))) for f in detected_faces]

        if self.mode == "sequential":
            # Assign sequential targets based on appearance order
            available_target_ids = list(self.spatial_target_ids)
            if not available_target_ids and default_target:
                available_target_ids = [default_target.target_id]

            for idx, face in enumerate(detected_faces):
                track_id = face.track_id or (idx + 1)
                assigned_target_id = None
                if track_id in self._track_memory:
                    assigned_target_id = self._track_memory[track_id][0]
                elif idx < len(available_target_ids):
                    assigned_target_id = available_target_ids[idx]
                elif available_target_ids:
                    assigned_target_id = available_target_ids[idx % len(available_target_ids)]

                target = None
                if assigned_target_id and self.target_manager:
                    target = self.target_manager.get_target_by_id(assigned_target_id)
                if target is None:
                    target = default_target

                if target is not None:
                    self._track_memory[track_id] = (target.target_id, now)

                results.append((face, target))
            return results

        # Fallback: treat as primary
        return [(detected_faces[0], default_target)] + [(f, None) for f in detected_faces[1:]]

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns diagnostic telemetry for the multi-identity binding matrix."""
        return {
            "mode": self.mode,
            "custom_mappings_count": len(self.custom_mappings),
            "custom_mappings": dict(self.custom_mappings),
            "spatial_target_ids": list(self.spatial_target_ids),
            "active_persisted_tracks": len(self._track_memory),
            "persisted_tracks": {str(k): v[0] for k, v in self._track_memory.items()},
        }
