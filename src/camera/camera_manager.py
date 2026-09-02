"""
Camera Manager for Device Discovery, Lifecycle Management, and Safe Dynamic Switching.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Callable
import cv2
import numpy as np

from src.camera.camera_backend import CameraBackend, FramePacket
from src.core.config_loader import CameraConfig
from src.utils.logger import get_logger

logger = get_logger("CameraManager")


@dataclass
class CameraDeviceInfo:
    index: int
    name: str
    is_available: bool


class CameraManager:
    """
    Manages active camera session, provides device discovery, and guarantees
    safe transitions when switching video sources.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.active_index = self.config.camera_index
        self.backend: Optional[CameraBackend] = None
        self._on_camera_switched_callbacks: List[Callable[[int], None]] = []

    def register_switch_callback(self, callback: Callable[[int], None]) -> None:
        """Registers a callback to be notified when camera switches (e.g. to reset tracking)."""
        self._on_camera_switched_callbacks.append(callback)

    @staticmethod
    def discover_cameras(max_probe: int = 4) -> List[CameraDeviceInfo]:
        """
        Discovers connected video capture devices by non-blocking index probe.
        """
        available: List[CameraDeviceInfo] = []
        for idx in range(max_probe):
            try:
                # Use DSHOW on Windows for fast probe
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(idx)
                if cap is not None and cap.isOpened():
                    is_readable, _ = cap.read()
                    cap.release()
                    available.append(CameraDeviceInfo(
                        index=idx,
                        name=f"Camera {idx}" + (" (Default/Front)" if idx == 0 else ""),
                        is_available=True,
                    ))
                else:
                    if cap:
                        cap.release()
            except Exception as e:
                logger.debug(f"Camera probe index {idx} failed: {e}")
        
        if not available:
            logger.info("No physical webcams discovered during probe.")
        else:
            logger.info(f"Discovered cameras: {[c.name for c in available]}")
        return available

    def start(self, camera_index: Optional[int] = None) -> bool:
        """Starts capturing from the selected camera index."""
        target_index = self.active_index if camera_index is None else camera_index
        self.active_index = target_index

        if self.backend and self.backend.is_opened():
            self.backend.close()

        self.backend = CameraBackend(
            camera_index=target_index,
            width=self.config.width,
            height=self.config.height,
            target_fps=self.config.fps,
            backend_name=self.config.backend,
        )

        success = self.backend.open()
        if success:
            logger.info(f"CameraManager: Camera index {target_index} active.")
        else:
            logger.warning(f"CameraManager: Failed to start camera index {target_index}.")
        return success

    def switch_camera(self, new_index: int) -> bool:
        """
        Switches camera cleanly without app restart:
        1. Stop current capture
        2. Notify reset callbacks
        3. Open new camera
        """
        logger.info(f"Switching camera from {self.active_index} to {new_index}...")
        self.stop()

        # Trigger reset callbacks (e.g. tracker reset)
        for cb in self._on_camera_switched_callbacks:
            try:
                cb(new_index)
            except Exception as e:
                logger.error(f"Error in camera switch callback: {e}")

        success = self.start(new_index)
        if not success:
            logger.error(f"Failed to switch to camera {new_index}.")
        return success

    def get_latest_frame(self) -> Optional[FramePacket]:
        """Returns the latest captured frame packet."""
        if not self.backend:
            return None
        return self.backend.read_latest()

    def is_running(self) -> bool:
        """Checks if active camera is running."""
        return self.backend is not None and self.backend.is_opened()

    def stop(self) -> None:
        """Stops active camera session."""
        if self.backend:
            self.backend.close()
            self.backend = None
            logger.info("CameraManager: Active camera stopped.")
