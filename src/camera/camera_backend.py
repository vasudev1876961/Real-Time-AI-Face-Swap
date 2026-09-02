"""
Threaded Camera Backend for Low-Latency Real-Time Frame Capture.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("CameraBackend")


@dataclass
class FramePacket:
    """Represents a captured camera frame with timing metadata."""
    frame: np.ndarray
    frame_index: int
    timestamp: float
    width: int
    height: int


class CameraBackend:
    """
    Low-latency threaded camera reader that continuously polls OpenCV VideoCapture
    and provides the most recent frame without buffering lag.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        backend_name: str = "AUTO",
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.backend_name = backend_name.upper()

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        self._latest_packet: Optional[FramePacket] = None
        self._frame_count = 0
        self._is_running = False
        self._last_read_time = 0.0

    def _get_opencv_backend(self) -> int:
        """Determines appropriate OpenCV backend API flag."""
        if self.backend_name == "DSHOW":
            return cv2.CAP_DSHOW
        elif self.backend_name == "MSMF":
            return cv2.CAP_MSMF
        elif self.backend_name == "V4L2":
            return cv2.CAP_V4L2
        return cv2.CAP_ANY

    def open(self) -> bool:
        """Opens the camera device and starts the background capture thread."""
        if self._is_running:
            self.close()

        logger.info(f"Opening camera index {self.camera_index} (Target: {self.width}x{self.height} @ {self.target_fps} FPS)...")
        api_preference = self._get_opencv_backend()

        # On Windows, cv2.CAP_DSHOW is significantly faster to initialize and avoids 5-second MSMF delays
        if api_preference == cv2.CAP_ANY and hasattr(cv2, "CAP_DSHOW"):
            try:
                self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            except Exception:
                self._cap = cv2.VideoCapture(self.camera_index)
        else:
            self._cap = cv2.VideoCapture(self.camera_index, api_preference)

        if not self._cap or not self._cap.isOpened():
            # Fallback without specific API
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            logger.warning(f"Could not open camera index {self.camera_index}.")
            self._cap = None
            return False

        # Set hardware properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened successfully: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        self._stop_event.clear()
        self._is_running = True
        self._frame_count = 0
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name=f"CamWorker-{self.camera_index}")
        self._thread.start()
        return True

    def _capture_worker(self) -> None:
        """Continuously reads frames from camera into single-item buffer."""
        consecutive_failures = 0
        while not self._stop_event.is_set():
            if not self._cap or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    logger.warning("Camera stream lost (too many consecutive read failures).")
                    break
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            now = time.time()
            self._frame_count += 1
            h, w = frame.shape[:2]

            packet = FramePacket(
                frame=frame,
                frame_index=self._frame_count,
                timestamp=now,
                width=w,
                height=h,
            )

            with self._lock:
                self._latest_packet = packet

        self._is_running = False

    def read_latest(self) -> Optional[FramePacket]:
        """Returns the most recent captured frame packet, or None if unavailable."""
        with self._lock:
            return self._latest_packet

    def is_opened(self) -> bool:
        """Returns True if the camera is active and delivering frames."""
        return self._is_running and (self._cap is not None and self._cap.isOpened())

    def close(self) -> None:
        """Stops capture thread and releases video device resources."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        if self._cap:
            try:
                self._cap.release()
            except Exception as e:
                logger.debug(f"Error releasing camera: {e}")
            self._cap = None

        self._is_running = False
        with self._lock:
            self._latest_packet = None
        logger.info(f"Camera {self.camera_index} released.")
