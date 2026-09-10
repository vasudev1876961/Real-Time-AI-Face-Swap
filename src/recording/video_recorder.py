"""
Asynchronous Threaded Video Recorder.
Decouples video disk I/O from the live camera and rendering loop via a worker queue,
preventing frame drops and UI stutter during recording.
"""

import os
import json
import time
import queue
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("ThreadedVideoRecorder")


class ThreadedVideoRecorder:
    """
    High-throughput non-blocking video recorder.
    Buffers incoming processed frames in an in-memory queue and writes to disk
    in a dedicated worker thread.
    """

    def __init__(self, queue_capacity: int = 150):
        self._queue_capacity = queue_capacity
        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_capacity)
        self._writer: Optional[cv2.VideoWriter] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_recording = False

        self._filepath: str = ""
        self._meta_filepath: str = ""
        self._start_time: float = 0.0
        self._frames_written: int = 0
        self._frames_dropped: int = 0
        self._video_fps: float = 30.0
        self._resolution: Tuple[int, int] = (1280, 720)
        self._extra_metadata: Dict[str, Any] = {}

    def is_recording(self) -> bool:
        """Returns True if recording is currently in progress."""
        return self._is_recording

    def start_recording(
        self,
        filepath: str,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (1280, 720),
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Initializes video writer and launches asynchronous writer thread.
        """
        if self._is_recording:
            logger.warning("Recording already active. Stop current recording first.")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self._filepath = filepath
        self._meta_filepath = os.path.splitext(filepath)[0] + ".json"
        self._video_fps = max(10.0, float(fps))
        self._resolution = resolution
        self._extra_metadata = extra_metadata or {}

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(filepath, fourcc, self._video_fps, resolution)
        if not self._writer.isOpened():
            logger.error(f"Failed to open VideoWriter for '{filepath}'")
            return False

        # Clear queue and flags
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._stop_event.clear()
        self._frames_written = 0
        self._frames_dropped = 0
        self._start_time = time.time()
        self._is_recording = True

        self._worker_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self._worker_thread.start()

        logger.info(f"Threaded recording started: '{filepath}' ({resolution[0]}x{resolution[1]} @ {self._video_fps:.1f} FPS)")
        return True

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Submits a processed video frame to the recording queue without blocking.
        If queue is saturated, drops frame to safeguard camera rendering responsiveness.
        """
        if not self._is_recording or self._stop_event.is_set():
            return False

        try:
            self._frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            self._frames_dropped += 1
            if self._frames_dropped % 30 == 1:
                logger.warning(f"Video recorder queue saturated! Dropped {self._frames_dropped} frames.")
            return False

    def stop_recording(self) -> Dict[str, Any]:
        """
        Stops acquisition, drains remaining buffered frames to disk, and closes files.
        Returns recording summary and generates JSON metadata file.
        """
        if not self._is_recording:
            return {}

        self._is_recording = False
        self._stop_event.set()

        # Wait for worker thread to finish draining queue
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        duration = round(time.time() - self._start_time, 2)
        avg_fps = round(self._frames_written / max(0.001, duration), 2)

        metadata = {
            "recorded_at": datetime.now().isoformat(),
            "filepath": self._filepath,
            "duration_seconds": duration,
            "frames_written": self._frames_written,
            "frames_dropped": self._frames_dropped,
            "actual_fps": avg_fps,
            "configured_fps": self._video_fps,
            "resolution": f"{self._resolution[0]}x{self._resolution[1]}",
        }
        metadata.update(self._extra_metadata)

        # Write metadata JSON
        try:
            with open(self._meta_filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Recorded metadata saved to: '{self._meta_filepath}'")
        except Exception as e:
            logger.warning(f"Could not save video metadata: {e}")

        logger.info(
            f"Recording finalized: '{self._filepath}' ({duration}s, {self._frames_written} frames written, "
            f"{self._frames_dropped} dropped, avg {avg_fps} FPS)"
        )
        return metadata

    def _writer_worker(self) -> None:
        """Worker loop reading frames from queue and writing to disk."""
        while not self._stop_event.is_set() or not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get(timeout=0.1)
                if self._writer is not None and frame is not None:
                    # Resize if frame dimensions don't match writer target
                    h, w = frame.shape[:2]
                    if (w, h) != self._resolution:
                        frame = cv2.resize(frame, self._resolution)
                    self._writer.write(frame)
                    self._frames_written += 1
                self._frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error writing video frame: {e}")

        # Final release
        if self._writer is not None:
            self._writer.release()
            self._writer = None
