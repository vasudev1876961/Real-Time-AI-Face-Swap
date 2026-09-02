"""
High-Resolution Performance Monitoring, Latency Breakdown, and Resource Tracking.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import psutil

from src.core.device import get_device_manager
from src.utils.logger import get_logger

logger = get_logger("PerformanceMetrics")


@dataclass
class PipelineTimings:
    """Breakdown of individual pipeline stage latencies in milliseconds."""
    capture_ms: float = 0.0
    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    alignment_ms: float = 0.0
    swap_ms: float = 0.0
    mask_ms: float = 0.0
    color_ms: float = 0.0
    blend_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0


class PerformanceMetrics:
    """
    Tracks rolling FPS, component-wise latency statistics, and hardware usage.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._frame_timestamps = deque(maxlen=window_size)
        self._total_latencies = deque(maxlen=window_size)
        self._swap_latencies = deque(maxlen=window_size)
        self._last_timings = PipelineTimings()
        self._total_processed_frames = 0
        self._fps: float = 0.0
        self._start_time = time.time()

    def record_frame(self, timings: PipelineTimings) -> None:
        """Records a new frame completion and updates rolling metrics."""
        now = time.time()
        self._frame_timestamps.append(now)
        self._total_latencies.append(timings.total_ms)
        self._swap_latencies.append(timings.swap_ms)
        self._last_timings = timings
        self._total_processed_frames += 1

        # Compute rolling FPS
        if len(self._frame_timestamps) >= 2:
            time_diff = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if time_diff > 0:
                self._fps = (len(self._frame_timestamps) - 1) / time_diff

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    @property
    def average_total_latency_ms(self) -> float:
        if not self._total_latencies:
            return 0.0
        return round(sum(self._total_latencies) / len(self._total_latencies), 1)

    @property
    def average_swap_latency_ms(self) -> float:
        if not self._swap_latencies:
            return 0.0
        return round(sum(self._swap_latencies) / len(self._swap_latencies), 1)

    @property
    def current_timings(self) -> PipelineTimings:
        return self._last_timings

    def get_summary(
        self,
        face_detected: bool = False,
        ai_active: bool = False,
        model_ready: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Returns structured dictionary for UI panels and loggers."""
        dm = get_device_manager()
        sys_stats = dm.get_system_stats()

        active = ai_active or kwargs.get("is_swapped", False)
        ready = model_ready or kwargs.get("model_loaded", False)

        if not ready:
            ai_status = "MODEL NOT FOUND"
        elif active:
            ai_status = "RUNNING"
        else:
            ai_status = "BYPASS"

        return {
            "fps": self.fps,
            "total_latency_ms": self.average_total_latency_ms,
            "swap_latency_ms": self.average_swap_latency_ms,
            "face_detected": "YES" if face_detected else "NO",
            "ai_status": ai_status,
            "gpu_active": "YES" if dm.is_gpu_available else "NO (CPU)",
            "provider": dm.active_provider.replace("ExecutionProvider", ""),
            "cpu_percent": sys_stats["cpu_percent"],
            "ram_percent": sys_stats["ram_percent"],
            "total_frames": self._total_processed_frames,
            "timings": {
                "detect": round(self._last_timings.detection_ms, 1),
                "track": round(self._last_timings.tracking_ms, 1),
                "align": round(self._last_timings.alignment_ms, 1),
                "swap": round(self._last_timings.swap_ms, 1),
                "blend": round(self._last_timings.blend_ms, 1),
                "total": round(self._last_timings.total_ms, 1),
            },
        }
