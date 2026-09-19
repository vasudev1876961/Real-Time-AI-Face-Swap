"""
High-Precision Real-Time FPS and Frame Latency Telemetry Monitor.
Tracks moving average FPS, percentile latencies (p50, p95, p99), jitter, and frame drops.
"""

import time
from collections import deque
from typing import Dict, Any, Optional
import numpy as np


class FPSMonitor:
    """
    Rolling-window frame rate and timing distribution profiler.
    Computes instant FPS, windowed mean FPS, jitter (standard deviation of frame deltas),
    and percentile latencies without memory allocation overhead during frame processing.
    """

    def __init__(self, window_seconds: float = 1.5, max_samples: int = 120):
        """
        Args:
            window_seconds: Duration in seconds to retain samples for rolling calculations.
            max_samples: Maximum number of recent frame deltas to keep in rolling buffer.
        """
        self.window_seconds = float(window_seconds)
        self.max_samples = int(max_samples)
        self._frame_times = deque(maxlen=self.max_samples)
        self._frame_deltas = deque(maxlen=self.max_samples)
        self._last_tick_time: Optional[float] = None
        self._total_frame_count: int = 0
        self._start_time: float = time.perf_counter()

    def reset(self) -> None:
        """Resets all metrics and timestamps."""
        self._frame_times.clear()
        self._frame_deltas.clear()
        self._last_tick_time = None
        self._total_frame_count = 0
        self._start_time = time.perf_counter()

    def tick(self) -> float:
        """
        Records the completion of one frame and returns the rolling FPS.
        """
        now = time.perf_counter()
        self._total_frame_count += 1

        if self._last_tick_time is not None:
            delta = now - self._last_tick_time
            if delta > 0.0:
                self._frame_times.append(now)
                self._frame_deltas.append(delta)

        self._last_tick_time = now

        # Prune entries older than window_seconds
        cutoff = now - self.window_seconds
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()
            if self._frame_deltas:
                self._frame_deltas.popleft()

        return self.get_fps()

    def get_fps(self) -> float:
        """Calculates current rolling average frames per second."""
        if not self._frame_deltas:
            return 0.0
        avg_delta = sum(self._frame_deltas) / len(self._frame_deltas)
        return float(1.0 / avg_delta) if avg_delta > 0 else 0.0

    def get_instantaneous_fps(self) -> float:
        """Returns the frame rate corresponding strictly to the most recent frame delta."""
        if not self._frame_deltas:
            return 0.0
        last_delta = self._frame_deltas[-1]
        return float(1.0 / last_delta) if last_delta > 0 else 0.0

    def get_latency_percentiles(self) -> Dict[str, float]:
        """
        Computes percentile frame durations in milliseconds (p50, p90, p95, p99).
        """
        if not self._frame_deltas:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

        deltas_ms = np.array(self._frame_deltas, dtype=np.float32) * 1000.0
        return {
            "p50": float(np.percentile(deltas_ms, 50)),
            "p90": float(np.percentile(deltas_ms, 90)),
            "p95": float(np.percentile(deltas_ms, 95)),
            "p99": float(np.percentile(deltas_ms, 99)),
        }

    def get_frame_jitter_ms(self) -> float:
        """
        Computes the standard deviation of frame times in milliseconds (frame-time jitter).
        """
        if len(self._frame_deltas) < 2:
            return 0.0
        deltas_ms = np.array(self._frame_deltas, dtype=np.float32) * 1000.0
        return float(np.std(deltas_ms))

    def get_drop_rate(self, target_fps: float = 30.0) -> float:
        """
        Estimates the ratio of missed/dropped frames relative to the target frame rate [0.0, 1.0].
        """
        if target_fps <= 0.0 or not self._frame_deltas:
            return 0.0
        actual_fps = self.get_fps()
        if actual_fps >= target_fps:
            return 0.0
        return float(np.clip((target_fps - actual_fps) / target_fps, 0.0, 1.0))

    def get_stats(self, target_fps: float = 30.0) -> Dict[str, Any]:
        """
        Returns a consolidated dictionary of all performance metrics.
        """
        percentiles = self.get_latency_percentiles()
        return {
            "fps": round(self.get_fps(), 1),
            "instant_fps": round(self.get_instantaneous_fps(), 1),
            "jitter_ms": round(self.get_frame_jitter_ms(), 2),
            "latency_p50_ms": round(percentiles["p50"], 1),
            "latency_p95_ms": round(percentiles["p95"], 1),
            "latency_p99_ms": round(percentiles["p99"], 1),
            "drop_rate": round(self.get_drop_rate(target_fps), 3),
            "total_frames": self._total_frame_count,
            "elapsed_sec": round(time.perf_counter() - self._start_time, 1),
        }
