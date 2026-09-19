"""
Adaptive Performance Governor and Stage-by-Stage Micro-Profiling Engine.
Monitors execution bottlenecks and dynamically adjusts pipeline load to sustain target 30+ FPS.
"""

import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("PerformanceGovernor")


class PipelineProfiler:
    """
    Context manager and tracker for stage-by-stage pipeline micro-benchmarking.
    Computes rolling averages, min, and max timings for all visual transformation stages.
    """

    def __init__(self, ema_alpha: float = 0.20):
        self.ema_alpha = float(ema_alpha)
        self._timings: Dict[str, float] = {}
        self._averages: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._min_times: Dict[str, float] = {}
        self._max_times: Dict[str, float] = {}

    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager to measure and log duration of a pipeline phase in milliseconds."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            self._record_stage(stage_name, dur_ms)

    def _record_stage(self, stage_name: str, duration_ms: float) -> None:
        self._timings[stage_name] = duration_ms
        prev_avg = self._averages.get(stage_name)
        if prev_avg is None:
            self._averages[stage_name] = duration_ms
            self._min_times[stage_name] = duration_ms
            self._max_times[stage_name] = duration_ms
            self._counts[stage_name] = 1
        else:
            self._averages[stage_name] = (self.ema_alpha * duration_ms) + ((1.0 - self.ema_alpha) * prev_avg)
            self._min_times[stage_name] = min(self._min_times[stage_name], duration_ms)
            self._max_times[stage_name] = max(self._max_times[stage_name], duration_ms)
            self._counts[stage_name] += 1

    def get_stage_timings(self) -> Dict[str, float]:
        """Returns the most recent execution time for each stage."""
        return self._timings.copy()

    def get_stage_averages(self) -> Dict[str, float]:
        """Returns smoothed average timings for each stage."""
        return {k: round(v, 2) for k, v in self._averages.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Returns full micro-profiling telemetry."""
        return {
            "recent_ms": {k: round(v, 2) for k, v in self._timings.items()},
            "average_ms": {k: round(v, 2) for k, v in self._averages.items()},
            "min_ms": {k: round(v, 2) for k, v in self._min_times.items()},
            "max_ms": {k: round(v, 2) for k, v in self._max_times.items()},
            "counts": self._counts.copy(),
        }

    def reset(self) -> None:
        """Clears all timing statistics."""
        self._timings.clear()
        self._averages.clear()
        self._counts.clear()
        self._min_times.clear()
        self._max_times.clear()


class AdaptivePerformanceGovernor:
    """
    Intelligent runtime load regulator.
    Monitors rolling FPS, system latency, and CPU/GPU pressure.
    Dynamically modulates detection interval, enhancement fidelity, and blending passes
    with hysteresis to ensure smooth 30+ FPS without jarring visual shifts.
    """

    STATE_OPTIMAL = "optimal"       # >= 27 FPS: Maximum neural quality and super-resolution
    STATE_BALANCED = "balanced"     # 20 - 27 FPS: Standard quality, medium enhancement
    STATE_THROTTLED = "throttled"   # < 20 FPS: High speed, zero-latency bilateral enhancement

    def __init__(
        self,
        target_fps: float = 30.0,
        low_fps_threshold: float = 22.0,
        high_fps_threshold: float = 28.0,
        hysteresis_frames: int = 45,
    ):
        """
        Args:
            target_fps: Desired interactive frame rate.
            low_fps_threshold: Threshold below which the governor steps down quality.
            high_fps_threshold: Threshold above which the governor steps up quality.
            hysteresis_frames: Minimum frame count required before transitioning states.
        """
        self.target_fps = float(target_fps)
        self.low_fps_threshold = float(low_fps_threshold)
        self.high_fps_threshold = float(high_fps_threshold)
        self.hysteresis_frames = int(hysteresis_frames)

        self.current_state = self.STATE_OPTIMAL
        self._frames_in_current_state = 0
        self.auto_tune_enabled = True

    def update(self, current_fps: float) -> str:
        """
        Updates governor state based on real-time rolling FPS.
        Returns the active governor state ("optimal", "balanced", "throttled").
        """
        if not self.auto_tune_enabled or current_fps <= 0.0:
            return self.current_state

        self._frames_in_current_state += 1

        # Check step down
        if current_fps < self.low_fps_threshold:
            if self._frames_in_current_state >= self.hysteresis_frames:
                if self.current_state == self.STATE_OPTIMAL:
                    self.current_state = self.STATE_BALANCED
                    self._frames_in_current_state = 0
                    logger.info(f"PerformanceGovernor stepped down to {self.current_state} (FPS: {current_fps:.1f})")
                elif self.current_state == self.STATE_BALANCED:
                    self.current_state = self.STATE_THROTTLED
                    self._frames_in_current_state = 0
                    logger.info(f"PerformanceGovernor stepped down to {self.current_state} (FPS: {current_fps:.1f})")

        # Check step up
        elif current_fps > self.high_fps_threshold:
            if self._frames_in_current_state >= self.hysteresis_frames:
                if self.current_state == self.STATE_THROTTLED:
                    self.current_state = self.STATE_BALANCED
                    self._frames_in_current_state = 0
                    logger.info(f"PerformanceGovernor recovered to {self.current_state} (FPS: {current_fps:.1f})")
                elif self.current_state == self.STATE_BALANCED:
                    self.current_state = self.STATE_OPTIMAL
                    self._frames_in_current_state = 0
                    logger.info(f"PerformanceGovernor recovered to {self.current_state} (FPS: {current_fps:.1f})")

        return self.current_state

    def get_recommended_detection_interval(self, base_interval: int = 3) -> int:
        """Calculates optimal face detection interval (skip frames between re-detection)."""
        if self.current_state == self.STATE_OPTIMAL:
            return max(1, base_interval)
        elif self.current_state == self.STATE_BALANCED:
            return max(1, base_interval + 2)
        else:  # throttled
            return max(1, base_interval + 5)

    def get_recommended_enhancement_mode(self, configured_mode: str = "onnx") -> str:
        """Determines whether neural super-resolution or bilateral enhancement should run."""
        if self.current_state == self.STATE_OPTIMAL:
            return configured_mode
        elif self.current_state == self.STATE_BALANCED:
            # Drop heavy ONNX super-resolution to adaptive bilateral texture if on CPU
            return "adaptive" if configured_mode == "onnx" else configured_mode
        else:  # throttled
            return "off" if configured_mode == "off" else "adaptive"

    def get_status_badge(self) -> Dict[str, Any]:
        """Returns structured state description for UI status display."""
        color_map = {
            self.STATE_OPTIMAL: "#4CAF50",   # Green
            self.STATE_BALANCED: "#FF9800",  # Orange
            self.STATE_THROTTLED: "#F44336", # Red
        }
        return {
            "state": self.current_state,
            "badge_text": self.current_state.upper(),
            "color": color_map.get(self.current_state, "#B0BEC5"),
            "auto_tune": self.auto_tune_enabled,
        }
