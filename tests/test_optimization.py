"""
Unit Tests for Optimization Subsystem: FPS Monitoring, GPU Utilities, and Performance Governor.
"""

import time
import pytest
import numpy as np

from src.optimization.fps_monitor import FPSMonitor
from src.optimization.gpu_utils import GPUManager, get_gpu_manager
from src.optimization.performance import PipelineProfiler, AdaptivePerformanceGovernor


def test_fps_monitor_basic_and_percentiles():
    monitor = FPSMonitor(window_seconds=1.0)
    assert monitor.get_fps() == 0.0

    # Simulate 10 frames at ~30 FPS (~33.3ms interval)
    for _ in range(10):
        time.sleep(0.01)  # small delta for test speed
        monitor.tick()

    fps = monitor.get_fps()
    assert fps > 0.0

    percentiles = monitor.get_latency_percentiles()
    assert "p50" in percentiles
    assert "p95" in percentiles
    assert percentiles["p50"] > 0.0

    stats = monitor.get_stats(target_fps=30.0)
    assert "fps" in stats
    assert "jitter_ms" in stats
    assert "latency_p95_ms" in stats
    assert stats["total_frames"] == 10

    monitor.reset()
    assert monitor.get_fps() == 0.0


def test_gpu_manager():
    gpu_mgr = get_gpu_manager()
    assert gpu_mgr is not None
    assert isinstance(gpu_mgr.device_name, str)
    assert isinstance(gpu_mgr.has_gpu, bool)

    vram = gpu_mgr.get_vram_info()
    assert "available" in vram
    assert "device" in vram

    gpu_mgr.clear_gpu_cache()

    opts = gpu_mgr.create_optimized_session_options()
    assert opts is not None


def test_adaptive_performance_governor_transitions():
    governor = AdaptivePerformanceGovernor(
        target_fps=30.0,
        low_fps_threshold=20.0,
        high_fps_threshold=28.0,
        hysteresis_frames=3,
    )
    assert governor.current_state == AdaptivePerformanceGovernor.STATE_OPTIMAL
    assert governor.get_recommended_detection_interval(3) == 3

    # Feed low FPS (15 FPS) for 3 frames to trigger step down to balanced
    for _ in range(3):
        governor.update(15.0)
    assert governor.current_state == AdaptivePerformanceGovernor.STATE_BALANCED
    assert governor.get_recommended_detection_interval(3) > 3

    # Feed low FPS again to trigger step down to throttled
    for _ in range(3):
        governor.update(15.0)
    assert governor.current_state == AdaptivePerformanceGovernor.STATE_THROTTLED

    # Feed high FPS (32 FPS) to trigger recovery to balanced
    for _ in range(3):
        governor.update(32.0)
    assert governor.current_state == AdaptivePerformanceGovernor.STATE_BALANCED

    # Feed high FPS to recover to optimal
    for _ in range(3):
        governor.update(32.0)
    assert governor.current_state == AdaptivePerformanceGovernor.STATE_OPTIMAL

    badge = governor.get_status_badge()
    assert badge["badge_text"] == "OPTIMAL"


def test_pipeline_profiler():
    profiler = PipelineProfiler()

    with profiler.profile_stage("test_stage"):
        time.sleep(0.005)

    timings = profiler.get_stage_timings()
    assert "test_stage" in timings
    assert timings["test_stage"] > 0.0

    summary = profiler.get_summary()
    assert "average_ms" in summary
    assert "test_stage" in summary["average_ms"]
