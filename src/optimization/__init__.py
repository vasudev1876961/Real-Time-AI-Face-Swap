"""
Optimization Subsystem: High-Resolution FPS Telemetry, GPU/VRAM Management,
Adaptive Performance Governing, and Stage-by-Stage Micro-Profiling.
"""

from src.optimization.fps_monitor import FPSMonitor
from src.optimization.gpu_utils import GPUManager, get_gpu_manager
from src.optimization.performance import PipelineProfiler, AdaptivePerformanceGovernor
from src.optimization.buffer_pool import FrameBufferPool, get_buffer_pool
from src.optimization.turbo_pipeline import TurboSpatialOptimizer

__all__ = [
    "FPSMonitor",
    "GPUManager",
    "get_gpu_manager",
    "PipelineProfiler",
    "AdaptivePerformanceGovernor",
    "FrameBufferPool",
    "get_buffer_pool",
    "TurboSpatialOptimizer",
]

