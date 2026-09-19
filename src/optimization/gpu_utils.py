"""
Hardware Accelerator Profiling and GPU Resource Management.
Monitors VRAM usage, manages ONNX Runtime execution provider configs, and handles memory cleanup.
"""

import os
import gc
from typing import Dict, Any, Optional, List
from src.utils.logger import get_logger

logger = get_logger("GPUUtils")


class GPUManager:
    """
    Hardware and VRAM resource manager.
    Provides device discovery, memory monitoring, and runtime execution optimizations
    for CUDA, DirectML, and CPU execution providers.
    """

    _instance: Optional["GPUManager"] = None

    def __new__(cls) -> "GPUManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._has_cuda = False
        self._has_dml = False
        self._device_name = "CPU"
        self._detect_hardware()

    def _detect_hardware(self) -> None:
        """Detects available GPU accelerators via ONNX Runtime and PyTorch if installed."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            self._has_cuda = "CUDAExecutionProvider" in available
            self._has_dml = "DmlExecutionProvider" in available

            if self._has_cuda:
                self._device_name = "NVIDIA CUDA"
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._device_name = torch.cuda.get_device_name(0)
                except ImportError:
                    pass
            elif self._has_dml:
                self._device_name = "DirectML (GPU)"
            else:
                self._device_name = "CPU (x86_64)"
        except Exception as e:
            logger.debug(f"Hardware detection note: {e}")
            self._device_name = "CPU"

    @property
    def has_gpu(self) -> bool:
        """Returns True if a hardware GPU provider is available."""
        return self._has_cuda or self._has_dml

    @property
    def is_cuda(self) -> bool:
        """Returns True if NVIDIA CUDA acceleration is active."""
        return self._has_cuda

    @property
    def device_name(self) -> str:
        """Returns human-readable device name."""
        return self._device_name

    def get_vram_info(self) -> Dict[str, Any]:
        """
        Queries current GPU VRAM utilization.
        Returns dictionary with total_mb, used_mb, free_mb, and usage_percent.
        """
        if not self._has_cuda:
            return {
                "available": False,
                "device": self._device_name,
                "total_mb": 0.0,
                "used_mb": 0.0,
                "free_mb": 0.0,
                "usage_percent": 0.0,
            }

        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                used_bytes = total_bytes - free_bytes
                total_mb = total_bytes / (1024.0 * 1024.0)
                used_mb = used_bytes / (1024.0 * 1024.0)
                free_mb = free_bytes / (1024.0 * 1024.0)
                pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
                return {
                    "available": True,
                    "device": self._device_name,
                    "total_mb": round(total_mb, 1),
                    "used_mb": round(used_mb, 1),
                    "free_mb": round(free_mb, 1),
                    "usage_percent": round(pct, 1),
                }
        except Exception as e:
            logger.debug(f"Could not read CUDA VRAM info: {e}")

        return {
            "available": True,
            "device": self._device_name,
            "total_mb": 0.0,
            "used_mb": 0.0,
            "free_mb": 0.0,
            "usage_percent": 0.0,
        }

    def clear_gpu_cache(self) -> None:
        """Flushes GPU VRAM allocations and triggers Python garbage collection."""
        gc.collect()
        if self._has_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass

    def get_recommended_thread_count(self) -> int:
        """Returns optimal thread count for ONNX Runtime CPU execution."""
        cpu_count = os.cpu_count() or 4
        return min(8, max(2, cpu_count - 1))

    def create_optimized_session_options(
        self,
        enable_graph_optimization: bool = True,
        thread_count: Optional[int] = None,
    ) -> Any:
        """
        Creates a pre-configured ONNX Runtime SessionOptions object with thread pools
        and memory arena optimizations enabled.
        """
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            threads = thread_count if thread_count is not None else self.get_recommended_thread_count()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 2
            opts.enable_cpu_mem_arena = True
            opts.enable_mem_pattern = True
            if enable_graph_optimization:
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return opts
        except Exception as e:
            logger.error(f"Failed to create SessionOptions: {e}")
            return None


def get_gpu_manager() -> GPUManager:
    """Returns singleton GPUManager instance."""
    return GPUManager()
