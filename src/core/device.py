"""
Device Detection and ONNX Runtime Execution Provider Management.

Detects available hardware acceleration (NVIDIA CUDA, DirectML, CPU) safely
at runtime without modifying system packages.
"""

import os
import sys
import psutil
from typing import List, Tuple, Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("DeviceManager")


def get_available_onnx_providers() -> List[str]:
    """
    Returns the list of execution providers supported by the currently installed
    onnxruntime package.
    """
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception as e:
        logger.warning(f"Failed to query onnxruntime providers: {e}")
        return ["CPUExecutionProvider"]


def get_execution_providers(preferred: str = "auto") -> Tuple[List[str], str]:
    """
    Selects the best available ONNX Runtime execution provider list based on
    system availability and user preference.

    Returns:
        (provider_list, active_provider_name)
    """
    available = get_available_onnx_providers()
    preferred_lower = str(preferred).strip().lower()

    if preferred_lower in ["cuda", "cudaexecutionprovider"] and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"

    if preferred_lower in ["dml", "dmlexecutionprovider"] and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"], "DmlExecutionProvider"

    if preferred_lower in ["tensorrt", "tensorrtexecutionprovider"] and "TensorrtExecutionProvider" in available:
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"], "TensorrtExecutionProvider"

    if preferred_lower in ["cpu", "cpuexecutionprovider"]:
        return ["CPUExecutionProvider"], "CPUExecutionProvider"

    # Automatic selection based on priority
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"
    elif "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"], "DmlExecutionProvider"
    elif "TensorrtExecutionProvider" in available:
        return ["TensorrtExecutionProvider", "CPUExecutionProvider"], "TensorrtExecutionProvider"

    return ["CPUExecutionProvider"], "CPUExecutionProvider"


class DeviceManager:
    """Manages runtime compute devices and provides diagnostic information."""

    def __init__(self, preferred_device: str = "auto"):
        self.preferred_device = preferred_device
        self.available_providers = get_available_onnx_providers()
        self.providers, self.active_provider = get_execution_providers(preferred_device)
        self.is_gpu_available = self.active_provider != "CPUExecutionProvider"
        self._log_device_info()

    def _log_device_info(self) -> None:
        logger.info(f"Available ONNX Providers: {self.available_providers}")
        logger.info(f"Selected Active Provider: {self.active_provider}")
        logger.info(f"GPU Acceleration: {'ENABLED' if self.is_gpu_available else 'DISABLED (CPU Fallback)'}")

    def get_providers_for_model(self, model_preferred: Optional[str] = None) -> List[str]:
        """Returns the provider list for a specific model."""
        pref = model_preferred if model_preferred and model_preferred != "auto" else self.preferred_device
        providers, _ = get_execution_providers(pref)
        return providers

    def get_system_stats(self) -> Dict[str, Any]:
        """Returns snapshot of current system memory and CPU utilization."""
        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        
        gpu_name = "N/A"
        if self.is_gpu_available:
            gpu_name = self.active_provider.replace("ExecutionProvider", "")

        return {
            "active_provider": self.active_provider,
            "is_gpu_available": self.is_gpu_available,
            "gpu_name": gpu_name,
            "cpu_percent": cpu_pct,
            "ram_percent": vm.percent,
            "ram_used_gb": round(vm.used / (1024 ** 3), 2),
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "cpu_cores": psutil.cpu_count(logical=True),
        }


_GLOBAL_DEVICE_MANAGER: Optional[DeviceManager] = None


def get_device_manager(preferred_device: str = "auto") -> DeviceManager:
    """Returns the singleton DeviceManager instance."""
    global _GLOBAL_DEVICE_MANAGER
    if _GLOBAL_DEVICE_MANAGER is None:
        _GLOBAL_DEVICE_MANAGER = DeviceManager(preferred_device)
    return _GLOBAL_DEVICE_MANAGER
