"""
Configuration Loader Compatibility Wrapper.
"""

from src.core.config_loader import (
    AppConfig,
    ModelsConfig,
    TargetsConfig,
    CameraConfig,
    PerformanceConfig,
    ProcessingConfig,
    VirtualCameraConfig,
    RuntimeConfig,
    SingleModelConfig,
    load_all_configs,
    load_app_config,
    load_models_config,
    load_targets_config,
)

__all__ = [
    "AppConfig",
    "ModelsConfig",
    "TargetsConfig",
    "CameraConfig",
    "PerformanceConfig",
    "ProcessingConfig",
    "VirtualCameraConfig",
    "RuntimeConfig",
    "SingleModelConfig",
    "load_all_configs",
    "load_app_config",
    "load_models_config",
    "load_targets_config",
]
