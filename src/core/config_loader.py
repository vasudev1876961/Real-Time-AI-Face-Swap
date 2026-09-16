"""
Configuration Loader and Typed Dataclass Definitions.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import yaml
from src.utils.logger import get_logger

logger = get_logger("ConfigLoader")


@dataclass
class CameraConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "AUTO"
    auto_reconnect: bool = True
    reconnect_interval_sec: float = 2.0


@dataclass
class PerformanceConfig:
    mode: str = "quality"  # "performance", "quality", "debug"
    target_fps: int = 30
    detection_interval: int = 3
    max_faces: int = 1
    enable_smoothing: bool = True
    ema_alpha: float = 0.65


@dataclass
class ProcessingConfig:
    mask_type: str = "smooth_hull"  # "smooth_hull", "pose_adaptive", "distance_transform", "convex_hull", "elliptical"
    mask_blur: int = 15
    mask_feather: float = 0.6
    mask_erosion: int = 1
    use_roi_blending: bool = True
    color_correction: str = "reinhard"
    color_blend_ratio: float = 0.70
    temporal_color_smoothing: bool = True
    color_smoothing_alpha: float = 0.70
    blending_method: str = "alpha"
    blending_strength: float = 1.0
    seamless_clone_mode: str = "NORMAL_CLONE"
    postprocess_sharpen: float = 0.30
    enhancement_strength: float = 0.50
    enhancement_mode: str = "onnx"  # "adaptive", "onnx", "off"
    texture_detail_transfer: float = 0.35  # High-frequency skin pore & micro-texture transfer [0.0, 1.0]
    illumination_matching: bool = True  # Retinex directional illumination adaptation



@dataclass
class RuntimeConfig:
    preferred_device: str = "auto"
    thread_count: int = 4
    allow_cpu_fallback: bool = True


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    storage: Dict[str, str] = field(
        default_factory=lambda: {
            "captures_dir": "outputs/captures",
            "recordings_dir": "outputs/recordings",
            "benchmarks_dir": "outputs/benchmarks",
        }
    )


@dataclass
class SingleModelConfig:
    model_path: str = ""
    model_type: str = "onnx"
    provider: str = "auto"
    enabled: bool = True
    input_size: List[int] = field(default_factory=lambda: [128, 128])
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelsConfig:
    face_analysis: SingleModelConfig = field(
        default_factory=lambda: SingleModelConfig(
            model_path="models/face_analysis/face_recognition.onnx",
            model_type="arcface_onnx",
            input_size=[112, 112],
        )
    )
    face_detection: SingleModelConfig = field(
        default_factory=lambda: SingleModelConfig(
            model_path="models/face_detection/scrfd_500m.onnx",
            model_type="scrfd_onnx",
            input_size=[640, 640],
        )
    )
    face_swap: SingleModelConfig = field(
        default_factory=lambda: SingleModelConfig(
            model_path="models/face_swap/model.onnx",
            model_type="inswapper_onnx",
            input_size=[128, 128],
        )
    )
    enhancement: SingleModelConfig = field(
        default_factory=lambda: SingleModelConfig(
            model_path="models/enhancement/face_enhancer.onnx",
            model_type="gfpgan_onnx",
            enabled=False,
        )
    )


@dataclass
class CategoryDef:
    id: str
    display_name: str


@dataclass
class TargetsConfig:
    target_root: str = "faces"
    categories: List[CategoryDef] = field(
        default_factory=lambda: [
            CategoryDef(id="all", display_name="All Categories"),
            CategoryDef(id="actresses", display_name="Actresses"),
            CategoryDef(id="actors", display_name="Actors"),
            CategoryDef(id="telugu_heroes", display_name="Telugu Heroes"),
        ]
    )
    require_reference_image: bool = True
    reference_filename: str = "reference.jpg"
    metadata_filename: str = "metadata.json"
    cache_embedding_filename: str = "face.npy"
    default_category: str = "all"
    auto_reload_on_change: bool = True


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        logger.warning(f"Config file not found: {path}, using defaults.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error parsing YAML {path}: {e}")
        return {}


def load_app_config(path: str = "configs/config.yaml") -> AppConfig:
    """Loads application configuration from YAML file."""
    data = _load_yaml(path)
    camera_data = data.get("camera", {})
    perf_data = data.get("performance", {})
    proc_data = data.get("processing", {})
    runtime_data = data.get("runtime", {})
    storage_data = data.get("storage", {})

    return AppConfig(
        camera=CameraConfig(**camera_data) if camera_data else CameraConfig(),
        performance=PerformanceConfig(**perf_data) if perf_data else PerformanceConfig(),
        processing=ProcessingConfig(**proc_data) if proc_data else ProcessingConfig(),
        runtime=RuntimeConfig(**runtime_data) if runtime_data else RuntimeConfig(),
        storage=storage_data if storage_data else {
            "captures_dir": "outputs/captures",
            "recordings_dir": "outputs/recordings",
            "benchmarks_dir": "outputs/benchmarks",
        },
    )


def load_models_config(path: str = "configs/models.yaml") -> ModelsConfig:
    """Loads models configuration from YAML file."""
    data = _load_yaml(path)
    models_cfg = ModelsConfig()

    for key in ["face_analysis", "face_detection", "face_swap", "enhancement"]:
        if key in data and isinstance(data[key], dict):
            entry = data[key]
            std_fields = {"model_path", "model_type", "provider", "enabled", "input_size"}
            extra = {k: v for k, v in entry.items() if k not in std_fields}
            cfg = SingleModelConfig(
                model_path=entry.get("model_path", getattr(models_cfg, key).model_path),
                model_type=entry.get("model_type", getattr(models_cfg, key).model_type),
                provider=entry.get("provider", "auto"),
                enabled=entry.get("enabled", getattr(models_cfg, key).enabled),
                input_size=entry.get("input_size", getattr(models_cfg, key).input_size),
                extra=extra,
            )
            setattr(models_cfg, key, cfg)
    return models_cfg


def load_targets_config(path: str = "configs/targets.yaml") -> TargetsConfig:
    """Loads target database configuration from YAML file."""
    data = _load_yaml(path)
    cfg = TargetsConfig()
    if "target_root" in data:
        cfg.target_root = data["target_root"]
    if "categories" in data and isinstance(data["categories"], list):
        cfg.categories = [
            CategoryDef(id=c.get("id", ""), display_name=c.get("display_name", ""))
            for c in data["categories"]
            if isinstance(c, dict)
        ]
    if "require_reference_image" in data:
        cfg.require_reference_image = data["require_reference_image"]
    if "reference_filename" in data:
        cfg.reference_filename = data["reference_filename"]
    if "metadata_filename" in data:
        cfg.metadata_filename = data["metadata_filename"]
    if "cache_embedding_filename" in data:
        cfg.cache_embedding_filename = data["cache_embedding_filename"]
    if "default_category" in data:
        cfg.default_category = data["default_category"]
    return cfg


def load_all_configs(base_dir: str = "configs") -> Tuple[AppConfig, ModelsConfig, TargetsConfig]:
    """Helper to load all configurations in one call."""
    app_cfg = load_app_config(os.path.join(base_dir, "config.yaml"))
    models_cfg = load_models_config(os.path.join(base_dir, "models.yaml"))
    targets_cfg = load_targets_config(os.path.join(base_dir, "targets.yaml"))
    return app_cfg, models_cfg, targets_cfg
