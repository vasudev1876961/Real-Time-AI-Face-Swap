"""
Model Manager for Model Lifecycle Coordination and Compatibility Verification.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

from src.models.base_swapper import BaseFaceSwapper
from src.models.onnx_swapper import ONNXSwapper
from src.detection.face_detector import FaceDetector, FaceData, get_face_detector
from src.targets.target_embedding import TargetEmbeddingExtractor
from src.targets.target_loader import TargetFace
from src.core.config_loader import ModelsConfig, SingleModelConfig
from src.core.device import get_device_manager
from src.utils.logger import get_logger

logger = get_logger("ModelManager")


class ModelManager:
    """
    Central coordinator for AI inference components.
    Ensures that the rest of the application interacts with models through
    clean, replaceable adapters.
    """

    def __init__(self, config: Optional[ModelsConfig] = None):
        self.config = config or ModelsConfig()
        self.detector: FaceDetector = get_face_detector(self.config.face_detection)
        self.embedding_extractor: TargetEmbeddingExtractor = TargetEmbeddingExtractor(self.config.face_analysis)
        self.swapper: BaseFaceSwapper = ONNXSwapper(self.config.face_swap)

    def is_swap_ready(self) -> bool:
        """Returns True if the swap model is loaded and ready for live transformation."""
        return self.swapper.is_loaded()

    def get_status_summary(self) -> Dict[str, Any]:
        """Provides status breakdown for UI and diagnostics."""
        dm = get_device_manager()
        swap_info = self.swapper.get_model_info()

        return {
            "detector_ready": self.detector.is_ready(),
            "detector_backend": self.detector.backend_type,
            "swapper_ready": self.swapper.is_loaded(),
            "swapper_provider": swap_info.get("provider", "None"),
            "swapper_latency_ms": self.swapper.get_last_latency_ms(),
            "device_active_provider": dm.active_provider,
            "is_gpu_available": dm.is_gpu_available,
        }

    def swap(
        self,
        aligned_face: np.ndarray,
        source_face: FaceData,
        target_face: TargetFace,
    ) -> Tuple[np.ndarray, float]:
        """
        Executes face swap and returns (swapped_crop, inference_latency_ms).
        """
        if not self.swapper.is_loaded():
            return aligned_face, 0.0

        swapped_crop = self.swapper.swap(aligned_face, source_face, target_face)
        latency_ms = self.swapper.get_last_latency_ms()
        return swapped_crop, latency_ms

    def reload_swapper(self, model_path: Optional[str] = None, provider: str = "auto") -> bool:
        """Reloads the face swap model from a new path or provider."""
        path = model_path or self.config.face_swap.model_path
        return self.swapper.load(path, provider)


_GLOBAL_MODEL_MANAGER: Optional[ModelManager] = None


def get_model_manager(config: Optional[ModelsConfig] = None) -> ModelManager:
    """Returns singleton ModelManager instance."""
    global _GLOBAL_MODEL_MANAGER
    if _GLOBAL_MODEL_MANAGER is None:
        _GLOBAL_MODEL_MANAGER = ModelManager(config)
    return _GLOBAL_MODEL_MANAGER
