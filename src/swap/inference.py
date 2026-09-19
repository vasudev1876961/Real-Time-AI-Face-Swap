"""
Neural Face Swap Inference Coordinator and Warmup Utilities.
"""

from typing import Optional, Tuple
import numpy as np
from src.detection.face_detector import FaceData
from src.targets.target_loader import TargetFace
from src.models.model_manager import get_model_manager, ModelManager


class SwapInferenceEngine:
    """
    High-level facade for executing neural face swaps and warmup passes.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.manager = model_manager or get_model_manager()

    def is_ready(self) -> bool:
        """Returns True if the underlying swapper model is loaded and ready."""
        return self.manager.is_swap_ready()

    def swap_face(
        self,
        aligned_crop: np.ndarray,
        source_face: FaceData,
        target_face: TargetFace,
    ) -> Tuple[np.ndarray, float]:
        """
        Executes face swap on aligned 128x128 crop.
        Returns (swapped_crop, latency_ms).
        """
        return self.manager.swap(aligned_crop, source_face, target_face)


__all__ = ["SwapInferenceEngine"]
