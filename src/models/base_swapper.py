"""
Abstract Base Class for Face Swapper Implementations.

Guarantees complete modularity so any custom or future trained model
can replace the default ONNX inference adapter without modifying the rest of the app.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from src.detection.face_detector import FaceData
from src.targets.target_loader import TargetFace


class BaseFaceSwapper(ABC):
    """
    Standard interface for all face-swap model backends.
    """

    @abstractmethod
    def load(self, model_path: str, provider: str = "auto") -> bool:
        """
        Loads model weights from local filesystem.
        Returns True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def swap(
        self,
        aligned_face: np.ndarray,
        source_face: FaceData,
        target_face: TargetFace,
    ) -> np.ndarray:
        """
        Performs face swap inference on an aligned face crop using target identity.

        Args:
            aligned_face: (H, W, 3) BGR image of aligned source face.
            source_face: Detected FaceData structure of the source person.
            target_face: TargetFace containing identity embedding and metadata.

        Returns:
            (H, W, 3) BGR swapped face crop.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Releases session resources and GPU memory."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if the model is currently active and ready for inference."""
        pass

    @abstractmethod
    def get_last_latency_ms(self) -> float:
        """Returns the execution time in milliseconds for the most recent inference pass."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Returns metadata dictionary about the active model (inputs, outputs, provider)."""
        pass
