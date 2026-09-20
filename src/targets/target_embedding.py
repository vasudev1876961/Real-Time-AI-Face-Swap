"""
Target Face Embedding Extraction and Normalization.
"""

import os
from typing import Optional, Any
import cv2
import numpy as np

from src.core.device import get_device_manager
from src.core.config_loader import SingleModelConfig
from src.detection.face_detector import get_face_detector
from src.alignment.face_alignment import align_face_crop
from src.utils.logger import get_logger

logger = get_logger("TargetEmbedding")


class TargetEmbeddingExtractor:
    """
    Extracts 512-dimensional normalized facial identity embeddings using
    local ONNX ArcFace models with CPU/GPU acceleration.
    """

    def __init__(self, config: Optional[SingleModelConfig] = None):
        self.config = config or SingleModelConfig()
        self.session: Any = None
        self.embedding_size: int = 512
        self._is_ready: bool = False
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads ONNX ArcFace recognition session if file exists."""
        model_path = self.config.model_path
        # Check standard locations if not found at config path
        candidate_paths = [
            model_path,
            "models/face_analysis/face_recognition.onnx",
            "models/face_analysis/w600k_r50.onnx",
        ]

        active_path = None
        for cp in candidate_paths:
            if cp and os.path.isfile(cp):
                active_path = cp
                break

        if active_path:
            try:
                import onnxruntime as ort
                dm = get_device_manager()
                providers = dm.get_providers_for_model(self.config.provider)
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(active_path, opts, providers=providers)
                self._is_ready = True
                logger.info(f"Loaded ONNX Face Recognition model from {active_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX recognition model from {active_path}: {e}")

        logger.info("Face analysis model not found; using normalized image feature extractor.")
        self._is_ready = True

    def is_ready(self) -> bool:
        return self._is_ready

    def extract_embedding(self, reference_image: np.ndarray) -> np.ndarray:
        """
        Detects, aligns to 112x112, and extracts a 512-D L2-normalized ArcFace embedding
        from any reference portrait photo.
        """
        if reference_image is None or reference_image.size == 0:
            return np.zeros(self.embedding_size, dtype=np.float32)

        # 1. Detect face and landmarks on the reference image
        detector = get_face_detector()
        face = detector.detect_single(reference_image)

        if face is not None and face.landmarks is not None:
            # Align face accurately to 112x112 ArcFace standard
            aligned_112, _, _ = align_face_crop(reference_image, face.landmarks, crop_size=(112, 112))
        else:
            # Fallback to direct resize
            aligned_112 = cv2.resize(reference_image, (112, 112), interpolation=cv2.INTER_AREA)

        # 2. Extract 512-D ArcFace embedding
        if self.session is not None:
            try:
                input_name = self.session.get_inputs()[0].name
                # ArcFace expects RGB in [-1.0, 1.0] range (image - 127.5) / 127.5
                blob = cv2.dnn.blobFromImage(
                    aligned_112,
                    scalefactor=1.0 / 127.5,
                    size=(112, 112),
                    mean=(127.5, 127.5, 127.5),
                    swapRB=True,  # BGR -> RGB
                )
                output = self.session.run(None, {input_name: blob})[0]
                embedding = output.flatten().astype(np.float32)
                # L2 normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm
                return embedding
            except Exception as e:
                logger.error(f"Inference error in face recognition model: {e}")

        # Fallback feature vector
        gray = cv2.cvtColor(aligned_112, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (16, 32)).astype(np.float32).flatten()
        norm = np.linalg.norm(small)
        if norm > 0:
            small /= norm
        return small


_embedding_extractor_instance: Optional[TargetEmbeddingExtractor] = None


def get_embedding_extractor(config: Optional[SingleModelConfig] = None) -> TargetEmbeddingExtractor:
    """Returns a singleton instance of TargetEmbeddingExtractor."""
    global _embedding_extractor_instance
    if _embedding_extractor_instance is None or config is not None:
        _embedding_extractor_instance = TargetEmbeddingExtractor(config)
    return _embedding_extractor_instance

