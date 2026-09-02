"""
Unit Tests for Model Abstraction, ONNX Swapper, and Model Manager.
"""

import pytest
import numpy as np

from src.models.base_swapper import BaseFaceSwapper
from src.models.onnx_swapper import ONNXSwapper
from src.models.model_manager import ModelManager, get_model_manager
from src.core.config_loader import ModelsConfig, SingleModelConfig
from src.detection.face_detector import FaceData
from src.targets.target_loader import TargetFace, TargetMetadata


def test_base_swapper_contract():
    # Subclass implementing BaseFaceSwapper
    class CustomSwapper(BaseFaceSwapper):
        def __init__(self):
            self._loaded = False

        def load(self, model_path: str, provider: str = "auto") -> bool:
            self._loaded = True
            return True

        def swap(self, aligned_face, source_face, target_face):
            return aligned_face

        def unload(self):
            self._loaded = False

        def is_loaded(self) -> bool:
            return self._loaded

        def get_last_latency_ms(self) -> float:
            return 1.5

        def get_model_info(self) -> dict:
            return {"loaded": self._loaded}

    swapper = CustomSwapper()
    assert not swapper.is_loaded()
    assert swapper.load("models/custom.onnx") is True
    assert swapper.is_loaded()
    assert swapper.get_last_latency_ms() == 1.5


def test_onnx_swapper_graceful_missing_model():
    swapper = ONNXSwapper(SingleModelConfig(model_path="models/face_swap/non_existent.onnx"))
    assert swapper.is_loaded() is False
    info = swapper.get_model_info()
    assert info["loaded"] is False

    dummy_face = np.zeros((128, 128, 3), dtype=np.uint8)
    kps = np.zeros((5, 2), dtype=np.float32)
    src_data = FaceData(bbox=(0, 0, 128, 128), score=0.9, landmarks=kps)
    meta = TargetMetadata(person_id="test", display_name="Test", category="actors")
    tgt = TargetFace(target_id="test", display_name="Test", category="actors", reference_image_path="", reference_image=dummy_face, embedding=np.zeros(512), metadata=meta)

    # Should safely return original face without crashing
    out = swapper.swap(dummy_face, src_data, tgt)
    assert out.shape == dummy_face.shape


def test_model_manager_status():
    mm = ModelManager()
    status = mm.get_status_summary()
    assert "detector_ready" in status
    assert "swapper_ready" in status
    assert "device_active_provider" in status
