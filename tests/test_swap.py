"""
Unit Tests for Neural Face Swap Adapters, Feed Dict Pre-Processing, and Inference Engine.
"""

import os
import pytest
import numpy as np

from src.models.base_swapper import BaseFaceSwapper
from src.models.onnx_swapper import ONNXSwapper
from src.swap.inference import SwapInferenceEngine
from src.detection.face_detector import FaceData
from src.targets.target_loader import TargetFace
from src.core.config_loader import SingleModelConfig


def test_base_swapper_abc():
    class DummySwapper(BaseFaceSwapper):
        def load(self, model_path: str, provider: str = "auto") -> bool:
            return True
        def swap(self, aligned_face, source_face, target_face):
            return aligned_face
        def unload(self):
            pass
        def is_loaded(self):
            return True
        def get_last_latency_ms(self):
            return 0.0
        def get_model_info(self):
            return {"name": "dummy"}

    swapper = DummySwapper()
    assert swapper.is_loaded() is True
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    assert swapper.swap(dummy, None, None) is dummy


def test_onnx_swapper_missing_model_fallback():
    cfg = SingleModelConfig(model_path="models/face_swap/non_existent.onnx")
    swapper = ONNXSwapper(config=cfg)
    assert swapper.is_loaded() is False
    assert swapper.session is None


def test_onnx_swapper_build_feed_dict():
    swapper = ONNXSwapper(config=SingleModelConfig(model_path=""))
    crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    emb = np.random.randn(512).astype(np.float32)

    feed = swapper._build_feed_dict(crop, emb)
    assert isinstance(feed, dict)
    assert len(feed) >= 1
    # Check tensor shape: standard INSwapper input is (1, 3, 128, 128)
    first_tensor = list(feed.values())[0]
    assert first_tensor.shape == (1, 3, 128, 128)
    assert first_tensor.dtype == np.float32


def test_swap_inference_engine_facade():
    engine = SwapInferenceEngine()
    assert engine is not None
    assert isinstance(engine.is_ready(), bool)
