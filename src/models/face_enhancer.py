"""
ONNX Runtime Face Enhancement Adapter.
Supports GFPGAN, CodeFormer, or Real-ESRGAN face restoration ONNX models.
"""

import os
import time
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

from src.core.device import get_device_manager
from src.core.config_loader import SingleModelConfig
from src.utils.logger import get_logger

logger = get_logger("ONNXFaceEnhancer")


class ONNXFaceEnhancer:
    """
    Adapter for running face restoration and super-resolution ONNX models.
    Supports provider acceleration (CUDA, DirectML, CPU fallback).
    """

    def __init__(self, config: Optional[SingleModelConfig] = None):
        self.config = config or SingleModelConfig()
        self.session: Any = None
        self._is_loaded: bool = False
        self._last_latency_ms: float = 0.0
        self._input_name: str = ""
        self._output_name: str = ""
        self._input_size: Tuple[int, int] = (512, 512)
        self._active_provider: str = "None"

        if self.config.model_path:
            self.load(self.config.model_path, self.config.provider)

    def load(self, model_path: str, provider: str = "auto") -> bool:
        """Loads ONNX face restoration model."""
        self.unload()

        if not model_path or not os.path.isfile(model_path):
            logger.info(
                f"Face enhancer model not found at: '{model_path}'. "
                "Neural restoration disabled; adaptive fidelity filter will be used."
            )
            self._is_loaded = False
            return False

        try:
            import onnxruntime as ort
            dm = get_device_manager()
            providers = dm.get_providers_for_model(provider)

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(8, max(4, os.cpu_count() or 4))
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            logger.info(f"Loading ONNX face enhancer from '{model_path}' with {providers}...")
            self.session = ort.InferenceSession(model_path, opts, providers=providers)
            self._active_provider = self.session.get_providers()[0] if self.session.get_providers() else "CPUExecutionProvider"

            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if inputs and outputs:
                self._input_name = inputs[0].name
                self._output_name = outputs[0].name
                shape = inputs[0].shape
                if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                    self._input_size = (shape[3], shape[2])

            self._is_loaded = True
            logger.info(f"ONNX face enhancer ready. Input size: {self._input_size}, Provider: {self._active_provider}")
            return True

        except Exception as e:
            logger.warning(f"Could not load ONNX face enhancer from '{model_path}': {e}")
            self._is_loaded = False
            return False

    def unload(self) -> None:
        """Releases ONNX session."""
        if self.session is not None:
            del self.session
            self.session = None
        self._is_loaded = False
        self._active_provider = "None"

    def is_loaded(self) -> bool:
        return self._is_loaded

    def enhance(self, face_bgr: np.ndarray, blend_weight: float = 0.8) -> np.ndarray:
        """
        Enhances face crop using neural restoration model.
        Args:
            face_bgr: Input face crop in BGR format [H, W, 3], uint8.
            blend_weight: Interpolation ratio between enhanced result and original crop [0.0, 1.0].
        Returns:
            Restored BGR face crop [H, W, 3], uint8.
        """
        if not self._is_loaded or self.session is None or face_bgr is None or face_bgr.size == 0:
            return face_bgr

        t0 = time.perf_counter()
        orig_h, orig_w = face_bgr.shape[:2]

        try:
            # Preprocessing: resize to model input size, BGR -> RGB, normalize to [-1, 1]
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, self._input_size, interpolation=cv2.INTER_LANCZOS4)
            img_norm = (resized.astype(np.float32) / 127.5) - 1.0
            img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

            # Inference
            outputs = self.session.run([self._output_name], {self._input_name: img_chw})
            out_tensor = outputs[0][0]  # (3, H, W)

            # Postprocessing: denormalize, clip, RGB -> BGR, resize back
            out_hwc = np.transpose(out_tensor, (1, 2, 0))
            out_hwc = (np.clip(out_hwc, -1.0, 1.0) + 1.0) * 127.5
            out_hwc = out_hwc.astype(np.uint8)

            enhanced_bgr = cv2.cvtColor(out_hwc, cv2.COLOR_RGB2BGR)
            enhanced_bgr = cv2.resize(enhanced_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

            self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

            if blend_weight < 1.0:
                return cv2.addWeighted(enhanced_bgr, blend_weight, face_bgr, 1.0 - blend_weight, 0.0)
            return enhanced_bgr

        except Exception as e:
            logger.error(f"ONNX face enhancement error: {e}")
            return face_bgr

    def get_last_latency_ms(self) -> float:
        return self._last_latency_ms
