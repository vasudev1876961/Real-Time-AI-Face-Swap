"""
ONNX Runtime Face Swapper Adapter Implementation.
"""

import os
import time
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

from src.models.base_swapper import BaseFaceSwapper
from src.detection.face_detector import FaceData
from src.targets.target_loader import TargetFace
from src.core.device import get_device_manager, get_hardware_autotuner
from src.core.config_loader import SingleModelConfig
from src.utils.logger import get_logger

logger = get_logger("ONNXSwapper")


class ONNXSwapper(BaseFaceSwapper):
    """
    Production ONNX Runtime adapter for neural face swapping models.
    Supports dynamic/fixed batch shapes, provider acceleration, and warmup passes.
    """

    def __init__(self, config: Optional[SingleModelConfig] = None):
        self.config = config or SingleModelConfig()
        self.session: Any = None
        self._is_loaded: bool = False
        self._last_latency_ms: float = 0.0
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, List[Any]] = {}
        self._active_provider: str = "None"
        self._model_path: str = ""
        self._input_size: Tuple[int, int] = (128, 128)
        self.emap: Optional[np.ndarray] = None

        if self.config.model_path:
            self.load(self.config.model_path, self.config.provider)

    def load(self, model_path: str, provider: str = "auto") -> bool:
        """Loads ONNX model weights and initializes execution provider."""
        self.unload()
        self._model_path = model_path

        if not model_path or not os.path.isfile(model_path):
            logger.warning(
                f"Face swap model file not found at: '{model_path}'. "
                "Face swap is disabled (Camera preview and face tracking remain active). "
                "Place your compatible ONNX model in 'models/face_swap/'."
            )
            self._is_loaded = False
            return False

        try:
            import onnxruntime as ort
            dm = get_device_manager()
            providers = dm.get_providers_for_model(provider)
            autotuner = get_hardware_autotuner()
            opts = autotuner.get_optimized_session_options()
            if opts is None:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = min(8, max(4, os.cpu_count() or 4))
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            logger.info(f"Loading ONNX swap model from '{model_path}' with providers {providers}...")
            self.session = ort.InferenceSession(model_path, opts, providers=providers)
            self._active_provider = self.session.get_providers()[0] if self.session.get_providers() else "CPUExecutionProvider"

            # Parse inputs and outputs
            self._input_names = [inp.name for inp in self.session.get_inputs()]
            self._output_names = [out.name for out in self.session.get_outputs()]
            for inp in self.session.get_inputs():
                self._input_shapes[inp.name] = inp.shape

            # Detect input resolution
            for inp in self.session.get_inputs():
                shape = inp.shape
                if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                    self._input_size = (shape[3], shape[2])
                    break

            # Extract 512x512 emap projection matrix if embedded in ONNX initializers (standard for inswapper_128)
            self.emap: Optional[np.ndarray] = None
            try:
                import onnx
                from onnx import numpy_helper
                onnx_model = onnx.load(model_path)
                # The true emap is the last initializer in INSwapper ONNX graph (name='initializer', shape=[512, 512])
                if onnx_model.graph.initializer:
                    last_init = onnx_model.graph.initializer[-1]
                    if list(last_init.dims) == [512, 512]:
                        self.emap = numpy_helper.to_array(last_init).astype(np.float32)
                        logger.info(f"Extracted embedding mapping matrix (emap) '{last_init.name}': shape {self.emap.shape}")
                    else:
                        for init in reversed(onnx_model.graph.initializer):
                            if list(init.dims) == [512, 512] and init.name.lower() in ("initializer", "emap"):
                                self.emap = numpy_helper.to_array(init).astype(np.float32)
                                logger.info(f"Extracted embedding mapping matrix (emap) '{init.name}': shape {self.emap.shape}")
                                break
            except Exception as emap_err:
                logger.debug(f"Could not extract emap from ONNX graph: {emap_err}")

            logger.info(f"Swap model loaded successfully. Inputs: {self._input_names}, Outputs: {self._output_names}, Resolution: {self._input_size}")
            self._is_loaded = True

            # Warmup inference
            self._warmup()
            return True

        except Exception as e:
            logger.error(f"Failed to load ONNX swap model: {e}")
            self.unload()
            return False

    def _warmup(self) -> None:
        """Executes a single dummy inference pass to pre-allocate runtime memory."""
        if not self._is_loaded or not self.session:
            return
        try:
            dummy_face = np.zeros((self._input_size[1], self._input_size[0], 3), dtype=np.uint8)
            dummy_emb = np.zeros(512, dtype=np.float32)
            feed_dict = self._build_feed_dict(dummy_face, dummy_emb)
            self.session.run(self._output_names, feed_dict)
            logger.info("Model warmup completed successfully.")
        except Exception as e:
            logger.debug(f"Warmup warning: {e}")

    def _build_feed_dict(self, face_crop: np.ndarray, target_emb: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepares input dictionary according to model's expected node names."""
        in_w, in_h = self._input_size
        if face_crop.shape[:2] != (in_h, in_w):
            interp = cv2.INTER_AREA if (face_crop.shape[0] > in_h) else cv2.INTER_LANCZOS4
            resized = cv2.resize(face_crop, (in_w, in_h), interpolation=interp)
        else:
            resized = face_crop

        # INSwapper standard: RGB format in [0.0, 1.0] range (HWC -> NCHW)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            rgb_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb_crop = resized
        img_tensor = np.transpose(rgb_crop.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # Target ArcFace Embedding: shape (1, 512), L2 normalized
        emb = target_emb.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Map embedding using emap projection if available (essential for INSwapper identity transfer)
        if self.emap is not None and emb.size == 512:
            emb = np.dot(emb.reshape(1, 512), self.emap).flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

        emb_tensor = emb.reshape(1, 512).astype(np.float32)

        feed_dict = {}
        input_names = self._input_names if self._input_names else ["target", "source"]
        for name in input_names:
            name_lower = name.lower()
            if "emb" in name_lower or "latent" in name_lower or "id" in name_lower or "source" in name_lower:
                feed_dict[name] = emb_tensor
            elif "target" in name_lower or "img" in name_lower or "input" in name_lower:
                feed_dict[name] = img_tensor
            else:
                shape = self._input_shapes.get(name, [])
                if len(shape) == 2 or (len(shape) >= 2 and shape[-1] == 512):
                    feed_dict[name] = emb_tensor
                else:
                    feed_dict[name] = img_tensor

        return feed_dict

    def swap(
        self,
        aligned_face: np.ndarray,
        source_face: FaceData,
        target_face: TargetFace,
    ) -> np.ndarray:
        """
        Executes face swap inference using target embedding and aligned face crop.
        """
        if not self._is_loaded or self.session is None:
            self._last_latency_ms = 0.0
            return aligned_face

        if aligned_face is None or aligned_face.size == 0 or target_face is None:
            return aligned_face

        orig_h, orig_w = aligned_face.shape[:2]
        t0 = time.perf_counter()

        try:
            feed_dict = self._build_feed_dict(aligned_face, target_face.embedding)
            outputs = self.session.run(self._output_names, feed_dict)
            output_tensor = outputs[0]

            # Output tensor (1, 3, H, W) NCHW in RGB format
            if len(output_tensor.shape) == 4:
                if output_tensor.shape[1] == 3:  # NCHW
                    out_img = np.transpose(output_tensor[0], (1, 2, 0))
                else:  # NHWC
                    out_img = output_tensor[0]
            else:
                out_img = output_tensor

            # INSwapper output is RGB in range [0.0, 1.0]
            if out_img.max() <= 1.05 and out_img.min() >= -0.05:
                out_rgb = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
            elif out_img.min() < -0.1:  # [-1, 1] range
                out_rgb = np.clip((out_img * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            else:
                out_rgb = np.clip(out_img, 0, 255).astype(np.uint8)

            # Convert RGB back to BGR for OpenCV pipeline
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            # Resize back to requested aligned crop dimensions
            if (out_bgr.shape[1], out_bgr.shape[0]) != (orig_w, orig_h):
                out_bgr = cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

            self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
            return out_bgr

        except Exception as e:
            logger.error(f"Error during ONNX swap inference: {e}")
            self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
            return aligned_face

    def unload(self) -> None:
        """Releases session resources."""
        self.session = None
        self._is_loaded = False
        self._input_names = []
        self._output_names = []
        self._input_shapes = {}
        logger.info("ONNX Swapper session unloaded.")

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_last_latency_ms(self) -> float:
        return self._last_latency_ms

    def get_model_info(self) -> dict:
        return {
            "loaded": self._is_loaded,
            "model_path": self._model_path,
            "provider": self._active_provider,
            "inputs": self._input_names,
            "outputs": self._output_names,
            "resolution": self._input_size,
            "last_latency_ms": round(self._last_latency_ms, 2),
        }
