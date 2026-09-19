"""
Face Swapper Interface and ONNX Adapter.
"""

from src.models.base_swapper import BaseFaceSwapper
from src.models.onnx_swapper import ONNXSwapper

__all__ = ["BaseFaceSwapper", "ONNXSwapper"]
