"""
Face Blender Module (Re-export and Interface Adapter).
Bridges to src.processing.blending for multi-band, alpha, and seamless cloning.
"""

from src.processing.blending import (
    FaceBlender,
    blend_face_crop,
    alpha_blend,
    seamless_clone,
    multiband_blend_crop,
)

__all__ = [
    "FaceBlender",
    "blend_face_crop",
    "alpha_blend",
    "seamless_clone",
    "multiband_blend_crop",
]
