"""
Blending and Visual Processing Compatibility Package.
"""

from src.blending.blender import FaceBlender, blend_face_crop, alpha_blend, seamless_clone
from src.blending.color_correction import apply_color_correction, reinhard_color_transfer
from src.blending.face_mask import FaceMaskGenerator, create_face_mask
from src.blending.post_processing import postprocess_frame, apply_unsharp_mask

__all__ = [
    "FaceBlender",
    "blend_face_crop",
    "alpha_blend",
    "seamless_clone",
    "apply_color_correction",
    "reinhard_color_transfer",
    "FaceMaskGenerator",
    "create_face_mask",
    "postprocess_frame",
    "apply_unsharp_mask",
]
