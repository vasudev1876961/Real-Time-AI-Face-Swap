"""
Face Mask Generation, Color Correction, Seamless Blending, and Postprocessing.
"""

from src.processing.mask import (
    FaceMaskGenerator,
    create_face_mask,
    feather_mask_distance_transform,
    get_anatomical_facial_contour,
)
from src.processing.color_correction import (
    reinhard_color_transfer,
    gain_color_match,
    match_histograms,
    apply_color_correction,
    TemporalColorStabilizer,
)
from src.processing.blending import (
    FaceBlender,
    blend_face_into_frame,
    compute_crop_roi,
)
from src.processing.postprocess import postprocess_frame
from src.processing.enhancement import AdaptiveFidelityEnhancer, FaceEnhancer
from src.processing.occlusion import OcclusionDetector
from src.processing.stabilizer import TemporalMotionStabilizer

__all__ = [
    "FaceMaskGenerator",
    "create_face_mask",
    "feather_mask_distance_transform",
    "get_anatomical_facial_contour",
    "reinhard_color_transfer",
    "gain_color_match",
    "match_histograms",
    "apply_color_correction",
    "TemporalColorStabilizer",
    "FaceBlender",
    "blend_face_into_frame",
    "compute_crop_roi",
    "postprocess_frame",
    "AdaptiveFidelityEnhancer",
    "FaceEnhancer",
    "OcclusionDetector",
    "TemporalMotionStabilizer",
]

