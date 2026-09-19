"""
Facial Landmark Geometric Utilities and Normalization Functions.
"""

from typing import Tuple, Optional
import numpy as np
from src.detection.face_landmarks import (
    ARCFACE_STANDARD_112,
    INSWAPPER_STANDARD_128,
    ARCFACE_STANDARD_512,
    FaceLandmarks,
    extract_convex_hull_mask,
    generate_face_mask_from_landmarks,
)

__all__ = [
    "ARCFACE_STANDARD_112",
    "INSWAPPER_STANDARD_128",
    "ARCFACE_STANDARD_512",
    "FaceLandmarks",
    "extract_convex_hull_mask",
    "generate_face_mask_from_landmarks",
]
