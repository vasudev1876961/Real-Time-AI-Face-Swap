"""
Face Mask Generation Module (Re-export and Interface Adapter).
Bridges to src.processing.mask for anatomical convex hulls and distance-transform feathering.
"""

from src.processing.mask import (
    FaceMaskGenerator,
    create_face_mask,
    create_feathered_mask,
    get_anatomical_facial_contour,
    feather_mask_distance_transform,
)

__all__ = [
    "FaceMaskGenerator",
    "create_face_mask",
    "create_feathered_mask",
    "get_anatomical_facial_contour",
    "feather_mask_distance_transform",
]
