"""
Face Mask Generation with Anatomical Curved Contour, Distance-Transform Feathering,
Pose-Adaptive Shaping, and Microsecond Parametric Caching.
"""

from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np

from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.core.config_loader import ProcessingConfig
from src.utils.logger import get_logger

logger = get_logger("FaceMask")


def smooth_contour_spline(pts: np.ndarray, num_subdivisions: int = 4) -> np.ndarray:
    """
    Subdivides and smooths a closed polygon using Chaikin's corner-cutting algorithm
    to produce an organic, rounded anatomical contour without sharp angular vertices.
    """
    if len(pts) < 3:
        return pts

    curr = pts.astype(np.float32)
    for _ in range(num_subdivisions):
        n = len(curr)
        next_pts = []
        for i in range(n):
            p0 = curr[i]
            p1 = curr[(i + 1) % n]
            # Chaikin cut: 75% p0 + 25% p1, and 25% p0 + 75% p1
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            next_pts.append(q)
            next_pts.append(r)
        curr = np.array(next_pts, dtype=np.float32)

    return np.round(curr).astype(np.int32)


def get_anatomical_facial_contour(
    crop_shape: Tuple[int, int],
    landmarks: np.ndarray,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> np.ndarray:
    """
    Constructs a 16-point anatomical facial boundary curve matching human facial anatomy:
    forehead below hairline, temples, zygomatic arches, mid-cheeks, mandibular jaw, and chin.
    Incorporates yaw and pitch adjustments to adapt boundaries when the head turns.
    """
    h, w = crop_shape[:2]
    pts = landmarks[:5].astype(np.float32)

    eye_center = (pts[0] + pts[1]) / 2.0
    eye_dist = max(float(np.linalg.norm(pts[1] - pts[0])), 12.0)

    # Pose normalization factors [-1.0, 1.0]
    norm_yaw = float(np.clip(yaw / 45.0, -1.0, 1.0))
    norm_pitch = float(np.clip(pitch / 45.0, -1.0, 1.0))

    # Lateral shifts for pose turn (pull the turned-away cheek inwards)
    left_pull = 1.0 - max(0.0, norm_yaw) * 0.35
    right_pull = 1.0 - max(0.0, -norm_yaw) * 0.35

    # Forehead apex (stays comfortably below hairline)
    forehead_pitch_shift = -norm_pitch * 0.15 * eye_dist
    forehead = eye_center - (pts[2] - eye_center) * 0.58 + np.array([norm_yaw * eye_dist * 0.1, forehead_pitch_shift], dtype=np.float32)

    # Forehead lateral arches
    forehead_left = pts[0] + np.array([-eye_dist * 0.22 * left_pull, -eye_dist * 0.46], dtype=np.float32)
    forehead_right = pts[1] + np.array([eye_dist * 0.22 * right_pull, -eye_dist * 0.46], dtype=np.float32)

    # Temples
    temple_left = pts[0] + np.array([-eye_dist * 0.44 * left_pull, -eye_dist * 0.18], dtype=np.float32)
    temple_right = pts[1] + np.array([eye_dist * 0.44 * right_pull, -eye_dist * 0.18], dtype=np.float32)

    # Zygomatic arches (upper cheekbones)
    zygoma_left = pts[0] + np.array([-eye_dist * 0.52 * left_pull, eye_dist * 0.32], dtype=np.float32)
    zygoma_right = pts[1] + np.array([eye_dist * 0.52 * right_pull, eye_dist * 0.32], dtype=np.float32)

    # Mid cheeks
    mid_cheek_left = pts[3] + np.array([-eye_dist * 0.42 * left_pull, -eye_dist * 0.08], dtype=np.float32)
    mid_cheek_right = pts[4] + np.array([eye_dist * 0.42 * right_pull, -eye_dist * 0.08], dtype=np.float32)

    # Lower jaw angles
    jaw_left = pts[3] + np.array([-eye_dist * 0.32 * left_pull, eye_dist * 0.24], dtype=np.float32)
    jaw_right = pts[4] + np.array([eye_dist * 0.32 * right_pull, eye_dist * 0.24], dtype=np.float32)

    # Lateral chin transitions
    chin_lat_left = pts[3] + np.array([-eye_dist * 0.14 * left_pull, eye_dist * 0.48], dtype=np.float32)
    chin_lat_right = pts[4] + np.array([eye_dist * 0.14 * right_pull, eye_dist * 0.48], dtype=np.float32)

    # Chin apex
    chin_pitch_shift = norm_pitch * 0.18 * eye_dist
    mouth_center = (pts[3] + pts[4]) / 2.0
    chin_bottom = mouth_center + (mouth_center - pts[2]) * 0.60 + np.array([norm_yaw * eye_dist * 0.08, chin_pitch_shift], dtype=np.float32)

    raw_polygon = np.array([
        forehead,
        forehead_right,
        temple_right,
        zygoma_right,
        mid_cheek_right,
        jaw_right,
        chin_lat_right,
        chin_bottom,
        chin_lat_left,
        jaw_left,
        mid_cheek_left,
        zygoma_left,
        temple_left,
        forehead_left,
    ], dtype=np.float32)

    # Smooth the 14 control points into a curved 56-point anatomical polygon
    smooth_poly = smooth_contour_spline(raw_polygon, num_subdivisions=2)
    # Clamp within crop boundaries
    smooth_poly[:, 0] = np.clip(smooth_poly[:, 0], 0, w - 1)
    smooth_poly[:, 1] = np.clip(smooth_poly[:, 1], 0, h - 1)
    return smooth_poly


def feather_mask_distance_transform(
    binary_mask: np.ndarray,
    radius: float = 7.0,
    falloff: str = "smoothstep",
) -> np.ndarray:
    """
    Applies exact Euclidean Signed Distance Field (SDF) feathering to eliminate edge banding.
    Provides mathematically guaranteed C1 continuity (zero first-derivative at both 0.0 and 1.0).
    """
    if radius <= 0.5:
        return (binary_mask > 127).astype(np.float32)

    # Compute Euclidean distance inside and outside the binary mask
    dist_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 5)

    # Signed distance: positive inside, negative outside
    signed_dist = dist_inside - dist_outside

    # Normalize across the transition radius [-radius, +radius] -> [0.0, 1.0]
    t = np.clip((signed_dist + radius) / (2.0 * radius), 0.0, 1.0)

    if falloff == "cosine":
        # Raised cosine S-curve: zero derivative at t=0 and t=1
        feathered = 0.5 - 0.5 * np.cos(np.pi * t)
    else:
        # Standard Hermite smoothstep: 3*t^2 - 2*t^3
        feathered = t * t * (3.0 - 2.0 * t)

    return feathered.astype(np.float32)


from src.processing.mask_precision import (
    DenseMeshContourProjector,
    AnatomicalContourBuilder,
    ActiveEdgeBoundarySnapper,
    ForeheadHairlineCarver,
    CurvatureAdaptiveFeatherer,
    TemporalMaskStabilizer,
    draw_mask_contour_hud,
)


def create_face_mask(
    crop_shape: Tuple[int, int] = (128, 128),
    mask_type: str = "smooth_hull",
    blur_kernel_size: int = 15,
    feather_factor: float = 0.6,
    erosion_pixels: int = 1,
    landmarks: Optional[np.ndarray] = None,
    yaw: float = 0.0,
    pitch: float = 0.0,
    mesh_landmarks: Optional[np.ndarray] = None,
    affine_mat: Optional[np.ndarray] = None,
    aligned_crop: Optional[np.ndarray] = None,
    enable_edge_snapping: bool = True,
    edge_snapping_strength: float = 0.65,
    enable_curvature_feathering: bool = True,
    enable_hairline_carving: bool = True,
    hairline_carving_strength: float = 0.50,
) -> np.ndarray:
    """
    Generates an optimized, feathered single-channel float32 mask [0.0, 1.0] for aligned face crop.
    Phase 13: Sub-pixel dense mesh projection, active edge snapping, hairline carving,
    and directional curvature-adaptive distance-transform feathering.

    Supported mask types:
      - 'smooth_hull': 24-point anatomical curve or 468p dense mesh contour (Recommended)
      - 'dense_mesh': Sub-pixel 468p MediaPipe face oval projection
      - 'distance_transform': Anatomical contour with exact Euclidean distance transform
      - 'pose_adaptive': Dynamic anatomical contour shaped by yaw and pitch angles
      - 'convex_hull': Standard 6-point landmark convex polygon with Gaussian feathering
      - 'elliptical': Classical oval template with Gaussian feathering
    """
    h, w = crop_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    mtype = (mask_type or "smooth_hull").lower().strip()

    has_valid_landmarks = landmarks is not None and len(landmarks) >= 5
    lms = landmarks if has_valid_landmarks else INSWAPPER_STANDARD_128

    poly = None

    # 1. High-precision dense mesh projection if 468p landmarks and affine matrix are available
    if mesh_landmarks is not None and affine_mat is not None and mtype in ("smooth_hull", "dense_mesh", "pose_adaptive"):
        poly = DenseMeshContourProjector.project_mesh_contour(
            mesh_landmarks=mesh_landmarks,
            affine_mat=affine_mat,
            crop_shape=(h, w),
            subdivisions=2,
        )

    # 2. Advanced 24-point anatomical curve fallback
    if poly is None and mtype in ("smooth_hull", "pose_adaptive", "distance_transform", "dense_mesh"):
        poly = AnatomicalContourBuilder.build_contour_24p(
            crop_shape=(h, w),
            landmarks=lms,
            yaw=yaw,
            pitch=pitch,
        )

    if poly is not None:
        # 3. Active Edge Snapping to actual skin boundaries in aligned_crop
        if enable_edge_snapping and aligned_crop is not None and edge_snapping_strength > 0.01:
            snapper = ActiveEdgeBoundarySnapper(search_radius=4, strength=edge_snapping_strength)
            poly = snapper.snap_contour_to_edges(poly, aligned_crop)

        cv2.fillPoly(binary_mask, [poly], 255)

    elif mtype == "convex_hull":
        if has_valid_landmarks:
            pts = lms[:5].astype(np.float32)
            eye_center = (pts[0] + pts[1]) / 2.0
            eye_dist = max(float(np.linalg.norm(pts[1] - pts[0])), 10.0)
            forehead = eye_center - (pts[2] - eye_center) * 0.55
            chin = (pts[3] + pts[4]) / 2.0 + ((pts[3] + pts[4]) / 2.0 - pts[2]) * 0.65
            left_temple = pts[0] + np.array([-eye_dist * 0.45, -eye_dist * 0.25], dtype=np.float32)
            right_temple = pts[1] + np.array([eye_dist * 0.45, -eye_dist * 0.25], dtype=np.float32)
            left_jaw = pts[3] + np.array([-eye_dist * 0.35, eye_dist * 0.2], dtype=np.float32)
            right_jaw = pts[4] + np.array([eye_dist * 0.35, eye_dist * 0.2], dtype=np.float32)
            poly_conv = np.array([forehead, right_temple, right_jaw, chin, left_jaw, left_temple], dtype=np.int32)
            hull = cv2.convexHull(poly_conv)
            cv2.fillConvexPoly(binary_mask, hull, 255)
        else:
            center = (int(w * 0.50), int(h * 0.54))
            axes = (int(w * 0.36), int(h * 0.39))
            cv2.ellipse(binary_mask, center, axes, 0, 0, 360, 255, -1)
    else:  # elliptical
        center = (int(w * 0.50), int(h * 0.54))
        axes = (int(w * 0.36), int(h * 0.39))
        cv2.ellipse(binary_mask, center, axes, 0, 0, 360, 255, -1)

    # 4. Forehead Hairline and Bangs Silhouette Carving
    if enable_hairline_carving and aligned_crop is not None and hairline_carving_strength > 0.01:
        carver = ForeheadHairlineCarver(strength=hairline_carving_strength)
        binary_mask = carver.carve_hairline(binary_mask, aligned_crop, landmarks=lms)

    # 5. Morphological erosion to inset boundary away from extreme margins
    if erosion_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_pixels * 2 + 1, erosion_pixels * 2 + 1))
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    # 6. Distance-Transform Feathering (Curvature-Adaptive or Uniform)
    if mtype in ("smooth_hull", "pose_adaptive", "distance_transform", "dense_mesh"):
        feather_radius = max(2.0, (blur_kernel_size * 0.5) * max(0.2, feather_factor))
        if enable_curvature_feathering:
            mask_normalized = CurvatureAdaptiveFeatherer.create_curvature_feathered_mask(
                binary_mask, base_radius=feather_radius
            )
        else:
            mask_normalized = feather_mask_distance_transform(binary_mask, radius=feather_radius, falloff="smoothstep")
    else:
        # Standard Gaussian feathering for backward compatibility
        ksize = max(5, blur_kernel_size | 1)
        sigma = max(1.0, (ksize / 3.0) * max(0.2, feather_factor))
        blurred = cv2.GaussianBlur(binary_mask.astype(np.float32), (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        mask_normalized = np.clip(blurred / 255.0, 0.0, 1.0)

    return mask_normalized


class FaceMaskGenerator:
    """
    High-Performance Configurable Mask Generation Engine with Parametric Caching
    and Phase 13 Sub-Pixel Anatomical Fitting.
    Eliminates redundant morphological and filtering operations at 30-60+ FPS.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self._max_cache_size = 64
        self._stabilizer = TemporalMaskStabilizer(alpha=0.75)

    def clear_cache(self) -> None:
        """Clears the internal mask cache and temporal stabilizer."""
        self._cache.clear()
        self._stabilizer.reset()

    def generate_mask(
        self,
        crop_shape: Tuple[int, int] = (128, 128),
        landmarks: Optional[np.ndarray] = None,
        feather_override: Optional[float] = None,
        mask_type_override: Optional[str] = None,
        yaw: float = 0.0,
        pitch: float = 0.0,
        aligned_crop: Optional[np.ndarray] = None,
        occlusion_detector: Optional[Any] = None,
        track_id: Optional[int] = None,
        mesh_landmarks: Optional[np.ndarray] = None,
        affine_mat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Retrieves or generates an optimized, anatomically fitted face mask.
        Supports dense mesh projection, edge snapping, curvature feathering,
        parametric caching, and temporal multi-frame stabilization.
        """
        m_type = mask_type_override or getattr(self.config, "mask_type", "smooth_hull")
        feather = feather_override if feather_override is not None else getattr(self.config, "mask_feather", 0.6)
        blur_k = getattr(self.config, "mask_blur", 15)
        erosion = getattr(self.config, "mask_erosion", 1)

        enable_edge_snap = getattr(self.config, "enable_edge_snapping", True)
        edge_snap_str = getattr(self.config, "edge_snapping_strength", 0.65)
        enable_curv_feather = getattr(self.config, "enable_curvature_feathering", True)
        enable_hair_carve = getattr(self.config, "enable_hairline_carving", True)
        hair_carve_str = getattr(self.config, "hairline_carving_strength", 0.50)

        # Quantize pose angles to 5-degree increments when caching is viable
        yaw_q = round(yaw / 5.0) * 5.0 if m_type == "pose_adaptive" else 0.0
        pitch_q = round(pitch / 5.0) * 5.0 if m_type == "pose_adaptive" else 0.0

        # If dense mesh is present or dynamic edge snapping is active, compute dynamically
        can_cache = (
            mesh_landmarks is None
            and not (enable_edge_snap and aligned_crop is not None)
            and not (enable_hair_carve and aligned_crop is not None)
        )

        is_standard = landmarks is None or np.allclose(landmarks[:5], INSWAPPER_STANDARD_128[:5], atol=1.0)
        lms_key = "std" if is_standard else tuple(np.round(landmarks[:5].flatten(), decimals=1))

        cache_key = (
            crop_shape[:2],
            m_type,
            blur_k,
            round(feather, 2),
            erosion,
            yaw_q,
            pitch_q,
            lms_key,
            enable_curv_feather,
        )

        if can_cache and cache_key in self._cache:
            base_res = self._cache[cache_key].copy()
        else:
            base_res = create_face_mask(
                crop_shape=crop_shape,
                mask_type=m_type,
                blur_kernel_size=blur_k,
                feather_factor=feather,
                erosion_pixels=erosion,
                landmarks=landmarks,
                yaw=yaw_q,
                pitch=pitch_q,
                mesh_landmarks=mesh_landmarks,
                affine_mat=affine_mat,
                aligned_crop=aligned_crop,
                enable_edge_snapping=enable_edge_snap,
                edge_snapping_strength=edge_snap_str,
                enable_curvature_feathering=enable_curv_feather,
                enable_hairline_carving=enable_hair_carve,
                hairline_carving_strength=hair_carve_str,
            )

            if can_cache:
                if len(self._cache) >= self._max_cache_size:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                self._cache[cache_key] = base_res.copy()

        # Phase 13 Temporal Matte Stabilization
        tid = track_id if track_id is not None else 1
        base_res = self._stabilizer.stabilize(base_res, track_id=tid)

        # Apply occlusion refinement if detector and crop are available
        if occlusion_detector is not None and aligned_crop is not None:
            return occlusion_detector.refine_mask_with_occlusion(
                base_res, aligned_crop, landmarks, track_id=track_id
            )

        return base_res

