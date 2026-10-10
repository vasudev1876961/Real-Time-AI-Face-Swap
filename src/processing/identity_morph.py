"""
Identity Morphing & Dual-Target Fusion Engine (Phase 15).
Enables seamless spherical linear interpolation (SLERP) between identity embeddings,
real-time dual-target facial fusion, and smooth cinematic morph transitions.
"""

import time
import math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from src.targets.target_loader import TargetFace, TargetMetadata
from src.utils.logger import get_logger

logger = get_logger("IdentityMorphEngine")


def slerp(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    """
    Performs Spherical Linear Interpolation (SLERP) between two unit-normalized embedding vectors.
    
    Args:
        p0: Source embedding vector of shape (D,)
        p1: Target embedding vector of shape (D,)
        t: Interpolation parameter in [0.0, 1.0]
        
    Returns:
        Interpolated unit-normalized embedding vector of shape (D,)
    """
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return p0.copy()
    if t >= 1.0:
        return p1.copy()

    # Ensure float32 and 1D
    v0 = p0.astype(np.float32).flatten()
    v1 = p1.astype(np.float32).flatten()

    norm0 = np.linalg.norm(v0)
    norm1 = np.linalg.norm(v1)

    if norm0 > 1e-6:
        v0 = v0 / norm0
    if norm1 > 1e-6:
        v1 = v1 / norm1

    dot = float(np.dot(v0, v1))
    dot = max(-1.0, min(1.0, dot))

    # If vectors are virtually parallel, fall back to normalized LERP
    if abs(dot) > 0.9995:
        linear = (1.0 - t) * v0 + t * v1
        norm_lin = np.linalg.norm(linear)
        return (linear / norm_lin) if norm_lin > 1e-6 else linear

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)

    if abs(sin_theta_0) < 1e-6:
        return v0.copy()

    theta_t = theta_0 * t
    sin_theta_t = math.sin(theta_t)

    s0 = math.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    res = (s0 * v0) + (s1 * v1)
    norm_res = np.linalg.norm(res)
    if norm_res > 1e-6:
        res = res / norm_res
    return res.astype(np.float32)


def smoothstep(t: float) -> float:
    """Hermite smoothstep easing curve S(t) = 3t^2 - 2t^3."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class IdentityMorphEngine:
    """
    Manages continuous identity interpolation and dual-target fusion.
    Guarantees jitter-free, seamless identity transitions between target identities
    and enables multi-identity weighted blending.
    """

    def __init__(self, default_duration: float = 0.50):
        """
        Args:
            default_duration: Default transition time in seconds for target morphs.
        """
        self.default_duration = max(0.05, float(default_duration))
        self.is_morphing = False
        self.morph_progress = 0.0
        self.morph_start_time = 0.0
        self.morph_duration = self.default_duration

        self.source_target: Optional[TargetFace] = None
        self.destination_target: Optional[TargetFace] = None
        self.cached_fused_target: Optional[TargetFace] = None
        self._last_fused_key: Optional[Tuple[str, str, float]] = None

    def fuse_targets(
        self,
        target_a: TargetFace,
        target_b: TargetFace,
        ratio: float = 0.5,
    ) -> TargetFace:
        """
        Blends two TargetFace identities into a synthetic composite TargetFace
        with spherical linear interpolated identity embedding.
        
        Args:
            target_a: Primary target face
            target_b: Secondary target face
            ratio: Blend weight where 0.0 is 100% target_a and 1.0 is 100% target_b
            
        Returns:
            Synthetic TargetFace with fused embedding and blended reference image.
        """
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if ratio <= 0.001:
            return target_a
        if ratio >= 0.999:
            return target_b

        cache_key = (target_a.target_id, target_b.target_id, round(ratio, 3))
        if self._last_fused_key == cache_key and self.cached_fused_target is not None:
            return self.cached_fused_target

        fused_emb = slerp(target_a.embedding, target_b.embedding, ratio)

        # Blend reference images for UI thumbnail display
        img_a = target_a.reference_image
        img_b = target_b.reference_image
        if img_a is not None and img_b is not None:
            ha, wa = img_a.shape[:2]
            hb, wb = img_b.shape[:2]
            if (ha, wa) != (hb, wb):
                import cv2
                img_b_matched = cv2.resize(img_b, (wa, ha), interpolation=cv2.INTER_LINEAR)
            else:
                img_b_matched = img_b
            fused_img = ((1.0 - ratio) * img_a.astype(np.float32) + ratio * img_b_matched.astype(np.float32)).astype(np.uint8)
        else:
            fused_img = img_a if img_a is not None else img_b

        pct_a = int(round((1.0 - ratio) * 100))
        pct_b = 100 - pct_a
        display_name = f"{target_a.display_name} ({pct_a}%) + {target_b.display_name} ({pct_b}%)"
        fused_id = f"fused_{target_a.target_id}_{target_b.target_id}_{pct_b}"

        meta = TargetMetadata(
            person_id=fused_id,
            display_name=display_name,
            category="fusion",
            source="synthetic_morph",
            notes=f"Dual-identity fusion: {target_a.display_name} ({1.0 - ratio:.2f}) and {target_b.display_name} ({ratio:.2f})",
        )

        fused_target = TargetFace(
            target_id=fused_id,
            display_name=display_name,
            category="fusion",
            reference_image_path="",
            reference_image=fused_img,
            embedding=fused_emb,
            metadata=meta,
        )

        self._last_fused_key = cache_key
        self.cached_fused_target = fused_target
        return fused_target

    def start_morph(
        self,
        source_target: TargetFace,
        dest_target: TargetFace,
        duration: Optional[float] = None,
    ) -> None:
        """
        Initiates a smooth temporal morph from source identity to destination identity.
        """
        if source_target is None or dest_target is None:
            return
        if source_target.target_id == dest_target.target_id:
            return

        self.source_target = source_target
        self.destination_target = dest_target
        self.morph_duration = max(0.05, float(duration or self.default_duration))
        self.morph_start_time = time.perf_counter()
        self.morph_progress = 0.0
        self.is_morphing = True
        logger.info(
            f"Started identity morph: '{source_target.display_name}' -> '{dest_target.display_name}' "
            f"over {self.morph_duration:.2f}s."
        )

    def cancel_morph(self) -> None:
        """Cancels any in-progress temporal morph."""
        self.is_morphing = False
        self.morph_progress = 0.0
        self.source_target = None
        self.destination_target = None

    def update(self) -> Optional[TargetFace]:
        """
        Advances morphing progress by current elapsed wall time.
        
        Returns:
            Current interpolated TargetFace during transition, destination when complete,
            or None if no morph is active.
        """
        if not self.is_morphing or self.source_target is None or self.destination_target is None:
            return None

        elapsed = time.perf_counter() - self.morph_start_time
        raw_t = elapsed / self.morph_duration

        if raw_t >= 1.0:
            # Transition complete
            self.morph_progress = 1.0
            self.is_morphing = False
            final_target = self.destination_target
            logger.info(f"Identity morph to '{final_target.display_name}' completed.")
            self.source_target = None
            self.destination_target = None
            return final_target

        self.morph_progress = raw_t
        eased_t = smoothstep(raw_t)
        return self.fuse_targets(self.source_target, self.destination_target, ratio=eased_t)

    def resolve_effective_target(
        self,
        primary_target: Optional[TargetFace],
        secondary_target: Optional[TargetFace] = None,
        fusion_ratio: float = 0.0,
    ) -> Optional[TargetFace]:
        """
        Resolves the final active TargetFace for the current frame, taking into
        account ongoing temporal morph transitions and static dual-target fusion ratio.
        """
        # 1. Check active temporal morph transition
        if self.is_morphing:
            morph_res = self.update()
            if morph_res is not None:
                return morph_res

        # 2. Check dual-target fusion ratio
        if primary_target is not None and secondary_target is not None and fusion_ratio > 0.001:
            return self.fuse_targets(primary_target, secondary_target, ratio=fusion_ratio)

        # 3. Default to primary target
        return primary_target

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns morphing status and telemetry metrics."""
        return {
            "is_morphing": self.is_morphing,
            "morph_progress": round(self.morph_progress, 3),
            "source_id": self.source_target.target_id if self.source_target else None,
            "source_name": self.source_target.display_name if self.source_target else None,
            "destination_id": self.destination_target.target_id if self.destination_target else None,
            "destination_name": self.destination_target.display_name if self.destination_target else None,
        }
