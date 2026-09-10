"""
Face Enhancement Subsystem.
Includes AdaptiveFidelityEnhancer (zero-latency real-time detail restoration)
and FaceEnhancer (unified coordinator with ONNX model support).
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.models.face_enhancer import ONNXFaceEnhancer
from src.core.config_loader import SingleModelConfig
from src.utils.logger import get_logger

logger = get_logger("FaceEnhancement")


class AdaptiveFidelityEnhancer:
    """
    Real-Time Adaptive Facial Fidelity & Detail Restorer.
    Applies multi-scale frequency decomposition, edge-preserving bilateral skin smoothing,
    and landmark-guided ocular/oral detail boost at zero noticeable latency (< 2 ms).
    """

    def __init__(self, default_strength: float = 0.40):
        self.default_strength = default_strength

    def enhance(
        self,
        face_bgr: np.ndarray,
        strength: Optional[float] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Enhances face crop with edge-preserving skin smoothing and feature sharpening.
        Args:
            face_bgr: Swapped BGR face crop [H, W, 3], uint8.
            strength: Enhancement intensity [0.0, 1.0]. If None, uses default_strength.
            landmarks: Optional 5-point normalized or absolute landmarks [[x, y], ...].
        Returns:
            Enhanced BGR face crop [H, W, 3], uint8.
        """
        if face_bgr is None or face_bgr.size == 0:
            return face_bgr

        k_str = self.default_strength if strength is None else float(strength)
        if k_str <= 0.01:
            return face_bgr

        h, w = face_bgr.shape[:2]

        # 1. Edge-preserving bilateral filter on luminance channel (YCrCb)
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel, cr, cb = cv2.split(ycrcb)

        # Bilateral filter smooths flat skin regions while preserving sharp edges
        d = 5
        sigma_color = 25.0
        sigma_space = 25.0
        smooth_y = cv2.bilateralFilter(y_channel, d, sigma_color, sigma_space)

        # 2. Multi-scale Unsharp Masking for micro-texture and pore clarity
        # Base blur (fine scale)
        blur_fine = cv2.GaussianBlur(y_channel, (0, 0), sigmaX=1.0)
        detail_fine = cv2.subtract(y_channel, blur_fine)

        # Base blur (medium scale)
        blur_med = cv2.GaussianBlur(y_channel, (0, 0), sigmaX=2.5)
        detail_med = cv2.subtract(blur_fine, blur_med)

        # Combine fine & medium details with edge protection
        boosted_y = smooth_y.astype(np.float32) + (detail_fine.astype(np.float32) * (1.2 * k_str)) + (detail_med.astype(np.float32) * (0.6 * k_str))
        boosted_y = np.clip(boosted_y, 0, 255).astype(np.uint8)

        # 3. Landmark-Guided Ocular & Dental Clarity Boost
        if landmarks is not None and len(landmarks) >= 5:
            # Create feature weighting mask
            feature_mask = np.zeros((h, w), dtype=np.float32)
            pts = np.array(landmarks, dtype=np.float32)

            # Check if landmarks are normalized [0, 1] or pixel coordinates
            if pts.max() <= 1.05:
                pts[:, 0] *= w
                pts[:, 1] *= h

            # Left & right eye regions (landmarks 0 & 1)
            eye_radius = max(4, int(w * 0.12))
            for i in [0, 1]:
                cx, cy = int(pts[i, 0]), int(pts[i, 1])
                cv2.circle(feature_mask, (cx, cy), eye_radius, 1.0, -1)

            # Mouth region (landmarks 3 & 4)
            mx = int((pts[3, 0] + pts[4, 0]) / 2.0)
            my = int((pts[3, 1] + pts[4, 1]) / 2.0)
            mw = max(6, int(abs(pts[4, 0] - pts[3, 0]) * 0.7))
            mh = max(4, int(w * 0.10))
            cv2.ellipse(feature_mask, (mx, my), (mw, mh), 0, 0, 360, 1.0, -1)

            # Soften feature mask boundaries
            feature_mask = cv2.GaussianBlur(feature_mask, (15, 15), 5.0)

            # Extra sharpening on ocular & oral regions
            high_laplacian = cv2.Laplacian(y_channel, cv2.CV_32F, ksize=3)
            extra_sharpen = np.clip(y_channel.astype(np.float32) - (0.35 * k_str * high_laplacian), 0, 255)

            boosted_y = (boosted_y.astype(np.float32) * (1.0 - feature_mask * 0.5) + extra_sharpen * (feature_mask * 0.5))
            boosted_y = np.clip(boosted_y, 0, 255).astype(np.uint8)

        # Merge channels back
        enhanced_ycrcb = cv2.merge([boosted_y, cr, cb])
        enhanced_bgr = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)

        # Final blend with original crop based on strength
        return cv2.addWeighted(enhanced_bgr, k_str, face_bgr, 1.0 - k_str, 0.0)


class FaceEnhancer:
    """
    Unified Face Restoration & Enhancement Engine.
    Coordinates between optional ONNX neural models (GFPGAN / CodeFormer)
    and the zero-overhead AdaptiveFidelityEnhancer.
    """

    def __init__(
        self,
        model_config: Optional[SingleModelConfig] = None,
        default_strength: float = 0.40,
    ):
        self.model_config = model_config or SingleModelConfig(
            model_path="models/enhancement/face_enhancer.onnx",
            model_type="gfpgan_onnx",
        )
        self.onnx_enhancer = ONNXFaceEnhancer(self.model_config)
        self.adaptive_enhancer = AdaptiveFidelityEnhancer(default_strength)
        self.default_strength = default_strength

    def has_neural_model(self) -> bool:
        """Returns True if an ONNX face restoration model is loaded."""
        return self.onnx_enhancer.is_loaded()

    def enhance(
        self,
        face_bgr: np.ndarray,
        strength: Optional[float] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Applies face enhancement to the face crop.
        If ONNX model is available, uses neural restoration.
        Otherwise, applies real-time adaptive fidelity enhancement.
        """
        k_str = self.default_strength if strength is None else float(strength)
        if k_str <= 0.01:
            return face_bgr

        if self.onnx_enhancer.is_loaded():
            enhanced = self.onnx_enhancer.enhance(face_bgr, blend_weight=k_str)
            # Secondary micro-clarity touch
            return self.adaptive_enhancer.enhance(enhanced, strength=k_str * 0.3, landmarks=landmarks)
        else:
            return self.adaptive_enhancer.enhance(face_bgr, strength=k_str, landmarks=landmarks)
