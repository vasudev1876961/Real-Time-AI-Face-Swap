"""
Post-Processing and Visual Overlay Utilities.
"""

from typing import Optional, List, Tuple, Any
import cv2
import numpy as np

from src.detection.face_detector import FaceData
from src.utils.logger import get_logger

logger = get_logger("PostProcess")


def apply_unsharp_mask(image: np.ndarray, amount: float = 0.2) -> np.ndarray:
    """Applies subtle unsharp masking to enhance facial details."""
    if amount <= 0.0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def postprocess_frame(
    frame: np.ndarray,
    sharpen_amount: float = 0.0,
    **kwargs: Any,
) -> np.ndarray:
    """Applies final image enhancements to the blended video frame."""
    if frame is None or frame.size == 0:
        return frame
    amount = kwargs.get("unsharp_amount", sharpen_amount)
    if kwargs.get("apply_sharpening", False) or amount > 0.0:
        frame = apply_unsharp_mask(frame, amount)
    return frame


def draw_debug_overlay(
    frame: np.ndarray,
    face: Optional[FaceData] = None,
    fps: float = 0.0,
    latency_ms: float = 0.0,
    active_target_name: Optional[str] = None,
    provider_name: str = "CPU",
    model_ready: bool = False,
) -> np.ndarray:
    """Renders debug metrics directly onto frame surface."""
    if frame is None or frame.size == 0:
        return frame

    overlay = frame.copy()
    color = (0, 255, 0) if model_ready else (0, 165, 255)
    cv2.putText(overlay, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if face and face.bbox:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return overlay
