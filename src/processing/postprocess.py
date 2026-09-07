"""
Post-Processing and Visual Overlay Utilities.
"""

from typing import Optional, List, Tuple, Any
import cv2
import numpy as np

from src.detection.face_detector import FaceData
from src.utils.logger import get_logger

logger = get_logger("PostProcess")


def apply_unsharp_mask(image: np.ndarray, amount: float = 0.3, sigma: float = 1.2) -> np.ndarray:
    """Applies high-frequency unsharp masking to restore crisp facial details."""
    if amount <= 0.0 or image is None or image.size == 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def postprocess_frame(
    frame: np.ndarray,
    sharpen_amount: float = 0.0,
    face_data: Optional[FaceData] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Applies final image enhancements and clarity sharpening to the video frame."""
    if frame is None or frame.size == 0:
        return frame

    amount = kwargs.get("unsharp_amount", sharpen_amount)
    if kwargs.get("apply_sharpening", False):
        amount = max(amount, 0.3)

    if amount > 0.0:
        # If face bounding box is available, sharpen the face region with a soft transition
        if face_data and face_data.bbox:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = face_data.bbox
            # Add 25% margin around face
            bw, bh = x2 - x1, y2 - y1
            margin_x, margin_y = int(bw * 0.25), int(bh * 0.25)
            rx1 = max(0, x1 - margin_x)
            ry1 = max(0, y1 - margin_y)
            rx2 = min(w, x2 + margin_x)
            ry2 = min(h, y2 + margin_y)

            face_roi = frame[ry1:ry2, rx1:rx2]
            sharpened_roi = apply_unsharp_mask(face_roi, amount=amount, sigma=1.2)
            frame = frame.copy()
            frame[ry1:ry2, rx1:rx2] = sharpened_roi
        else:
            frame = apply_unsharp_mask(frame, amount=min(amount, 0.4), sigma=1.2)

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
