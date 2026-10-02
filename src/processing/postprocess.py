"""
Post-Processing, Photorealistic Clarity Sharpening, and Live Frame Rate (FPS) Telemetry HUD.
Renders real-time stage latencies, rolling FPS, buffer recycling metrics, and mask contour wireframes.
"""

from typing import Optional, List, Tuple, Any, Dict
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


def draw_fps_telemetry_hud(
    frame: np.ndarray,
    fps: float = 0.0,
    latency_ms: float = 0.0,
    timings: Optional[Any] = None,
    governor_badge: str = "optimal",
    provider_name: str = "CPU",
    target_fps: float = 30.0,
    is_swapped: bool = True,
    active_target_name: Optional[str] = None,
) -> np.ndarray:
    """
    Renders a broadcast-quality, semi-transparent glassmorphic telemetry HUD
    in the upper corner showing rolling FPS, stage-by-stage latencies, and system health.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Glassmorphic background dimensions
    box_x = 16
    box_y = 16
    box_w = min(420, w - 32)
    box_h = 100

    # Draw dark translucent glass badge
    sub_img = overlay[box_y : box_y + box_h, box_x : box_x + box_w]
    black_rect = np.zeros_like(sub_img)
    glass = cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 0)
    overlay[box_y : box_y + box_h, box_x : box_x + box_w] = glass

    # Sleek cyber border
    border_color = (0, 255, 180) if fps >= target_fps * 0.9 else (0, 165, 255)
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), border_color, 1, cv2.LINE_AA)

    # Line 1: Header + FPS
    status_indicator = "[LIVE SWAP]" if is_swapped else "[PREVIEW]"
    fps_text = f"{fps:.1f} FPS"
    target_text = f"Target: {int(target_fps)}"
    title_text = f"{status_indicator} {fps_text} ({target_text}) | {provider_name}"
    cv2.putText(
        overlay,
        title_text,
        (box_x + 12, box_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        title_text,
        (box_x + 12, box_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        border_color,
        1,
        cv2.LINE_AA,
    )

    # Line 2: Latency & Governor
    gov_color = (0, 255, 120) if governor_badge == "optimal" else (0, 215, 255)
    lat_text = f"Latency: {latency_ms:.1f}ms | Load: {governor_badge.upper()}"
    if active_target_name:
        lat_text += f" | ID: {active_target_name}"
    cv2.putText(
        overlay,
        lat_text,
        (box_x + 12, box_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Line 3: Stage Breakdown Waterfall
    t_track = getattr(timings, "tracking_ms", 0.0) if timings else 0.0
    t_swap = getattr(timings, "swap_ms", 0.0) if timings else 0.0
    t_mask = getattr(timings, "mask_ms", 0.0) if timings else 0.0
    t_blend = getattr(timings, "blend_ms", 0.0) if timings else 0.0
    stage_text = f"Track: {t_track:.1f}ms | Swap: {t_swap:.1f}ms | Mask: {t_mask:.1f}ms | Blend: {t_blend:.1f}ms"
    cv2.putText(
        overlay,
        stage_text,
        (box_x + 12, box_y + 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 230, 255),
        1,
        cv2.LINE_AA,
    )

    return overlay


def draw_debug_overlay(
    frame: np.ndarray,
    face: Optional[FaceData] = None,
    fps: float = 0.0,
    latency_ms: float = 0.0,
    active_target_name: Optional[str] = None,
    provider_name: str = "CPU",
    model_ready: bool = False,
    timings: Optional[Any] = None,
    governor_badge: str = "optimal",
) -> np.ndarray:
    """Renders debug metrics directly onto frame surface."""
    if frame is None or frame.size == 0:
        return frame

    overlay = draw_fps_telemetry_hud(
        frame=frame,
        fps=fps,
        latency_ms=latency_ms,
        timings=timings,
        governor_badge=governor_badge,
        provider_name=provider_name,
        is_swapped=model_ready,
        active_target_name=active_target_name,
    )

    if face and face.bbox:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 180), 2, cv2.LINE_AA)

        # Draw 5 alignment landmarks
        if face.landmarks is not None and len(face.landmarks) >= 5:
            for pt in face.landmarks[:5]:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1, cv2.LINE_AA)

    return overlay


def postprocess_frame(
    frame: np.ndarray,
    sharpen_amount: float = 0.0,
    face_data: Optional[FaceData] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Applies final image enhancements, clarity sharpening, and optional on-frame FPS telemetry HUD.
    """
    if frame is None or frame.size == 0:
        return frame

    amount = kwargs.get("unsharp_amount", sharpen_amount)
    if kwargs.get("apply_sharpening", False):
        amount = max(amount, 0.3)

    if amount > 0.0:
        if face_data and face_data.bbox:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = face_data.bbox
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

    # Phase 13 On-Frame Live FPS & Telemetry HUD Overlay
    show_hud = kwargs.get("show_hud", False) or kwargs.get("show_fps_hud", False)
    if show_hud:
        fps_val = float(kwargs.get("fps", 0.0))
        lat_val = float(kwargs.get("latency_ms", 0.0))
        timings_obj = kwargs.get("timings", None)
        gov_badge = str(kwargs.get("governor", "optimal"))
        prov_name = str(kwargs.get("provider", "CPU"))
        is_swapped = bool(kwargs.get("is_swapped", True))
        t_name = kwargs.get("target_name", None)

        frame = draw_fps_telemetry_hud(
            frame=frame,
            fps=fps_val,
            latency_ms=lat_val,
            timings=timings_obj,
            governor_badge=gov_badge,
            provider_name=prov_name,
            is_swapped=is_swapped,
            active_target_name=t_name,
        )

    # Optional Mask Contour Wireframe HUD
    if kwargs.get("visualize_mask_hud", False) and face_data and face_data.landmarks is not None:
        if face_data.mesh_landmarks is not None and len(face_data.mesh_landmarks) > 400:
            from src.detection.face_detector import MEDIAPIPE_FACE_OVAL_INDICES
            from src.processing.mask_precision import draw_mask_contour_hud
            oval_pts = face_data.mesh_landmarks[MEDIAPIPE_FACE_OVAL_INDICES].astype(np.int32)
            frame = draw_mask_contour_hud(frame, oval_pts, color=(0, 255, 255), thickness=2)

    return frame
