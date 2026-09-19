"""
Frame Rendering and Color Space Conversion Engine.
Converts BGR OpenCV frames to RGB, Qt QImage, and letterboxed aspect-ratio viewports.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class FrameRenderer:
    """
    Handles image format transformations, color conversions, and aspect-ratio scaling.
    """

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Converts BGR numpy image to RGB."""
        if frame is None or frame.size == 0:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def letterbox_resize(
        frame: np.ndarray,
        target_size: Tuple[int, int],
        pad_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Scales frame to fit target (width, height) while preserving aspect ratio.
        Returns (letterboxed_frame, scale_factor, (pad_x, pad_y)).
        """
        h, w = frame.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        canvas = np.full((target_h, target_w, 3), pad_color, dtype=frame.dtype)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        return canvas, scale, (pad_x, pad_y)


__all__ = ["FrameRenderer"]
