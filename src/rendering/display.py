"""
Display Surface and Video Viewport Helpers.
"""

from typing import Tuple, Optional
import numpy as np
from src.rendering.renderer import FrameRenderer


class VideoDisplaySurface:
    """
    Manages display buffer allocation and format conversion for desktop UI widgets.
    """

    def __init__(self, default_resolution: Tuple[int, int] = (1280, 720)):
        self.resolution = default_resolution
        self.renderer = FrameRenderer()

    def prepare_for_display(
        self,
        frame: np.ndarray,
        viewport_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Converts BGR frame to RGB and optionally letterboxes to target viewport size."""
        if frame is None or frame.size == 0:
            w, h = viewport_size or self.resolution
            return np.zeros((h, w, 3), dtype=np.uint8)

        rgb = self.renderer.bgr_to_rgb(frame)
        if viewport_size is not None and (rgb.shape[1], rgb.shape[0]) != viewport_size:
            letterboxed, _, _ = self.renderer.letterbox_resize(rgb, viewport_size)
            return letterboxed
        return rgb


__all__ = ["VideoDisplaySurface"]
