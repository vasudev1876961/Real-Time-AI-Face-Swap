"""
Interactive Live Split-Screen & Comparison Renderer (Phase 15).
Provides vertical wipe comparison, dual side-by-side display, and pixel difference heatmap
for real-time before/after demonstration in browser, PyQt UI, and virtual camera streams.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("SplitScreenRenderer")


class SplitScreenRenderer:
    """
    Renders interactive before/after split-screen views onto video frames.
    """

    def __init__(self):
        pass

    @staticmethod
    def render(
        original_frame: np.ndarray,
        transformed_frame: np.ndarray,
        mode: str = "off",
        split_position: float = 0.50,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        Renders the split-screen view.

        Args:
            original_frame: Authentic source webcam frame.
            transformed_frame: Neural transformed / swapped frame.
            mode: Comparison mode ("off", "split_vertical", "side_by_side", "difference").
            split_position: Position of the vertical divider [0.05, 0.95].
            draw_labels: Whether to overlay sleek glassmorphic label badges.

        Returns:
            Composited frame.
        """
        if original_frame is None or transformed_frame is None:
            return transformed_frame if transformed_frame is not None else original_frame

        mode_clean = (mode or "off").strip().lower()
        if mode_clean in ("off", "none", ""):
            return transformed_frame

        h, w = transformed_frame.shape[:2]
        oh, ow = original_frame.shape[:2]
        if (oh, ow) != (h, w):
            orig_matched = cv2.resize(original_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            orig_matched = original_frame

        if mode_clean in ("split_vertical", "wipe", "vertical"):
            return SplitScreenRenderer._render_vertical_split(
                orig_matched, transformed_frame, split_position, draw_labels
            )
        elif mode_clean in ("side_by_side", "dual"):
            return SplitScreenRenderer._render_side_by_side(
                orig_matched, transformed_frame, draw_labels
            )
        elif mode_clean in ("difference", "diff", "heatmap"):
            return SplitScreenRenderer._render_difference_heatmap(
                orig_matched, transformed_frame, draw_labels
            )
        else:
            return transformed_frame

    @staticmethod
    def _render_vertical_split(
        original: np.ndarray,
        transformed: np.ndarray,
        split_pos: float,
        draw_labels: bool,
    ) -> np.ndarray:
        h, w = original.shape[:2]
        split_x = int(np.clip(split_pos, 0.05, 0.95) * w)

        # Composite: Left side original, Right side transformed
        result = transformed.copy()
        result[:, :split_x] = original[:, :split_x]

        # Draw glowing neon divider bar (Cyan-Purple gradient aesthetic)
        divider_w = 3
        x0 = max(0, split_x - divider_w // 2)
        x1 = min(w, split_x + divider_w // 2 + 1)
        
        # Soft outer glow
        glow_x0 = max(0, split_x - 4)
        glow_x1 = min(w, split_x + 5)
        glow_roi = result[:, glow_x0:glow_x1].astype(np.float32)
        glow_color = np.array([255, 200, 0], dtype=np.float32)  # Bright cyan/amber in BGR
        result[:, glow_x0:glow_x1] = np.clip(glow_roi * 0.7 + glow_color * 0.3, 0, 255).astype(np.uint8)

        # Crisp inner neon line
        result[:, x0:x1] = (255, 230, 0)  # Bright Cyan (BGR)

        # Handle circle grab handle in center of line
        cy = h // 2
        cv2.circle(result, (split_x, cy), 10, (20, 20, 25), -1)
        cv2.circle(result, (split_x, cy), 10, (255, 230, 0), 2)
        cv2.circle(result, (split_x, cy), 4, (255, 255, 255), -1)

        if draw_labels:
            # Sleek Glassmorphic Badges
            # Original Label (Left)
            SplitScreenRenderer._draw_badge(
                result, "ORIGINAL", (max(16, split_x - 110), 32), bg_color=(20, 20, 25), border_color=(100, 100, 100)
            )
            # AI Swap Label (Right)
            SplitScreenRenderer._draw_badge(
                result, "AI SWAP", (min(w - 110, split_x + 16), 32), bg_color=(25, 15, 40), border_color=(220, 50, 255)
            )

        return result

    @staticmethod
    def _render_side_by_side(
        original: np.ndarray,
        transformed: np.ndarray,
        draw_labels: bool,
    ) -> np.ndarray:
        h, w = original.shape[:2]
        half_w = w // 2

        # Resize both to half width
        orig_half = cv2.resize(original, (half_w, h), interpolation=cv2.INTER_AREA)
        trans_half = cv2.resize(transformed, (w - half_w, h), interpolation=cv2.INTER_AREA)

        result = np.hstack([orig_half, trans_half])

        # Center line divider
        cv2.line(result, (half_w, 0), (half_w, h), (255, 230, 0), 2)

        if draw_labels:
            SplitScreenRenderer._draw_badge(result, "ORIGINAL", (16, 32), bg_color=(20, 20, 25), border_color=(100, 100, 100))
            SplitScreenRenderer._draw_badge(result, "AI SWAP", (half_w + 16, 32), bg_color=(25, 15, 40), border_color=(220, 50, 255))

        return result

    @staticmethod
    def _render_difference_heatmap(
        original: np.ndarray,
        transformed: np.ndarray,
        draw_labels: bool,
    ) -> np.ndarray:
        h, w = original.shape[:2]
        diff = cv2.absdiff(original, transformed)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Amplify difference for high visibility
        diff_boosted = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff_boosted, cv2.COLORMAP_TURBO)

        # Blend 60% heatmap + 40% transformed frame
        result = cv2.addWeighted(transformed, 0.40, heatmap, 0.60, 0)

        if draw_labels:
            SplitScreenRenderer._draw_badge(
                result, "DELTA HEATMAP", (16, 32), bg_color=(20, 10, 30), border_color=(0, 180, 255)
            )

        return result

    @staticmethod
    def _draw_badge(
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        bg_color: Tuple[int, int, int] = (20, 20, 25),
        border_color: Tuple[int, int, int] = (150, 150, 150),
    ) -> None:
        """Draws a compact modern pill badge overlay with anti-aliased text."""
        x, y = pos
        pad_x, pad_y = 10, 5
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.45
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        bx0 = max(0, x)
        by0 = max(0, y - th - pad_y)
        bx1 = min(img.shape[1], bx0 + tw + 2 * pad_x)
        by1 = min(img.shape[0], by0 + th + 2 * pad_y)

        # Semi-transparent glassmorphic box
        roi = img[by0:by1, bx0:bx1]
        if roi.size > 0:
            bg_rect = np.full(roi.shape, bg_color, dtype=np.uint8)
            blended = cv2.addWeighted(roi, 0.35, bg_rect, 0.65, 0)
            img[by0:by1, bx0:bx1] = blended
            cv2.rectangle(img, (bx0, by0), (bx1, by1), border_color, 1, lineType=cv2.LINE_AA)

        cv2.putText(
            img,
            text,
            (bx0 + pad_x, by0 + th + pad_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
