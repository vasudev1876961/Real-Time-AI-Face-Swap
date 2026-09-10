"""
High-Performance Qt Video Display Surface with Letterboxing, Comparison Modes,
and Interactive Split-Screen Wipe.
"""

from typing import Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QMouseEvent
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy

from src.utils.logger import get_logger

logger = get_logger("CameraWidget")


class CameraWidget(QWidget):
    """
    Renders processed BGR video frames into a responsive Qt viewport.
    Supports Normal, Side-by-Side, Split-Screen Wipe, and Difference Heatmap comparison modes.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #121212; border-radius: 8px;")
        self.setMouseTracking(True)

        self._pixmap: Optional[QPixmap] = None
        self._status_text: str = "Camera Inactive (Click 'Start Camera')"
        self._is_recording: bool = False
        self._fps_text: str = ""

        # Comparison View Modes
        self._display_mode: str = "normal"  # "normal", "side_by_side", "split_screen", "difference"
        self._split_ratio: float = 0.50
        self._is_dragging_split: bool = False
        self._last_rendered_frame: Optional[np.ndarray] = None
        self._last_original_frame: Optional[np.ndarray] = None
        self._last_fps: float = 0.0

    def set_display_mode(self, mode: str) -> None:
        """Sets the viewport comparison mode."""
        valid_modes = {"normal", "side_by_side", "split_screen", "difference"}
        if mode in valid_modes:
            self._display_mode = mode
            logger.info(f"Viewport display mode changed to '{mode}'")
            if self._last_rendered_frame is not None:
                self.update_dual_frames(
                    self._last_rendered_frame,
                    self._last_original_frame,
                    fps=self._last_fps,
                    is_recording=self._is_recording,
                )

    def get_display_mode(self) -> str:
        return self._display_mode

    def update_frame(self, bgr_frame: np.ndarray, fps: float = 0.0, is_recording: bool = False) -> None:
        """Compatibility wrapper for single frame update."""
        self.update_dual_frames(bgr_frame, None, fps=fps, is_recording=is_recording)

    def update_dual_frames(
        self,
        rendered_bgr: np.ndarray,
        original_bgr: Optional[np.ndarray] = None,
        fps: float = 0.0,
        is_recording: bool = False,
    ) -> None:
        """
        Renders frame according to the active display mode (Normal, Side-by-Side, Split-Screen, Heatmap).
        """
        if rendered_bgr is None or rendered_bgr.size == 0:
            return

        self._last_rendered_frame = rendered_bgr
        self._last_original_frame = original_bgr
        self._last_fps = fps
        self._is_recording = is_recording

        orig = original_bgr if original_bgr is not None else rendered_bgr
        display_frame = rendered_bgr

        if self._display_mode == "side_by_side":
            # Side-by-Side View
            h, w = rendered_bgr.shape[:2]
            half_w = max(100, w // 2)
            left_crop = cv2.resize(orig, (half_w, h))
            right_crop = cv2.resize(rendered_bgr, (half_w, h))

            # Composite side-by-side with dividing line
            display_frame = np.zeros((h, half_w * 2 + 4, 3), dtype=np.uint8)
            display_frame[:, :half_w] = left_crop
            display_frame[:, half_w : half_w + 4] = (59, 130, 246)  # Accent divider line
            display_frame[:, half_w + 4 :] = right_crop

            # Overlay labels
            cv2.putText(display_frame, "ORIGINAL", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display_frame, "ORIGINAL", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(display_frame, "SWAPPED", (half_w + 20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display_frame, "SWAPPED", (half_w + 20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 1)

        elif self._display_mode == "split_screen":
            # Interactive Split-Screen Wipe
            h, w = rendered_bgr.shape[:2]
            split_x = int(np.clip(self._split_ratio * w, 10, w - 10))

            display_frame = rendered_bgr.copy()
            display_frame[:, :split_x] = orig[:, :split_x]
            # Draw vertical divider line
            display_frame[:, max(0, split_x - 2) : min(w, split_x + 2)] = (59, 130, 246)

            # Labels
            cv2.putText(display_frame, "ORIGINAL", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display_frame, "ORIGINAL", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(display_frame, "SWAPPED", (w - 140, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display_frame, "SWAPPED", (w - 140, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 1)

        elif self._display_mode == "difference":
            # Absolute Difference Heatmap
            diff = cv2.absdiff(rendered_bgr, orig)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            norm_diff = cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_TURBO)
            display_frame = cv2.addWeighted(heatmap, 0.75, orig, 0.25, 0)
            cv2.putText(display_frame, "DIFFERENCE HEATMAP", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display_frame, "DIFFERENCE HEATMAP", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

        # Convert to QPixmap
        h, w, ch = display_frame.shape
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w

        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._status_text = ""
        self._fps_text = f"{fps:.1f} FPS" if fps > 0 else ""
        self.update()

    def set_placeholder_message(self, message: str) -> None:
        """Displays placeholder text when camera is inactive."""
        self._pixmap = None
        self._status_text = message
        self._fps_text = ""
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._display_mode == "split_screen" and event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging_split = True
            self._update_split_from_mouse(event.pos().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._display_mode == "split_screen" and (self._is_dragging_split or event.buttons() == Qt.MouseButton.LeftButton):
            self._update_split_from_mouse(event.pos().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging_split = False

    def _update_split_from_mouse(self, mouse_x: int) -> None:
        """Updates split ratio from mouse X coordinate."""
        widget_w = max(1, self.width())
        self._split_ratio = float(np.clip(mouse_x / widget_w, 0.05, 0.95))
        if self._last_rendered_frame is not None:
            self.update_dual_frames(
                self._last_rendered_frame,
                self._last_original_frame,
                fps=self._last_fps,
                is_recording=self._is_recording,
            )

    def paintEvent(self, event) -> None:
        """Paints video frame scaled to widget geometry while maintaining aspect ratio."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Fill dark background
        painter.fillRect(self.rect(), QColor("#121212"))

        if self._pixmap and not self._pixmap.isNull():
            # Scale pixmap maintaining aspect ratio
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            # Center within widget
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Draw Recording Indicator Badge if recording
            if self._is_recording:
                painter.setBrush(QColor(220, 20, 60, 220))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(x + 16, y + 16, 110, 28, 14, 14)
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                painter.drawText(QRect(x + 16, y + 16, 110, 28), Qt.AlignmentFlag.AlignCenter, "● REC")

        else:
            # Render placeholder message
            painter.setPen(QColor("#888888"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._status_text)
