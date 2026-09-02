"""
High-Performance Qt Video Display Surface with Letterboxing and Status Overlays.
"""

from typing import Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy

from src.utils.logger import get_logger

logger = get_logger("CameraWidget")


class CameraWidget(QWidget):
    """
    Renders processed BGR video frames into a responsive Qt viewport.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #121212; border-radius: 8px;")

        self._pixmap: Optional[QPixmap] = None
        self._status_text: str = "Camera Inactive (Click 'Start Camera')"
        self._is_recording: bool = False
        self._fps_text: str = ""

    def update_frame(self, bgr_frame: np.ndarray, fps: float = 0.0, is_recording: bool = False) -> None:
        """Converts BGR numpy image to QPixmap and triggers repaint."""
        if bgr_frame is None or bgr_frame.size == 0:
            return

        h, w, ch = bgr_frame.shape
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w

        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._status_text = ""
        self._fps_text = f"{fps:.1f} FPS" if fps > 0 else ""
        self._is_recording = is_recording
        self.update()

    def set_placeholder_message(self, message: str) -> None:
        """Displays informative placeholder text when video feed is stopped."""
        self._pixmap = None
        self._status_text = message
        self._fps_text = ""
        self.update()

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
