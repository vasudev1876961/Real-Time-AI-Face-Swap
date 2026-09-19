"""
Status and Telemetry Dashboard Panel for Live System Health and Pipeline Metrics.
"""

from typing import Dict, Any, Optional
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QGroupBox,
    QFrame,
)

from src.utils.logger import get_logger

logger = get_logger("StatusPanelWidget")


class StatusPanelWidget(QGroupBox):
    """
    Renders live telemetry, FPS counters, latency breakdowns, and device stats.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("System & Pipeline Status", parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 16, 12, 12)

        # Labels & Value fields
        self.val_fps = self._create_metric_label("0.0", "#4CAF50", bold=True, size=13)
        self.val_latency = self._create_metric_label("0.0 ms", "#00BCD4")
        self.val_face = self._create_metric_label("NO", "#F44336", bold=True)
        self.val_ai = self._create_metric_label("IDLE", "#FF9800")
        self.val_governor = self._create_metric_label("OPTIMAL", "#4CAF50", bold=True)
        self.val_gpu = self._create_metric_label("DETECTING...", "#9C27B0")
        self.val_system = self._create_metric_label("CPU: 0% | RAM: 0%", "#B0BEC5", size=9)

        # Grid Layout
        row = 0
        layout.addWidget(QLabel("Live Frame Rate:"), row, 0)
        layout.addWidget(self.val_fps, row, 1)

        row += 1
        layout.addWidget(QLabel("Pipeline Latency:"), row, 0)
        layout.addWidget(self.val_latency, row, 1)

        row += 1
        layout.addWidget(QLabel("Face Detected:"), row, 0)
        layout.addWidget(self.val_face, row, 1)

        row += 1
        layout.addWidget(QLabel("AI Swapper:"), row, 0)
        layout.addWidget(self.val_ai, row, 1)

        row += 1
        layout.addWidget(QLabel("Performance Mode:"), row, 0)
        layout.addWidget(self.val_governor, row, 1)

        row += 1
        layout.addWidget(QLabel("Hardware Device:"), row, 0)
        layout.addWidget(self.val_gpu, row, 1)

        row += 1
        layout.addWidget(QLabel("System Load:"), row, 0)
        layout.addWidget(self.val_system, row, 1)

    def _create_metric_label(self, default_text: str, color: str, bold: bool = False, size: int = 10) -> QLabel:
        lbl = QLabel(default_text)
        font = QFont("Segoe UI", size)
        if bold:
            font.setBold(True)
        lbl.setFont(font)
        lbl.setStyleSheet(f"color: {color};")
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    def update_metrics(self, summary: Dict[str, Any]) -> None:
        """Updates UI telemetry fields with latest metrics dictionary."""
        fps = summary.get("rolling_fps", summary.get("fps", 0.0))
        self.val_fps.setText(f"{fps:.1f} FPS")

        lat = summary.get("total_latency_ms", 0.0)
        timings = summary.get("timings", {})
        jitter = summary.get("jitter_ms", 0.0)
        percentiles = summary.get("percentiles", {})
        p95 = percentiles.get("p95", 0.0)
        p99 = percentiles.get("p99", 0.0)

        tooltip = (
            f"Latency Breakdown:\n"
            f"  Detect: {timings.get('detect', 0)} ms\n"
            f"  Track: {timings.get('track', 0)} ms\n"
            f"  Align: {timings.get('align', 0)} ms\n"
            f"  Swap: {timings.get('swap', 0)} ms\n"
            f"  Blend: {timings.get('blend', 0)} ms\n"
            f"  Total: {timings.get('total', 0)} ms\n"
            f"Jitter: {jitter:.1f} ms | P95: {p95:.1f} ms | P99: {p99:.1f} ms"
        )
        self.val_latency.setText(f"{lat:.1f} ms")
        self.val_latency.setToolTip(tooltip)

        # Face detected
        face_det = summary.get("face_detected", "NO")
        self.val_face.setText(face_det)
        self.val_face.setStyleSheet("color: #4CAF50; font-weight: bold;" if face_det == "YES" else "color: #F44336; font-weight: bold;")

        # AI Status
        ai_stat = summary.get("ai_status", "IDLE")
        self.val_ai.setText(ai_stat)
        if ai_stat == "RUNNING":
            self.val_ai.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif "MODEL NOT FOUND" in ai_stat:
            self.val_ai.setStyleSheet("color: #FFB300; font-weight: bold;")
        else:
            self.val_ai.setStyleSheet("color: #9E9E9E;")

        # Governor State
        gov = summary.get("governor")
        if gov and isinstance(gov, dict):
            badge = gov.get("badge_text", "OPTIMAL")
            color = gov.get("color", "#4CAF50")
            self.val_governor.setText(badge)
            self.val_governor.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Hardware provider and VRAM
        provider = summary.get("provider", "CPU")
        is_gpu = summary.get("gpu_active", "NO")
        vram = summary.get("vram", {})
        if vram and vram.get("available", False) and vram.get("total_mb", 0) > 0:
            used_mb = vram.get("used_mb", 0)
            tot_mb = vram.get("total_mb", 0)
            self.val_gpu.setText(f"{provider} | {used_mb:.0f}/{tot_mb:.0f} MB")
        else:
            self.val_gpu.setText(f"{provider} ({'GPU' if 'YES' in is_gpu else 'CPU'})")

        # System resources
        cpu_pct = summary.get("cpu_percent", 0.0)
        ram_pct = summary.get("ram_percent", 0.0)
        self.val_system.setText(f"CPU: {cpu_pct:.0f}% | RAM: {ram_pct:.0f}%")
