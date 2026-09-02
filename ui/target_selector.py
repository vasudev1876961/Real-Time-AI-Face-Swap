"""
Target Selector Widget with Category Filtering and Live Thumbnail Preview.
"""

from typing import Optional, List, Dict
import cv2
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QGroupBox,
    QFrame,
)

from src.targets.target_manager import TargetManager
from src.targets.target_loader import TargetFace
from src.utils.logger import get_logger

logger = get_logger("TargetSelectorWidget")


class TargetSelectorWidget(QGroupBox):
    """
    GUI Component for filtering target categories, selecting a target identity,
    and previewing the reference portrait and metadata.
    """

    target_selected = pyqtSignal(str)  # Emits target_id or "" when deselected

    def __init__(self, target_manager: TargetManager, parent: Optional[QWidget] = None):
        super().__init__("Target Face Selection", parent)
        self.target_manager = target_manager
        self._current_category = "all"
        self._init_ui()
        self.refresh_targets()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Category Filter Row
        cat_layout = QHBoxLayout()
        cat_label = QLabel("Category:")
        cat_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.cat_combo = QComboBox()
        self.cat_combo.addItem("All Categories", "all")
        self.cat_combo.addItem("Actresses", "actresses")
        self.cat_combo.addItem("Actors", "actors")
        self.cat_combo.addItem("Telugu Heroes", "telugu_heroes")
        self.cat_combo.currentIndexChanged.connect(self._on_category_changed)
        cat_layout.addWidget(cat_label)
        cat_layout.addWidget(self.cat_combo, 1)
        layout.addLayout(cat_layout)

        # 2. Target Dropdown Row
        tgt_layout = QHBoxLayout()
        tgt_label = QLabel("Target:")
        tgt_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        tgt_layout.addWidget(tgt_label)
        tgt_layout.addWidget(self.target_combo, 1)
        layout.addLayout(tgt_layout)

        # 3. Thumbnail Preview Frame
        preview_container = QFrame()
        preview_container.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333333; border-radius: 6px;")
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.thumbnail_label = QLabel("No Target Selected")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setFixedSize(140, 140)
        self.thumbnail_label.setStyleSheet("color: #777; font-size: 11px;")
        preview_layout.addWidget(self.thumbnail_label)

        layout.addWidget(preview_container)

        # 4. Target Info Label
        self.info_label = QLabel("Identity: None")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self.info_label)

        # 5. Buttons Row
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.clicked.connect(self.clear_selection)
        self.refresh_btn = QPushButton("Rescan DB")
        self.refresh_btn.clicked.connect(self.refresh_targets)

        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.refresh_btn)
        layout.addLayout(btn_layout)

    def _on_category_changed(self, index: int) -> None:
        self._current_category = self.cat_combo.currentData() or "all"
        self._populate_target_dropdown()

    def _populate_target_dropdown(self) -> None:
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem("-- Select Target Face --", "")

        targets = self.target_manager.list_by_category(self._current_category)
        for t in targets:
            self.target_combo.addItem(f"{t.display_name} ({t.category})", t.target_id)

        self.target_combo.blockSignals(False)

        # Sync selection if active
        selected = self.target_manager.get_selected_target()
        if selected and (self._current_category == "all" or selected.category == self._current_category):
            idx = self.target_combo.findData(selected.target_id)
            if idx >= 0:
                self.target_combo.setCurrentIndex(idx)
        else:
            self._update_preview(None)

    def _on_target_changed(self, index: int) -> None:
        target_id = self.target_combo.currentData()
        if target_id:
            self.target_manager.select_target(target_id)
            target = self.target_manager.get_target(target_id)
            self._update_preview(target)
            self.target_selected.emit(target_id)
        else:
            self.clear_selection()

    def _update_preview(self, target: Optional[TargetFace]) -> None:
        if target is None or target.reference_image is None:
            self.thumbnail_label.setText("No Target Selected")
            self.thumbnail_label.setPixmap(QPixmap())
            self.info_label.setText("Identity: None")
            return

        # Render thumbnail
        rgb = cv2.cvtColor(target.reference_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            130, 130,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.thumbnail_label.setPixmap(pix)
        self.info_label.setText(f"ID: {target.target_id} | {target.category.title()}")

    def clear_selection(self) -> None:
        self.target_manager.select_target(None)
        self.target_combo.blockSignals(True)
        self.target_combo.setCurrentIndex(0)
        self.target_combo.blockSignals(False)
        self._update_preview(None)
        self.target_selected.emit("")

    def refresh_targets(self) -> None:
        self.target_manager.reload()
        self._populate_target_dropdown()
