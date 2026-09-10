"""
Target Selector Widget with Category Filtering, Live Thumbnail Preview,
and Dynamic Custom Target Import Dialog.
"""

import os
from typing import Optional, List, Dict
import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QFrame,
    QDialog,
    QFileDialog,
    QMessageBox,
)

from src.targets.target_manager import TargetManager
from src.targets.target_loader import TargetFace
from src.utils.logger import get_logger

logger = get_logger("TargetSelectorWidget")


class AddTargetDialog(QDialog):
    """
    Modal dialog allowing users to import any reference face portrait into the target database.
    """

    target_added = pyqtSignal(str)  # Emits new person_id

    def __init__(self, target_manager: TargetManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Import New Target Face")
        self.resize(440, 480)
        self.target_manager = target_manager
        self._selected_image_path: Optional[str] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 1. Preview Box
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: #1a1a1a; border: 1px solid #3b82f6; border-radius: 6px;")
        p_layout = QVBoxLayout(preview_frame)
        p_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.preview_label = QLabel("No Image Selected\n(Click 'Browse Photo' below)")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedSize(160, 160)
        self.preview_label.setStyleSheet("color: #888; font-size: 11px;")
        p_layout.addWidget(self.preview_label)
        layout.addWidget(preview_frame)

        # 2. Browse Button
        self.browse_btn = QPushButton("📁 Browse Portrait Photo...")
        self.browse_btn.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px;")
        self.browse_btn.clicked.connect(self._on_browse_photo)
        layout.addWidget(self.browse_btn)

        # 3. Form Grid
        grid = QGridLayout()
        grid.addWidget(QLabel("Display Name:"), 0, 0)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Mahesh Babu or Jane Doe")
        grid.addWidget(self.name_edit, 0, 1)

        grid.addWidget(QLabel("Category:"), 1, 0)
        self.cat_combo = QComboBox()
        self.cat_combo.addItem("Telugu Heroes", "telugu_heroes")
        self.cat_combo.addItem("Actors", "actors")
        self.cat_combo.addItem("Actresses", "actresses")
        self.cat_combo.addItem("Custom", "custom")
        grid.addWidget(self.cat_combo, 1, 1)

        grid.addWidget(QLabel("Person ID:"), 2, 0)
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("(Optional slug, e.g., mahesh_babu)")
        grid.addWidget(self.id_edit, 2, 1)

        layout.addLayout(grid)

        # 4. Action Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        self.save_btn = QPushButton("✔ Import & Select Target")
        self.save_btn.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 8px;")
        self.save_btn.clicked.connect(self._on_save_target)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

    def _on_browse_photo(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Portrait Image",
            "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp)",
        )
        if file_path:
            self._selected_image_path = file_path
            bgr = cv2.imread(file_path)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.preview_label.setPixmap(pix)

                # Auto-populate display name from filename if empty
                if not self.name_edit.text():
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    clean_name = base_name.replace("_", " ").title()
                    self.name_edit.setText(clean_name)

    def _on_save_target(self) -> None:
        if not self._selected_image_path:
            QMessageBox.warning(self, "Missing Image", "Please select a reference portrait image first.")
            return

        display_name = self.name_edit.text().strip()
        if not display_name:
            QMessageBox.warning(self, "Missing Name", "Please enter a display name for the target.")
            return

        category = self.cat_combo.currentData() or "custom"
        person_id = self.id_edit.text().strip() or None

        success, msg, target = self.target_manager.add_custom_target(
            image_or_path=self._selected_image_path,
            display_name=display_name,
            category=category,
            person_id=person_id,
        )

        if success and target:
            QMessageBox.information(self, "Success", f"Target '{display_name}' imported successfully!")
            self.target_added.emit(target.target_id)
            self.accept()
        else:
            QMessageBox.critical(self, "Import Error", msg)


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
        self.cat_combo.addItem("Custom", "custom")
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
        self.add_btn = QPushButton("+ Add Target")
        self.add_btn.setStyleSheet("background-color: #059669; color: white; font-weight: bold;")
        self.add_btn.clicked.connect(self._open_add_target_dialog)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_selection)

        self.refresh_btn = QPushButton("Rescan DB")
        self.refresh_btn.clicked.connect(self.refresh_targets)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.refresh_btn)
        layout.addLayout(btn_layout)

    def _open_add_target_dialog(self) -> None:
        dialog = AddTargetDialog(self.target_manager, self)
        dialog.target_added.connect(self._on_custom_target_added)
        dialog.exec()

    def _on_custom_target_added(self, new_target_id: str) -> None:
        self.refresh_targets()
        idx = self.target_combo.findData(new_target_id)
        if idx >= 0:
            self.target_combo.setCurrentIndex(idx)

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
