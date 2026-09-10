"""
PyQt6 Main Application Window for Real-Time AI Face Swap Camera.
Supports Live Camera, Video File Processing, Viewport Comparison Modes,
Face Enhancement Tuning, and Asynchronous Recording.
"""

import os
import json
import time
from datetime import datetime
from typing import Optional, List
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QIcon, QAction
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QSlider,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QStatusBar,
    QFrame,
    QSplitter,
)

from ui.camera_widget import CameraWidget
from ui.target_selector import TargetSelectorWidget
from ui.status_panel import StatusPanelWidget

from src.camera.camera_manager import CameraManager, CameraDeviceInfo
from src.pipeline.realtime_pipeline import RealTimePipeline, PipelineResult
from src.pipeline.video_processor import VideoFileProcessor
from src.recording.video_recorder import ThreadedVideoRecorder
from src.recording.capture import SnapshotCaptureManager
from src.core.config_loader import AppConfig, ModelsConfig, TargetsConfig, load_all_configs
from src.utils.logger import get_logger

logger = get_logger("MainWindow")


class MainWindow(QMainWindow):
    """
    Main Application Window integrating the live camera feed, target selection,
    face enhancement controls, viewport comparison modes, and asynchronous video recording.
    """

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        models_config: Optional[ModelsConfig] = None,
        targets_config: Optional[TargetsConfig] = None,
    ):
        super().__init__()
        self.setWindowTitle("Real-Time AI Face Swap Camera")
        self.resize(1280, 820)
        self.setMinimumSize(960, 600)

        # Load configurations
        a_cfg, m_cfg, t_cfg = load_all_configs()
        self.app_config = app_config or a_cfg
        self.models_config = models_config or m_cfg
        self.targets_config = targets_config or t_cfg

        # Initialize core subsystems
        self.camera_manager = CameraManager(self.app_config.camera)
        self.pipeline = RealTimePipeline(self.app_config, self.models_config, self.targets_config)
        self.camera_manager.register_switch_callback(lambda idx: self.pipeline.reset_tracker())

        # High-throughput asynchronous recording & snapshot engines
        self.video_recorder = ThreadedVideoRecorder()
        captures_dir = self.app_config.storage.get("captures_dir", "outputs/captures")
        self.capture_manager = SnapshotCaptureManager(captures_dir)

        # Timer for frame acquisition loop (~60 FPS display polling)
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._on_render_tick)

        self._init_ui()
        self._apply_dark_theme()
        self._populate_camera_list()

    def _init_ui(self) -> None:
        """Constructs the comprehensive user interface layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(8)

        # 1. Header & Disclaimer Banner
        disclaimer_frame = QFrame()
        disclaimer_frame.setStyleSheet("background-color: #1e293b; border-left: 4px solid #3b82f6; border-radius: 4px; padding: 4px;")
        disc_layout = QHBoxLayout(disclaimer_frame)
        disc_layout.setContentsMargins(8, 4, 8, 4)

        disc_label = QLabel("⚠️ AI-GENERATED FACE TRANSFORMATION — Use only with authorized/consenting subjects and permitted target assets.")
        disc_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        disc_label.setStyleSheet("color: #93c5fd;")
        disc_layout.addWidget(disc_label)
        main_layout.addWidget(disclaimer_frame)

        # 2. Main Work Area (Splitter: Viewport Left, Controls/Status Right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Container (Video + Viewport Mode Bar + Camera Controls)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.camera_widget = CameraWidget()
        left_layout.addWidget(self.camera_widget, 1)

        # 3. Viewport Comparison Toolbar
        mode_bar = QFrame()
        mode_bar.setStyleSheet("background-color: #1a202c; border-radius: 6px; padding: 3px;")
        mode_layout = QHBoxLayout(mode_bar)
        mode_layout.setContentsMargins(6, 4, 6, 4)
        mode_layout.setSpacing(6)

        mode_lbl = QLabel("Viewport:")
        mode_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        mode_lbl.setStyleSheet("color: #94a3b8;")
        mode_layout.addWidget(mode_lbl)

        self.btn_view_normal = QPushButton("Normal")
        self.btn_view_normal.setCheckable(True)
        self.btn_view_normal.setChecked(True)

        self.btn_view_split = QPushButton("Split Screen")
        self.btn_view_split.setCheckable(True)

        self.btn_view_side = QPushButton("Side-by-Side")
        self.btn_view_side.setCheckable(True)

        self.btn_view_diff = QPushButton("Diff Heatmap")
        self.btn_view_diff.setCheckable(True)

        for btn in [self.btn_view_normal, self.btn_view_split, self.btn_view_side, self.btn_view_diff]:
            btn.setStyleSheet("""
                QPushButton { background-color: #334155; padding: 4px 10px; border-radius: 4px; font-size: 11px; }
                QPushButton:checked { background-color: #2563eb; font-weight: bold; color: white; }
            """)

        self.btn_view_normal.clicked.connect(lambda: self._set_viewport_mode("normal"))
        self.btn_view_split.clicked.connect(lambda: self._set_viewport_mode("split_screen"))
        self.btn_view_side.clicked.connect(lambda: self._set_viewport_mode("side_by_side"))
        self.btn_view_diff.clicked.connect(lambda: self._set_viewport_mode("difference"))

        mode_layout.addWidget(self.btn_view_normal)
        mode_layout.addWidget(self.btn_view_split)
        mode_layout.addWidget(self.btn_view_side)
        mode_layout.addWidget(self.btn_view_diff)
        mode_layout.addStretch(1)

        self.process_file_btn = QPushButton("🎬 Process Video File...")
        self.process_file_btn.setStyleSheet("background-color: #d97706; color: white; font-weight: bold; padding: 4px 12px; border-radius: 4px; font-size: 11px;")
        self.process_file_btn.clicked.connect(self._open_process_video_dialog)
        mode_layout.addWidget(self.process_file_btn)

        left_layout.addWidget(mode_bar)

        # 4. Camera & Capture Action Bar
        action_bar = QFrame()
        action_bar.setStyleSheet("background-color: #1e1e1e; border-radius: 8px; padding: 6px;")
        act_layout = QHBoxLayout(action_bar)
        act_layout.setSpacing(10)

        self.start_btn = QPushButton("▶ Start Camera")
        self.start_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 8px 16px; border-radius: 5px;")
        self.start_btn.clicked.connect(self.start_camera)

        self.stop_btn = QPushButton("⏹ Stop Camera")
        self.stop_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 8px 16px; border-radius: 5px;")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)

        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(130)

        self.switch_cam_btn = QPushButton("Switch Device")
        self.switch_cam_btn.clicked.connect(self._on_switch_camera_clicked)

        self.capture_btn = QPushButton("📷 Capture")
        self.capture_btn.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 8px 14px; border-radius: 5px;")
        self.capture_btn.clicked.connect(self.capture_screenshot)

        self.record_btn = QPushButton("⏺ Record Video")
        self.record_btn.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; padding: 8px 14px; border-radius: 5px;")
        self.record_btn.clicked.connect(self.toggle_recording)

        act_layout.addWidget(self.start_btn)
        act_layout.addWidget(self.stop_btn)
        act_layout.addWidget(QLabel("Camera:"))
        act_layout.addWidget(self.camera_combo)
        act_layout.addWidget(self.switch_cam_btn)
        act_layout.addStretch(1)
        act_layout.addWidget(self.capture_btn)
        act_layout.addWidget(self.record_btn)

        left_layout.addWidget(action_bar)
        splitter.addWidget(left_container)

        # Right Container (Sidebar: Target Selector, Status, Tuning Controls)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Target Selector
        self.target_selector = TargetSelectorWidget(self.pipeline.target_manager)
        right_layout.addWidget(self.target_selector)

        # Live Status Panel
        self.status_panel = StatusPanelWidget()
        right_layout.addWidget(self.status_panel)

        # Processing & Performance Tuning Group
        tuning_group = QGroupBox("Live Pipeline Adjustments")
        tune_layout = QGridLayout(tuning_group)
        tune_layout.setSpacing(8)

        # Mask Type Dropdown
        tune_layout.addWidget(QLabel("Mask Type:"), 0, 0)
        self.mask_combo = QComboBox()
        self.mask_combo.addItem("Smooth Anatomical (Default)", "smooth_hull")
        self.mask_combo.addItem("Distance Transform (Zero Halo)", "distance_transform")
        self.mask_combo.addItem("Pose-Adaptive", "pose_adaptive")
        self.mask_combo.addItem("Standard Convex Hull", "convex_hull")
        self.mask_combo.addItem("Classic Elliptical", "elliptical")
        cur_mask = getattr(self.app_config.processing, "mask_type", "smooth_hull")
        for i in range(self.mask_combo.count()):
            if self.mask_combo.itemData(i) == cur_mask:
                self.mask_combo.setCurrentIndex(i)
                break
        self.mask_combo.currentIndexChanged.connect(self._on_mask_type_changed)
        tune_layout.addWidget(self.mask_combo, 0, 1)

        # Mask Feather Slider
        tune_layout.addWidget(QLabel("Mask Feather:"), 1, 0)
        self.feather_slider = QSlider(Qt.Orientation.Horizontal)
        self.feather_slider.setRange(1, 20)
        self.feather_slider.setValue(int(self.app_config.processing.mask_feather * 10))
        self.feather_slider.valueChanged.connect(self._on_feather_changed)
        tune_layout.addWidget(self.feather_slider, 1, 1)

        # Color Correction Mode
        tune_layout.addWidget(QLabel("Color Match:"), 2, 0)
        self.color_combo = QComboBox()
        self.color_combo.addItem("Reinhard Lab Transfer", "reinhard")
        self.color_combo.addItem("Gain Matching", "gain_matching")
        self.color_combo.addItem("Histogram Match", "histogram")
        self.color_combo.addItem("None (Raw)", "none")
        cur_color = getattr(self.app_config.processing, "color_correction", "reinhard")
        for i in range(self.color_combo.count()):
            if self.color_combo.itemData(i) == cur_color:
                self.color_combo.setCurrentIndex(i)
                break
        self.color_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        tune_layout.addWidget(self.color_combo, 2, 1)

        # Face Enhancement Strength Slider
        tune_layout.addWidget(QLabel("Face Enhance:"), 3, 0)
        self.enhance_slider = QSlider(Qt.Orientation.Horizontal)
        self.enhance_slider.setRange(0, 100)
        init_enh = int(getattr(self.app_config.processing, "enhancement_strength", 0.40) * 100)
        self.enhance_slider.setValue(init_enh)
        self.enhance_slider.valueChanged.connect(self._on_enhancement_changed)
        tune_layout.addWidget(self.enhance_slider, 3, 1)

        # Performance Profile Mode
        tune_layout.addWidget(QLabel("Profile:"), 4, 0)
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Quality Mode (30 FPS Target)", "quality")
        self.profile_combo.addItem("Performance Mode (Max Speed)", "performance")
        self.profile_combo.addItem("Debug Mode (Telemetry Overlay)", "debug")
        cur_prof = getattr(self.app_config.performance, "mode", "quality")
        for i in range(self.profile_combo.count()):
            if self.profile_combo.itemData(i) == cur_prof:
                self.profile_combo.setCurrentIndex(i)
                break
        self.profile_combo.currentIndexChanged.connect(self._on_profile_mode_changed)
        tune_layout.addWidget(self.profile_combo, 4, 1)

        right_layout.addWidget(tuning_group)
        right_layout.addStretch(1)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select target and click 'Start Camera'.")

    def _set_viewport_mode(self, mode: str) -> None:
        """Switches the viewport comparison mode and updates button states."""
        self.camera_widget.set_display_mode(mode)
        self.btn_view_normal.setChecked(mode == "normal")
        self.btn_view_split.setChecked(mode == "split_screen")
        self.btn_view_side.setChecked(mode == "side_by_side")
        self.btn_view_diff.setChecked(mode == "difference")

    def _open_process_video_dialog(self) -> None:
        """Opens file dialog and runs offline batch video processing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File to Swap",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if not file_path:
            return

        target = self.pipeline.get_selected_target()
        target_name = target.display_name if target else "No Target (Preview Only)"
        reply = QMessageBox.question(
            self,
            "Confirm Video Processing",
            f"Process video '{os.path.basename(file_path)}' using target '{target_name}'?\n\n"
            f"Processed output will be saved into outputs/recordings/.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        rec_dir = self.app_config.storage.get("recordings_dir", "outputs/recordings")
        out_path = os.path.join(rec_dir, f"swapped_{os.path.basename(file_path)}")
        processor = VideoFileProcessor(self.pipeline)
        self.status_bar.showMessage(f"Processing video: {os.path.basename(file_path)}...")

        prog = QProgressDialog("Processing video frames...", "Cancel", 0, 100, self)
        prog.setWindowTitle("Offline Video Face Swapper")
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.show()

        def on_prog(curr, total, fps, eta):
            if prog.wasCanceled():
                return
            pct = int((curr / max(1, total)) * 100)
            prog.setValue(pct)
            prog.setLabelText(f"Frame {curr}/{total} ({pct}%) | {fps:.1f} FPS | ETA: {eta:.1f}s")

        try:
            summary = processor.process_video(
                file_path,
                out_path,
                target_id=target.target_id if target else None,
                progress_callback=on_prog,
            )
            prog.close()
            QMessageBox.information(
                self,
                "Processing Complete",
                f"Video successfully processed and saved to:\n{out_path}\n\n"
                f"Total Frames: {summary['total_frames']} ({summary['swapped_frames']} swapped)\n"
                f"Average FPS: {summary['fps_processed']}\n"
                f"Audio Preserved: {summary['audio_preserved']}",
            )
            self.status_bar.showMessage(f"Video saved: {os.path.basename(out_path)}")
        except Exception as e:
            prog.close()
            QMessageBox.critical(self, "Processing Error", f"Failed to process video: {e}")

    def _apply_dark_theme(self) -> None:
        """Applies modern dark style stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
                color: #e2e8f0;
            }
            QWidget {
                color: #e2e8f0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 14px;
                font-weight: bold;
                font-size: 11px;
                color: #94a3b8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QComboBox {
                background-color: #1e293b;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
                color: #f8fafc;
            }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                selection-background-color: #3b82f6;
                color: #f8fafc;
            }
            QPushButton {
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 6px 12px;
                color: #f8fafc;
            }
            QPushButton:hover {
                background-color: #475569;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border-color: #334155;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #334155;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
            QStatusBar {
                background-color: #0f172a;
                color: #94a3b8;
                border-top: 1px solid #1e293b;
            }
        """)

    def _populate_camera_list(self) -> None:
        """Probes and populates available video capture devices."""
        cams = self.camera_manager.discover_cameras(max_probe=3)
        self.camera_combo.clear()
        if cams:
            for cam in cams:
                self.camera_combo.addItem(cam.name, cam.index)
        else:
            self.camera_combo.addItem("Default Camera (0)", 0)

    @pyqtSlot()
    def start_camera(self) -> None:
        """Starts video acquisition from selected camera index."""
        selected_idx = self.camera_combo.currentData() or 0
        success = self.camera_manager.start(selected_idx)
        if success:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.camera_combo.setEnabled(False)
            self.switch_cam_btn.setEnabled(True)
            self._render_timer.start(16)
            self.status_bar.showMessage(f"Camera active (Index {selected_idx}).")
        else:
            QMessageBox.warning(
                self,
                "Camera Error",
                f"Could not open camera device {selected_idx}. Check camera permissions or connection.",
            )

    @pyqtSlot()
    def stop_camera(self) -> None:
        """Stops video capture."""
        if self.video_recorder.is_recording():
            self.toggle_recording()

        self._render_timer.stop()
        self.camera_manager.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.camera_widget.set_placeholder_message("Camera Stopped. Click 'Start Camera' to resume.")
        self.status_bar.showMessage("Camera stopped.")

    @pyqtSlot()
    def _on_switch_camera_clicked(self) -> None:
        """Switches to selected camera without closing app."""
        new_index = self.camera_combo.currentData() or 0
        if self.video_recorder.is_recording():
            self.toggle_recording()

        success = self.camera_manager.switch_camera(new_index)
        if success:
            self.status_bar.showMessage(f"Switched to camera index {new_index}.")
        else:
            self.status_bar.showMessage(f"Failed to switch to camera {new_index}.")

    def _on_render_tick(self) -> None:
        """Processes and renders the newest available camera frame."""
        packet = self.camera_manager.get_latest_frame()
        if packet is None or packet.frame is None:
            return

        # Execute real-time pipeline pass
        result: PipelineResult = self.pipeline.process_frame(packet.frame)

        # Update Video Display with both rendered & original frames for comparison modes
        self.camera_widget.update_dual_frames(
            rendered_bgr=result.rendered_frame,
            original_bgr=result.original_frame,
            fps=result.metrics_summary.get("fps", 0.0),
            is_recording=self.video_recorder.is_recording(),
        )

        # Update Telemetry Panel
        self.status_panel.update_metrics(result.metrics_summary)
        self.status_bar.showMessage(f"Status: {result.status_message}")

        # Non-blocking threaded video recording write
        if self.video_recorder.is_recording():
            self.video_recorder.write_frame(result.rendered_frame)

    @pyqtSlot()
    def capture_screenshot(self) -> None:
        """Saves current processed frame with metadata."""
        packet = self.camera_manager.get_latest_frame()
        if packet is None or packet.frame is None:
            self.status_bar.showMessage("Cannot capture: No active camera feed.")
            return

        result = self.pipeline.process_frame(packet.frame)
        target = result.target
        target_name = target.display_name if target else None
        target_id = target.target_id if target else None
        cat = target.category if target else None

        saved_path = self.capture_manager.capture(
            result.rendered_frame,
            target_name=target_name,
            target_id=target_id,
            category=cat,
        )
        if saved_path:
            self.status_bar.showMessage(f"Screenshot saved: {os.path.basename(saved_path)}")
        else:
            self.status_bar.showMessage("Failed to save screenshot.")

    @pyqtSlot()
    def toggle_recording(self) -> None:
        """Starts or stops non-blocking asynchronous video recording."""
        if not self.video_recorder.is_recording():
            # Start recording
            packet = self.camera_manager.get_latest_frame()
            if packet is None or packet.frame is None:
                self.status_bar.showMessage("Cannot record: Camera is not running.")
                return

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            target = self.pipeline.get_selected_target()
            target_id = target.target_id if target else "notarget"
            rec_dir = self.app_config.storage.get("recordings_dir", "outputs/recordings")
            filepath = os.path.join(rec_dir, f"{timestamp}_{target_id}.mp4")

            h, w = packet.frame.shape[:2]
            fps = max(15.0, float(self.app_config.camera.fps))
            meta = {
                "target_id": target_id,
                "target_name": target.display_name if target else "none",
                "category": target.category if target else "none",
                "provider": self.pipeline.model_manager.swapper.get_model_info().get("provider", "CPU"),
            }

            if self.video_recorder.start_recording(filepath, fps=fps, resolution=(w, h), extra_metadata=meta):
                self.record_btn.setText("⏹ Stop Recording")
                self.record_btn.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 8px 14px; border-radius: 5px;")
                self.status_bar.showMessage(f"Recording started: {os.path.basename(filepath)}")
        else:
            # Stop recording
            meta = self.video_recorder.stop_recording()
            self.record_btn.setText("⏺ Record Video")
            self.record_btn.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; padding: 8px 14px; border-radius: 5px;")
            dur = meta.get("duration_seconds", 0)
            self.status_bar.showMessage(f"Recording saved: {os.path.basename(meta.get('filepath', ''))} ({dur}s)")

    def _on_mask_type_changed(self, index: int) -> None:
        mtype = self.mask_combo.currentData() or "smooth_hull"
        self.app_config.processing.mask_type = mtype
        self.pipeline.mask_generator.clear_cache()
        logger.info(f"Mask type switched to '{mtype}'")

    def _on_feather_changed(self, value: int) -> None:
        self.app_config.processing.mask_feather = float(value) / 10.0
        self.pipeline.mask_generator.clear_cache()

    def _on_color_mode_changed(self, index: int) -> None:
        mode = self.color_combo.currentData() or "reinhard"
        self.app_config.processing.color_correction = mode

    def _on_enhancement_changed(self, value: int) -> None:
        self.app_config.processing.enhancement_strength = float(value) / 100.0

    def _on_profile_mode_changed(self, index: int) -> None:
        mode = self.profile_combo.currentData() or "quality"
        self.app_config.performance.mode = mode

    def closeEvent(self, event) -> None:
        """Gracefully releases video devices on window close."""
        self.stop_camera()
        event.accept()
