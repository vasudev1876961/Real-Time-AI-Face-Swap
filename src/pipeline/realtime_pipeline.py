"""
Real-Time Pipeline Execution Engine with Asynchronous Background Inference.
Coordinates frame ingestion, detection, tracking, neural swapping, and alpha blending.
"""

import time
import threading
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

from src.core.config_loader import AppConfig, ModelsConfig, TargetsConfig
from src.detection.face_detector import get_face_detector, FaceData
from src.detection.face_landmarks import INSWAPPER_STANDARD_128, FaceLandmarks
from src.tracking.face_tracker import FaceTracker
from src.alignment.face_alignment import FaceAligner
from src.models.model_manager import get_model_manager
from src.processing.mask import FaceMaskGenerator
from src.processing.color_correction import apply_color_correction, TemporalColorStabilizer
from src.processing.blending import FaceBlender
from src.processing.postprocess import postprocess_frame
from src.targets.target_manager import get_target_manager
from src.targets.target_loader import TargetFace
from src.performance.metrics import PerformanceMetrics, PipelineTimings
from src.utils.logger import get_logger

logger = get_logger("RealTimePipeline")


class PipelineResult:
    """Structured container holding all output artifacts from one pipeline pass."""

    def __init__(
        self,
        rendered_frame: np.ndarray,
        original_frame: np.ndarray,
        face_data: Optional[FaceData],
        target: Optional[TargetFace],
        metrics_summary: Dict[str, Any],
        is_swapped: bool,
        status_message: str,
    ):
        self.rendered_frame = rendered_frame
        self.original_frame = original_frame
        self.face_data = face_data
        self.target = target
        self.metrics_summary = metrics_summary
        self.is_swapped = is_swapped
        self.status_message = status_message


class RealTimePipeline:
    """
    Main Orchestrator for Real-Time AI Face Transformation.
    Runs video capture & blending on main thread with asynchronous background inference
    to guarantee smooth 30+ FPS interactive frame rate.
    """

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        models_config: Optional[ModelsConfig] = None,
        targets_config: Optional[TargetsConfig] = None,
    ):
        self.app_config = app_config or AppConfig()
        self.models_config = models_config or ModelsConfig()
        self.targets_config = targets_config or TargetsConfig()

        self.detector = get_face_detector(self.models_config.face_detection)
        self.tracker = FaceTracker(self.detector, self.app_config.performance)
        self.aligner = FaceAligner(target_crop_size=(128, 128))
        self.model_manager = get_model_manager(self.models_config)
        self.target_manager = get_target_manager(self.targets_config)
        self.mask_generator = FaceMaskGenerator(self.app_config.processing)
        self.blender = FaceBlender(self.app_config.processing)
        self.color_stabilizer = TemporalColorStabilizer(
            alpha=getattr(self.app_config.processing, "color_smoothing_alpha", 0.70)
        )
        self.metrics = PerformanceMetrics()

        self.enable_swapping = True

        # Asynchronous worker state for non-blocking 30+ FPS execution
        self._lock = threading.Lock()
        self._async_in_progress = False
        self._cached_swapped_crop: Optional[np.ndarray] = None
        self._cached_target_id: Optional[str] = None
        self._last_swap_latency_ms: float = 0.0

        logger.info("RealTimePipeline initialized successfully.")

    def reset_tracker(self) -> None:
        """Resets tracking state (invoked when camera or resolution switches)."""
        self.tracker.reset()
        self.color_stabilizer.reset()
        self.mask_generator.clear_cache()
        with self._lock:
            self._cached_swapped_crop = None

    def select_target(self, target_id: Optional[str]) -> bool:
        """Selects target identity by ID."""
        with self._lock:
            self._cached_swapped_crop = None
        return self.target_manager.select_target(target_id)

    def get_selected_target(self) -> Optional[TargetFace]:
        """Returns currently active target."""
        return self.target_manager.get_selected_target()

    def _async_swap_worker(
        self,
        aligned_crop: np.ndarray,
        face_data: FaceData,
        target_face: TargetFace,
    ) -> None:
        """Runs heavy neural inference in background thread."""
        try:
            swapped, lat = self.model_manager.swap(aligned_crop, face_data, target_face)
            with self._lock:
                self._cached_swapped_crop = swapped
                self._cached_target_id = target_face.target_id
                self._last_swap_latency_ms = lat
        except Exception as e:
            logger.error(f"Async swap worker exception: {e}")
        finally:
            with self._lock:
                self._async_in_progress = False

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """
        Executes one complete non-blocking pipeline pass on a video frame.
        """
        if frame is None or frame.size == 0:
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            return PipelineResult(
                rendered_frame=empty,
                original_frame=empty,
                face_data=None,
                target=None,
                metrics_summary=self.metrics.get_summary(False, False, False),
                is_swapped=False,
                status_message="Empty Frame",
            )

        t_start = time.perf_counter()
        timings = PipelineTimings()
        orig_frame = frame.copy()
        h, w = frame.shape[:2]

        # 1. Fast Face Tracking & Landmark Estimation (1-5 ms)
        t0 = time.perf_counter()
        face = self.tracker.update(frame)
        timings.tracking_ms = (time.perf_counter() - t0) * 1000.0

        is_swapped = False
        status_msg = "Preview Only"
        active_target = self.target_manager.get_selected_target()
        model_ready = self.model_manager.is_swap_ready()

        rendered_frame = frame

        if face is not None and active_target is not None and self.enable_swapping:
            if model_ready:
                try:
                    # 2. Face Alignment
                    t0 = time.perf_counter()
                    crop_size = (128, 128)
                    aligned_crop, mat, inv_mat = self.aligner.align(frame, face.landmarks, crop_size=crop_size)
                    timings.alignment_ms = (time.perf_counter() - t0) * 1000.0

                    # 3. Asynchronous Model Inference
                    with self._lock:
                        if not self._async_in_progress:
                            # Launch next background inference pass
                            self._async_in_progress = True
                            threading.Thread(
                                target=self._async_swap_worker,
                                args=(aligned_crop.copy(), face, active_target),
                                daemon=True,
                            ).start()

                        swapped_crop = self._cached_swapped_crop
                        timings.swap_ms = self._last_swap_latency_ms

                    # If swapped texture is ready, blend it immediately
                    if swapped_crop is not None:
                        # Estimate head pose for pose-adaptive mask adjustment
                        yaw, pitch = 0.0, 0.0
                        if face.landmarks is not None and len(face.landmarks) >= 5:
                            yaw, pitch = FaceLandmarks.calculate_head_pose_angles(face.landmarks)

                        # 4. Mask Generation using optimized anatomical contour & distance transform
                        t0 = time.perf_counter()
                        mask_crop = self.mask_generator.generate_mask(
                            crop_shape=crop_size,
                            landmarks=INSWAPPER_STANDARD_128,
                            yaw=yaw,
                            pitch=pitch,
                        )
                        timings.mask_ms = (time.perf_counter() - t0) * 1000.0

                        # 5. Mask-Weighted Color Matching & Temporal Stabilization
                        t0 = time.perf_counter()
                        stabilizer = (
                            self.color_stabilizer
                            if getattr(self.app_config.processing, "temporal_color_smoothing", True)
                            else None
                        )
                        corrected_crop = apply_color_correction(
                            original_crop=aligned_crop,
                            swapped_crop=swapped_crop,
                            method=self.app_config.processing.color_correction,
                            blend_ratio=self.app_config.processing.color_blend_ratio,
                            mask=mask_crop,
                            stabilizer=stabilizer,
                        )

                        # High-frequency facial clarity and texture enhancement
                        clarity_factor = getattr(self.app_config.processing, "postprocess_sharpen", 0.35)
                        if clarity_factor > 0.0:
                            blurred_crop = cv2.GaussianBlur(corrected_crop, (0, 0), 1.0)
                            enhanced_crop = cv2.addWeighted(corrected_crop, 1.0 + clarity_factor, blurred_crop, -clarity_factor, 0)
                        else:
                            enhanced_crop = corrected_crop
                        timings.color_ms = (time.perf_counter() - t0) * 1000.0

                        # 6. Accelerated ROI Blending
                        t0 = time.perf_counter()
                        rendered_frame = self.blender.blend(
                            original_frame=frame,
                            swapped_crop=enhanced_crop,
                            mask_crop=mask_crop,
                            inv_matrix=inv_mat,
                        )
                        timings.blend_ms = (time.perf_counter() - t0) * 1000.0

                        is_swapped = True
                        status_msg = f"Swapped: {active_target.display_name}"
                    else:
                        status_msg = "Warming up..."

                except Exception as e:
                    logger.error(f"Pipeline execution error: {e}")
                    rendered_frame = frame
                    status_msg = "Pipeline Error"
            else:
                status_msg = "Model Not Found (Preview Only)"
        elif face is None:
            status_msg = "No Face Detected"
        elif active_target is None:
            status_msg = "No Target Selected"

        # 6. Post-processing & Telemetry
        sharpen_amt = getattr(self.app_config.processing, "postprocess_sharpen", 0.35)
        rendered_frame = postprocess_frame(
            rendered_frame,
            sharpen_amount=sharpen_amt,
            face_data=face,
            timings=timings,
            show_hud=(self.app_config.performance.mode == "debug"),
        )

        timings.total_ms = (time.perf_counter() - t_start) * 1000.0
        self.metrics.record_frame(timings)

        metrics_summary = self.metrics.get_summary(
            face_detected=(face is not None),
            is_swapped=is_swapped,
            model_loaded=model_ready,
        )

        return PipelineResult(
            rendered_frame=rendered_frame,
            original_frame=orig_frame,
            face_data=face,
            target=active_target,
            metrics_summary=metrics_summary,
            is_swapped=is_swapped,
            status_message=status_msg,
        )
