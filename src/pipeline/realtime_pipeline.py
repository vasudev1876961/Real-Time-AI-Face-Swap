"""
Real-Time Pipeline Execution Engine with Multi-Face Tracking, Occlusion-Aware Masking,
Temporal Motion Stabilization, and Virtual Camera Broadcasting.
Coordinates frame ingestion, detection, tracking, neural swapping, and alpha blending.
"""

import time
import threading
from typing import Optional, Dict, Any, Tuple, List
import cv2
import numpy as np

from src.core.config_loader import AppConfig, ModelsConfig, TargetsConfig
from src.detection.face_detector import get_face_detector, FaceData
from src.detection.face_landmarks import INSWAPPER_STANDARD_128, FaceLandmarks
from src.tracking.face_tracker import FaceTracker, TrackedFace
from src.alignment.face_alignment import FaceAligner
from src.models.model_manager import get_model_manager
from src.processing.mask import FaceMaskGenerator
from src.processing.color_correction import apply_color_correction, TemporalColorStabilizer
from src.processing.blending import FaceBlender
from src.processing.postprocess import postprocess_frame
from src.processing.enhancement import FaceEnhancer, inject_original_skin_texture
from src.processing.occlusion import OcclusionDetector
from src.processing.stabilizer import TemporalMotionStabilizer
from src.processing.color_grading import ColorGradingEngine, ColorGradingConfig
from src.processing.mouth_preservation import OralCavityPreserver
from src.camera.virtual_camera import VirtualCameraBroadcaster
from src.targets.target_manager import get_target_manager
from src.targets.target_loader import TargetFace
from src.performance.metrics import PerformanceMetrics, PipelineTimings
from src.optimization import FPSMonitor, get_gpu_manager, PipelineProfiler, AdaptivePerformanceGovernor
from src.utils.logger import get_logger

logger = get_logger("RealTimePipeline")


class AsyncInferenceWorker:
    """
    Dedicated persistent background worker thread for neural face swapping.
    Uses drop-oldest single-item queue with threading.Condition to eliminate
    per-frame OS thread creation overhead while processing the freshest frame available.
    """

    def __init__(self, pipeline: "RealTimePipeline"):
        self.pipeline = pipeline
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending_job: Optional[Tuple[np.ndarray, FaceData, TargetFace]] = None
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="AsyncSwapWorker")
        self._thread.start()

    def submit(self, aligned_crop: np.ndarray, face_data: FaceData, target_face: TargetFace) -> None:
        """Submits a job, replacing any unstarted pending job (drop-oldest policy)."""
        with self._cv:
            self._pending_job = (aligned_crop, face_data, target_face)
            self._cv.notify()

    def _worker_loop(self) -> None:
        while self._running:
            job = None
            with self._cv:
                while self._running and self._pending_job is None:
                    self._cv.wait(timeout=0.1)
                if not self._running:
                    break
                job = self._pending_job
                self._pending_job = None

            if job is not None:
                aligned_crop, face_data, target_face = job
                self.pipeline._execute_swap_inference(aligned_crop, face_data, target_face)

    def stop(self) -> None:
        self._running = False
        with self._cv:
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)


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
        all_faces: Optional[List[FaceData]] = None,
        active_tracks: Optional[List[TrackedFace]] = None,
        is_broadcasting: bool = False,
        broadcast_url: str = "",
    ):
        self.rendered_frame = rendered_frame
        self.original_frame = original_frame
        self.face_data = face_data
        self.target = target
        self.metrics_summary = metrics_summary
        self.is_swapped = is_swapped
        self.status_message = status_message
        self.all_faces = all_faces or ([] if face_data is None else [face_data])
        self.active_tracks = active_tracks or []
        self.is_broadcasting = is_broadcasting
        self.broadcast_url = broadcast_url


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
        self.enhancer = FaceEnhancer(
            model_config=getattr(self.models_config, "enhancement", None),
            default_strength=getattr(self.app_config.processing, "enhancement_strength", 0.40),
        )
        self.metrics = PerformanceMetrics()

        # Phase 6 Advancements: Occlusion, Stabilization, Multi-Face, Virtual Camera
        self.occlusion_detector = OcclusionDetector(
            sensitivity=getattr(self.app_config.processing, "occlusion_sensitivity", 0.50)
        )
        self.motion_stabilizer = TemporalMotionStabilizer(
            motion_alpha=getattr(self.app_config.processing, "motion_stabilization_alpha", 0.60)
        )

        vcam_cfg = getattr(self.app_config, "virtual_camera", None)
        vcam_port = getattr(vcam_cfg, "port", 8080) if vcam_cfg else 8080
        self.broadcaster = VirtualCameraBroadcaster(port=vcam_port)

        self.enable_swapping = True
        self.multi_face_mode = getattr(self.app_config.processing, "multi_face_mode", "primary")
        self.target_mappings: Dict[int, str] = {}

        # Performance & Telemetry Engine (Phase 7)
        self.fps_monitor = FPSMonitor()
        self.gpu_manager = get_gpu_manager()
        self.profiler = PipelineProfiler()
        self.governor = AdaptivePerformanceGovernor(
            target_fps=getattr(self.app_config.performance, "target_fps", 30)
        )

        # Phase 8: Studio Color Grading & Oral Cavity Preservation
        self.color_grader = ColorGradingEngine()
        self.mouth_preserver = OralCavityPreserver(
            default_strength=getattr(self.app_config.processing, "mouth_preservation_strength", 0.65)
        )

        # Asynchronous worker state and persistent worker thread
        self._lock = threading.Lock()
        self._async_in_progress = False
        self._cached_swapped_crop: Optional[np.ndarray] = None
        self._cached_target_id: Optional[str] = None
        self._cached_swaps: Dict[Tuple[int, str], np.ndarray] = {}
        self._last_swap_latency_ms: float = 0.0
        self.async_worker = AsyncInferenceWorker(self)

        logger.info("RealTimePipeline initialized successfully (Phase 8).")

    def reset_tracker(self) -> None:
        """Resets tracking state (invoked when camera or resolution switches)."""
        self.tracker.reset()
        self.color_stabilizer.reset()
        self.motion_stabilizer.reset()
        self.occlusion_detector.reset()
        self.mask_generator.clear_cache()
        self.fps_monitor.reset()
        self.profiler.reset()
        with self._lock:
            self._cached_swapped_crop = None
            self._cached_swaps.clear()

    def stop(self) -> None:
        """Stops background threads and broadcasting."""
        if hasattr(self, "async_worker"):
            self.async_worker.stop()
        if hasattr(self, "broadcaster"):
            self.broadcaster.stop()

    def select_target(self, target_id: Optional[str]) -> bool:
        """Selects target identity by ID."""
        with self._lock:
            self._cached_swapped_crop = None
            self._cached_swaps.clear()
        return self.target_manager.select_target(target_id)

    def get_selected_target(self) -> Optional[TargetFace]:
        """Returns currently active target."""
        return self.target_manager.get_selected_target()

    def set_target_for_track(self, track_id: int, target_id: Optional[str]) -> None:
        """Assigns a specific target identity to a tracked face ID."""
        if target_id is None:
            self.target_mappings.pop(track_id, None)
        else:
            self.target_mappings[track_id] = target_id

    def set_multi_face_mode(self, mode: str) -> None:
        """Sets multi-face swapping mode ('primary', 'all', 'mapped')."""
        self.multi_face_mode = mode

    def start_broadcasting(self, width: int = 1280, height: int = 720, fps: int = 30) -> bool:
        """Starts virtual camera and HTTP MJPEG streaming."""
        return self.broadcaster.start(width, height, fps)

    def stop_broadcasting(self) -> None:
        """Stops virtual camera and HTTP stream server."""
        self.broadcaster.stop()

    def is_broadcasting(self) -> bool:
        """Returns whether virtual camera / network stream is active."""
        return self.broadcaster.is_active()

    def get_broadcast_url(self) -> str:
        """Returns the local network URL for live streaming."""
        return self.broadcaster.get_stream_url() if self.broadcaster.is_active() else ""

    def _get_active_color_grading_config(self) -> ColorGradingConfig:
        """Retrieves active ColorGradingConfig blending preset and explicit overrides."""
        preset_name = getattr(self.app_config.processing, "color_grading_preset", "neutral")
        base = ColorGradingEngine.get_preset(preset_name)
        exp = getattr(self.app_config.processing, "color_grading_exposure", base.exposure)
        contrast = getattr(self.app_config.processing, "color_grading_contrast", base.contrast)
        sat = getattr(self.app_config.processing, "color_grading_saturation", base.saturation)
        temp = getattr(self.app_config.processing, "color_grading_temperature", base.temperature)
        tint = getattr(self.app_config.processing, "color_grading_tint", base.tint)
        gamma = getattr(self.app_config.processing, "color_grading_gamma", base.gamma)
        return ColorGradingConfig(
            enabled=True,
            exposure=exp,
            contrast=contrast,
            saturation=sat,
            temperature=temp,
            tint=tint,
            gamma=gamma,
        )

    def _execute_swap_inference(
        self,
        aligned_crop: np.ndarray,
        face_data: FaceData,
        target_face: TargetFace,
    ) -> None:
        """Runs heavy neural inference in dedicated persistent background worker."""
        try:
            t0 = time.perf_counter()
            swapped, lat = self.model_manager.swap(aligned_crop, face_data, target_face)

            mode = self.governor.get_recommended_enhancement_mode(
                getattr(self.app_config.processing, "enhancement_mode", "onnx")
            )
            enh_strength = getattr(self.app_config.processing, "enhancement_strength", 0.50)
            if mode == "onnx" and self.enhancer.has_neural_model() and enh_strength > 0.01:
                swapped = self.enhancer.enhance(
                    swapped,
                    strength=enh_strength,
                    landmarks=INSWAPPER_STANDARD_128,
                    keep_native_resolution=True,
                )

            total_lat = (time.perf_counter() - t0) * 1000.0

            with self._lock:
                self._cached_swapped_crop = swapped
                self._cached_target_id = target_face.target_id
                track_id = face_data.track_id or 1
                self._cached_swaps[(track_id, target_face.target_id)] = swapped
                self._last_swap_latency_ms = total_lat
        except Exception as e:
            logger.error(f"Async swap worker exception: {e}")

    def _async_swap_worker(
        self,
        aligned_crop: np.ndarray,
        face_data: FaceData,
        target_face: TargetFace,
    ) -> None:
        """Backwards-compatible wrapper for single-pass swap worker."""
        self._execute_swap_inference(aligned_crop, face_data, target_face)

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """
        Executes one complete non-blocking pipeline pass on a video frame,
        supporting multi-face scenes, occlusion carving, and temporal stabilization.
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

        # 1. Multi-Face Tracking & Association
        # Dynamically modulate detection interval via performance governor
        rec_interval = self.governor.get_recommended_detection_interval(
            self.app_config.performance.detection_interval
        )
        self.tracker.config.detection_interval = rec_interval

        t0 = time.perf_counter()
        faces = self.tracker.update_all(frame)
        primary_face = faces[0] if faces else None
        active_tracks = self.tracker.get_active_tracks()
        timings.tracking_ms = (time.perf_counter() - t0) * 1000.0

        is_swapped = False
        status_msg = "Preview Only"
        default_target = self.target_manager.get_selected_target()
        model_ready = self.model_manager.is_swap_ready()

        rendered_frame = frame

        # Determine faces to transform based on multi_face_mode
        if faces and self.enable_swapping and model_ready:
            if self.multi_face_mode == "all":
                faces_to_process = faces
            elif self.multi_face_mode == "mapped":
                faces_to_process = [
                    f for f in faces if (f.track_id in self.target_mappings or default_target is not None)
                ]
            else:
                faces_to_process = [primary_face] if primary_face else []

            swapped_any = False
            for face in faces_to_process:
                # Resolve target for this specific face
                target_id = self.target_mappings.get(face.track_id) if face.track_id else None
                face_target = self.target_manager.get_target_by_id(target_id) if target_id else default_target
                if face_target is None:
                    continue

                try:
                    # 2. Alignment & Affine Transform
                    t0 = time.perf_counter()
                    crop_size = (128, 128)
                    aligned_crop, mat, inv_mat = self.aligner.align(frame, face.landmarks, crop_size=crop_size)

                    # Temporal Motion & Affine Stabilization
                    if getattr(self.app_config.processing, "enable_stabilization", True):
                        mat, inv_mat = self.motion_stabilizer.stabilize_transform(
                            mat, inv_mat, track_id=face.track_id or 1
                        )
                    timings.alignment_ms = (time.perf_counter() - t0) * 1000.0

                    # 3. Asynchronous Neural Model Inference via persistent worker queue
                    track_key = (face.track_id or 1, face_target.target_id)
                    self.async_worker.submit(aligned_crop.copy(), face, face_target)

                    with self._lock:
                        swapped_crop = self._cached_swaps.get(track_key, self._cached_swapped_crop)
                        timings.swap_ms = self._last_swap_latency_ms

                    if swapped_crop is not None:
                        # 4. Pose angles & Mask Generation
                        yaw, pitch = 0.0, 0.0
                        if face.landmarks is not None and len(face.landmarks) >= 5:
                            yaw, pitch = FaceLandmarks.calculate_head_pose_angles(face.landmarks)

                        t0 = time.perf_counter()
                        occl_det = (
                            self.occlusion_detector
                            if getattr(self.app_config.processing, "enable_occlusion", True)
                            else None
                        )
                        mask_crop = self.mask_generator.generate_mask(
                            crop_shape=crop_size,
                            landmarks=INSWAPPER_STANDARD_128,
                            yaw=yaw,
                            pitch=pitch,
                            aligned_crop=aligned_crop,
                            occlusion_detector=occl_det,
                            track_id=face.track_id or 1,
                        )
                        timings.mask_ms = (time.perf_counter() - t0) * 1000.0

                        # 5. Color Correction & Directional Illumination Transfer
                        t0 = time.perf_counter()
                        stabilizer = (
                            self.color_stabilizer
                            if getattr(self.app_config.processing, "temporal_color_smoothing", True)
                            else None
                        )
                        illum_match = getattr(self.app_config.processing, "illumination_matching", True)
                        corrected_crop = apply_color_correction(
                            original_crop=aligned_crop,
                            swapped_crop=swapped_crop,
                            method=self.app_config.processing.color_correction,
                            blend_ratio=self.app_config.processing.color_blend_ratio,
                            mask=mask_crop,
                            stabilizer=stabilizer,
                            illumination_match=illum_match,
                        )

                        # 6. Enhancement & Skin Texture Injection
                        enh_strength = getattr(self.app_config.processing, "enhancement_strength", 0.50)
                        mode = getattr(self.app_config.processing, "enhancement_mode", "onnx")
                        tex_transfer = getattr(self.app_config.processing, "texture_detail_transfer", 0.35)

                        if mode == "onnx" and self.enhancer.has_neural_model():
                            if tex_transfer > 0.01:
                                enhanced_crop = inject_original_skin_texture(
                                    original_crop=aligned_crop,
                                    swapped_crop=corrected_crop,
                                    amount=tex_transfer * enh_strength,
                                    mask=mask_crop,
                                    occlusion_matte=None,
                                )
                            else:
                                enhanced_crop = corrected_crop
                        elif mode != "off" and enh_strength > 0.0:
                            enhanced_crop = self.enhancer.enhance(
                                corrected_crop,
                                strength=enh_strength,
                                landmarks=INSWAPPER_STANDARD_128,
                                original_crop=aligned_crop,
                                texture_amount=tex_transfer,
                                keep_native_resolution=False,
                            )
                        else:
                            enhanced_crop = corrected_crop

                        # Phase 8: Oral Cavity & Natural Dental Fidelity Preservation
                        if getattr(self.app_config.processing, "enable_mouth_preservation", True):
                            mouth_str = getattr(self.app_config.processing, "mouth_preservation_strength", 0.65)
                            enhanced_crop = self.mouth_preserver.preserve_mouth_fidelity(
                                original_crop=aligned_crop,
                                swapped_crop=enhanced_crop,
                                landmarks=INSWAPPER_STANDARD_128,
                                strength=mouth_str,
                            )

                        # Phase 8: Studio Color Grading
                        grade_cfg = self._get_active_color_grading_config()
                        if grade_cfg.enabled:
                            enhanced_crop = self.color_grader.apply(enhanced_crop, grade_cfg)

                        # Temporal Luminance Stabilization
                        if getattr(self.app_config.processing, "enable_stabilization", True):
                            enhanced_crop = self.motion_stabilizer.stabilize_luminance(
                                enhanced_crop, track_id=face.track_id or 1
                            )
                        timings.color_ms = (time.perf_counter() - t0) * 1000.0

                        # 7. Multi-Band ROI Blending
                        t0 = time.perf_counter()
                        rendered_frame = self.blender.blend(
                            original_frame=rendered_frame,
                            swapped_crop=enhanced_crop,
                            mask_crop=mask_crop,
                            inv_matrix=inv_mat,
                        )
                        timings.blend_ms = (time.perf_counter() - t0) * 1000.0
                        swapped_any = True

                except Exception as e:
                    logger.error(f"Pipeline multi-face error on track {face.track_id}: {e}")

            if swapped_any:
                is_swapped = True
                status_msg = f"Swapped: {len(faces_to_process)} face(s)"
            else:
                status_msg = "Warming up..."
        elif not faces:
            status_msg = "No Face Detected"
        elif not default_target and not self.target_mappings:
            status_msg = "No Target Selected"
        elif not model_ready:
            status_msg = "Model Not Found (Preview Only)"

        # 8. Post-processing & Telemetry HUD
        sharpen_amt = getattr(self.app_config.processing, "postprocess_sharpen", 0.30)
        rendered_frame = postprocess_frame(
            rendered_frame,
            sharpen_amount=sharpen_amt,
            face_data=primary_face,
            timings=timings,
            show_hud=(self.app_config.performance.mode == "debug"),
        )

        timings.total_ms = (time.perf_counter() - t_start) * 1000.0
        self.metrics.record_frame(timings)
        rolling_fps = self.fps_monitor.tick()
        self.governor.update(rolling_fps)

        # 9. Broadcast frame to Virtual Camera / HTTP Network Stream
        if self.broadcaster.is_active():
            self.broadcaster.send_frame(rendered_frame)

        metrics_summary = self.metrics.get_summary(
            face_detected=(len(faces) > 0),
            is_swapped=is_swapped,
            model_loaded=model_ready,
        )
        # Augment with Phase 7 optimization telemetry
        metrics_summary["rolling_fps"] = round(rolling_fps, 1)
        metrics_summary["jitter_ms"] = round(self.fps_monitor.get_frame_jitter_ms(), 2)
        metrics_summary["percentiles"] = self.fps_monitor.get_latency_percentiles()
        metrics_summary["governor"] = self.governor.get_status_badge()
        metrics_summary["vram"] = self.gpu_manager.get_vram_info()

        return PipelineResult(
            rendered_frame=rendered_frame,
            original_frame=orig_frame,
            face_data=primary_face,
            target=default_target,
            metrics_summary=metrics_summary,
            is_swapped=is_swapped,
            status_message=status_msg,
            all_faces=faces,
            active_tracks=active_tracks,
            is_broadcasting=self.broadcaster.is_active(),
            broadcast_url=self.get_broadcast_url(),
        )

    def process_frame_sync(
        self,
        frame: np.ndarray,
        target: Optional[TargetFace] = None,
        enhance_strength: Optional[float] = None,
        face_mode: Optional[str] = None,
        target_map: Optional[Dict[int, str]] = None,
    ) -> PipelineResult:
        """
        Executes a 100% synchronous, deterministic pipeline pass on a frame.
        Guarantees instant transformation without waiting on background threads.
        Ideal for offline video processing and multi-face batch photo transformations.
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

        # 1. Multi-Face Tracking
        t0 = time.perf_counter()
        faces = self.tracker.update_all(frame)
        primary_face = faces[0] if faces else None
        active_tracks = self.tracker.get_active_tracks()
        timings.tracking_ms = (time.perf_counter() - t0) * 1000.0

        is_swapped = False
        status_msg = "Preview Only"
        default_target = target or self.target_manager.get_selected_target()
        model_ready = self.model_manager.is_swap_ready()
        rendered_frame = frame.copy()

        mode = face_mode or self.multi_face_mode

        if faces and model_ready and self.enable_swapping:
            if mode == "all":
                faces_to_process = faces
            elif mode == "mapped" and target_map:
                faces_to_process = [f for f in faces if f.track_id in target_map]
            else:
                faces_to_process = [primary_face] if primary_face else []

            swapped_any = False
            for face in faces_to_process:
                # Resolve target
                t_id = target_map.get(face.track_id) if (target_map and face.track_id) else None
                face_target = self.target_manager.get_target_by_id(t_id) if t_id else default_target
                if face_target is None:
                    continue

                try:
                    # 2. Alignment
                    t0 = time.perf_counter()
                    crop_size = (128, 128)
                    aligned_crop, mat, inv_mat = self.aligner.align(frame, face.landmarks, crop_size=crop_size)

                    if getattr(self.app_config.processing, "enable_stabilization", True):
                        mat, inv_mat = self.motion_stabilizer.stabilize_transform(
                            mat, inv_mat, track_id=face.track_id or 1
                        )
                    timings.alignment_ms = (time.perf_counter() - t0) * 1000.0

                    # 3. Synchronous Swap
                    t0 = time.perf_counter()
                    swapped_crop, lat = self.model_manager.swap(aligned_crop, face, face_target)
                    timings.swap_ms = lat

                    if swapped_crop is not None:
                        # 4. Pose & Mask
                        yaw, pitch = 0.0, 0.0
                        if face.landmarks is not None and len(face.landmarks) >= 5:
                            yaw, pitch = FaceLandmarks.calculate_head_pose_angles(face.landmarks)

                        t0 = time.perf_counter()
                        occl_det = (
                            self.occlusion_detector
                            if getattr(self.app_config.processing, "enable_occlusion", True)
                            else None
                        )
                        mask_crop = self.mask_generator.generate_mask(
                            crop_shape=crop_size,
                            landmarks=INSWAPPER_STANDARD_128,
                            yaw=yaw,
                            pitch=pitch,
                            aligned_crop=aligned_crop,
                            occlusion_detector=occl_det,
                        )
                        timings.mask_ms = (time.perf_counter() - t0) * 1000.0

                        # 5. Color Correction
                        t0 = time.perf_counter()
                        stabilizer = (
                            self.color_stabilizer
                            if getattr(self.app_config.processing, "temporal_color_smoothing", True)
                            else None
                        )
                        illum_match = getattr(self.app_config.processing, "illumination_matching", True)
                        corrected_crop = apply_color_correction(
                            original_crop=aligned_crop,
                            swapped_crop=swapped_crop,
                            method=self.app_config.processing.color_correction,
                            blend_ratio=self.app_config.processing.color_blend_ratio,
                            mask=mask_crop,
                            stabilizer=stabilizer,
                            illumination_match=illum_match,
                        )

                        # 6. Enhancement
                        k_strength = (
                            enhance_strength
                            if enhance_strength is not None
                            else getattr(self.app_config.processing, "enhancement_strength", 0.50)
                        )
                        enh_mode = getattr(self.app_config.processing, "enhancement_mode", "onnx")
                        tex_transfer = getattr(self.app_config.processing, "texture_detail_transfer", 0.35)

                        if enh_mode != "off" and k_strength > 0.0:
                            enhanced_crop = self.enhancer.enhance(
                                corrected_crop,
                                strength=k_strength,
                                landmarks=INSWAPPER_STANDARD_128,
                                original_crop=aligned_crop,
                                texture_amount=tex_transfer,
                                keep_native_resolution=True,
                            )
                        else:
                            enhanced_crop = corrected_crop

                        # Phase 8: Oral Cavity & Natural Dental Fidelity Preservation
                        if getattr(self.app_config.processing, "enable_mouth_preservation", True):
                            mouth_str = getattr(self.app_config.processing, "mouth_preservation_strength", 0.65)
                            enhanced_crop = self.mouth_preserver.preserve_mouth_fidelity(
                                original_crop=aligned_crop,
                                swapped_crop=enhanced_crop,
                                landmarks=INSWAPPER_STANDARD_128,
                                strength=mouth_str,
                            )

                        # Phase 8: Studio Color Grading
                        grade_cfg = self._get_active_color_grading_config()
                        if grade_cfg.enabled:
                            enhanced_crop = self.color_grader.apply(enhanced_crop, grade_cfg)

                        if getattr(self.app_config.processing, "enable_stabilization", True):
                            enhanced_crop = self.motion_stabilizer.stabilize_luminance(
                                enhanced_crop, track_id=face.track_id or 1
                            )
                        timings.color_ms = (time.perf_counter() - t0) * 1000.0

                        # 7. Blending
                        t0 = time.perf_counter()
                        rendered_frame = self.blender.blend(
                            original_frame=rendered_frame,
                            swapped_crop=enhanced_crop,
                            mask_crop=mask_crop,
                            inv_matrix=inv_mat,
                        )
                        timings.blend_ms = (time.perf_counter() - t0) * 1000.0
                        swapped_any = True

                except Exception as e:
                    logger.error(f"Synchronous pipeline error: {e}")

            if swapped_any:
                is_swapped = True
                status_msg = f"Swapped: {len(faces_to_process)} face(s)"
        elif not faces:
            status_msg = "No Face Detected"
        elif not default_target:
            status_msg = "No Target Selected"
        elif not model_ready:
            status_msg = "Model Not Found"

        sharpen_amt = getattr(self.app_config.processing, "postprocess_sharpen", 0.30)
        rendered_frame = postprocess_frame(
            rendered_frame,
            sharpen_amount=sharpen_amt,
            face_data=primary_face,
            timings=timings,
            show_hud=False,
        )

        timings.total_ms = (time.perf_counter() - t_start) * 1000.0
        self.metrics.record_frame(timings)

        metrics_summary = self.metrics.get_summary(
            face_detected=(len(faces) > 0),
            is_swapped=is_swapped,
            model_loaded=model_ready,
        )

        return PipelineResult(
            rendered_frame=rendered_frame,
            original_frame=orig_frame,
            face_data=primary_face,
            target=default_target,
            metrics_summary=metrics_summary,
            is_swapped=is_swapped,
            status_message=status_msg,
            all_faces=faces,
            active_tracks=active_tracks,
            is_broadcasting=False,
            broadcast_url="",
        )
