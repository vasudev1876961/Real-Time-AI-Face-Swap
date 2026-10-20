"""
Interactive Web Studio Application for Real-Time AI Face Swap.
Powered by FastAPI, Uvicorn, and HTML5/Vanilla JS.
Provides live ultra-low-latency video streaming, target identity switching,
pipeline parameter tuning, studio telemetry HUD, and snapshot capture.
"""

import os
import sys
import time
import json
import threading
import datetime
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config_loader import AppConfig, ModelsConfig, TargetsConfig, load_all_configs
from src.pipeline.realtime_pipeline import RealTimePipeline, PipelineResult
from src.camera.camera_manager import CameraManager
from src.targets.target_loader import TargetFace
from src.recording.video_recorder import ThreadedVideoRecorder
from src.utils.logger import get_logger

logger = get_logger("WebStudio")

# Directory paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
CAPTURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "captures")
FACES_DIR = os.path.join(PROJECT_ROOT, "faces")
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "outputs", "recordings")

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)


class ConfigUpdateModel(BaseModel):
    enhancement_strength: Optional[float] = None
    eye_realism_strength: Optional[float] = None
    mouth_preservation_strength: Optional[float] = None
    specular_lighting_strength: Optional[float] = None
    occlusion_sensitivity: Optional[float] = None
    color_grading_preset: Optional[str] = None
    color_correction: Optional[str] = None
    enable_stabilization: Optional[bool] = None
    enable_swapping: Optional[bool] = None
    multi_face_mode: Optional[str] = None
    expression_transfer_strength: Optional[float] = None
    enable_expression_transfer: Optional[bool] = None
    enable_turbo_spatial_caching: Optional[bool] = None
    enable_spectacles_preservation: Optional[bool] = None
    spectacles_preservation_strength: Optional[float] = None
    enable_hand_occlusion: Optional[bool] = None
    hand_occlusion_strength: Optional[float] = None
    visualize_occlusion_hud: Optional[bool] = None

    # Phase 13 Mask Precision & FPS Telemetry
    enable_dense_mesh_mask: Optional[bool] = None
    enable_edge_snapping: Optional[bool] = None
    edge_snapping_strength: Optional[float] = None
    enable_curvature_feathering: Optional[bool] = None
    enable_hairline_carving: Optional[bool] = None
    hairline_carving_strength: Optional[float] = None
    visualize_mask_hud: Optional[bool] = None
    show_fps_hud: Optional[bool] = None

    # Phase 15 Volumetric Relighting, Identity Morphing & Split-Screen Comparison
    enable_volumetric_relighting: Optional[bool] = None
    volumetric_shadow_strength: Optional[float] = None
    subsurface_scattering_warmth: Optional[float] = None
    enable_identity_morphing: Optional[bool] = None
    identity_morph_duration: Optional[float] = None
    dual_target_fusion_ratio: Optional[float] = None
    secondary_target_id: Optional[str] = None
    split_screen_mode: Optional[str] = None
    split_screen_position: Optional[float] = None

    # Phase 16 Cinematic Depth, Micro-Expressions & Motion Blur
    enable_cinematic_bokeh: Optional[bool] = None
    bokeh_blur_radius: Optional[float] = None
    bokeh_focus_depth: Optional[float] = None
    bokeh_specular_boost: Optional[float] = None
    enable_micro_expressions: Optional[bool] = None
    micro_expression_strength: Optional[float] = None
    micro_expression_sensitivity: Optional[float] = None
    enable_motion_blur_matching: Optional[bool] = None
    motion_blur_sensitivity: Optional[float] = None
    visualize_depth_hud: Optional[bool] = None

    # Phase 17 Studio Rim Lighting, Multi-Identity Matrix & Temporal Coherence
    enable_studio_lighting: Optional[bool] = None
    studio_rim_intensity: Optional[float] = None
    studio_rim_thickness: Optional[int] = None
    studio_rim_color: Optional[str] = None
    studio_backdrop_dim: Optional[float] = None
    enable_temporal_coherence: Optional[bool] = None
    identity_coherence_factor: Optional[float] = None
    temporal_jitter_suppression: Optional[float] = None
    multi_identity_mode: Optional[str] = None
    spatial_target_ids: Optional[List[str]] = None


class StudioLightingUpdateModel(BaseModel):
    enabled: Optional[bool] = None
    rim_intensity: Optional[float] = None
    rim_thickness: Optional[int] = None
    rim_color: Optional[str] = None
    backdrop_dim: Optional[float] = None


class MultiIdentityUpdateModel(BaseModel):
    mode: Optional[str] = None
    custom_mappings: Optional[Dict[str, str]] = None
    spatial_target_ids: Optional[List[str]] = None


class TemporalCoherenceUpdateModel(BaseModel):
    enabled: Optional[bool] = None
    smoothing_factor: Optional[float] = None
    jitter_suppression: Optional[float] = None


class WebPipelineRunner:
    """
    Background worker thread continuously capturing frames from camera (or synthetic fallback),
    passing them through RealTimePipeline, and caching the freshest JPEG buffers for web clients.
    """

    def __init__(
        self,
        app_cfg: AppConfig,
        models_cfg: ModelsConfig,
        targets_cfg: TargetsConfig,
    ):
        self.app_cfg = app_cfg
        self.models_cfg = models_cfg
        self.targets_cfg = targets_cfg

        self.pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)
        self.camera_manager = CameraManager(app_cfg.camera)
        self.recorder = ThreadedVideoRecorder()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Cached buffers
        self._latest_rendered_jpeg: Optional[bytes] = None
        self._latest_original_jpeg: Optional[bytes] = None
        self._latest_result: Optional[PipelineResult] = None
        self._last_frame_time = time.time()
        self._active_clients = 0

        # Synthetic fallback state
        self._use_synthetic_source = False
        self._synthetic_frames: List[np.ndarray] = []
        self._synth_frame_idx = 0
        self._load_synthetic_assets()

    def _load_synthetic_assets(self) -> None:
        """Loads reference images to serve as animated synthetic loop when no webcam is present."""
        sample_paths = [
            os.path.join(CAPTURES_DIR, "capture_2026-08-24_21-47-58_hero_001.jpg"),
            os.path.join(CAPTURES_DIR, "test_phase9_rekha_prabhas.jpg"),
            os.path.join(CAPTURES_DIR, "test_swapped_phase7.jpg"),
        ]
        for p in sample_paths:
            if os.path.isfile(p):
                img = cv2.imread(p)
                if img is not None:
                    self._synthetic_frames.append(img)

        # If no capture files exist, generate a clean synthetic portrait frame with a simulated face
        if not self._synthetic_frames:
            canvas = np.full((720, 1280, 3), 32, dtype=np.uint8)
            # Draw synthetic face silhouette
            cv2.ellipse(canvas, (640, 360), (140, 190), 0, 0, 360, (190, 160, 140), -1)
            cv2.circle(canvas, (590, 320), 12, (50, 40, 30), -1)
            cv2.circle(canvas, (690, 320), 12, (50, 40, 30), -1)
            cv2.ellipse(canvas, (640, 430), (45, 20), 0, 0, 180, (80, 50, 150), -1)
            self._synthetic_frames.append(canvas)

    def start(self) -> None:
        """Starts acquisition loop."""
        if self._running:
            return

        cam_started = False
        try:
            cam_started = self.camera_manager.start()
        except Exception as e:
            logger.warning(f"Hardware camera could not be opened: {e}. Switching to synthetic source.")

        if not cam_started:
            self._use_synthetic_source = True
            logger.info("Web Studio operating in Synthetic Video Loop mode.")
        else:
            logger.info("Web Studio successfully acquired physical camera feed.")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="WebPipelineThread")
        self._thread.start()

    def stop(self) -> None:
        """Stops acquisition loop."""
        self._running = False
        if hasattr(self, "recorder") and self.recorder.is_recording():
            self.recorder.stop_recording()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.camera_manager.stop()
        self.pipeline.stop()

    def _loop(self) -> None:
        """Worker loop executing real-time pipeline at target framerate."""
        target_fps = getattr(self.app_cfg.performance, "target_fps", 30)
        frame_interval = 1.0 / max(1, target_fps)
        last_frame_idx = -1

        while self._running:
            t0 = time.perf_counter()
            frame = None

            if not self._use_synthetic_source and self.camera_manager.is_running():
                pkt = self.camera_manager.get_latest_frame()
                if pkt is None or pkt.frame is None:
                    time.sleep(0.005)
                    continue
                if pkt.frame_index == last_frame_idx:
                    time.sleep(0.003)
                    continue
                last_frame_idx = pkt.frame_index
                frame = pkt.frame
            else:
                # Cycle synthetic frames with gentle animated oscillation to trigger tracker
                base_frame = self._synthetic_frames[self._synth_frame_idx % len(self._synthetic_frames)]
                self._synth_frame_idx += 1

                # Apply subtle sine-wave translation to simulate natural head motion
                h, w = base_frame.shape[:2]
                dx = int(8.0 * np.sin(time.time() * 2.0))
                dy = int(4.0 * np.cos(time.time() * 2.5))
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = cv2.warpAffine(base_frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Process frame through full RealTimePipeline
            result = self.pipeline.process_frame(frame)

            # Write frame to active MP4 recording if enabled
            if hasattr(self, "recorder") and self.recorder.is_recording():
                self.recorder.write_frame(result.rendered_frame)

            # Encode both rendered and original frames as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, rend_buf = cv2.imencode(".jpg", result.rendered_frame, encode_params)
            _, orig_buf = cv2.imencode(".jpg", result.original_frame, encode_params)

            with self._lock:
                self._latest_rendered_jpeg = rend_buf.tobytes()
                self._latest_original_jpeg = orig_buf.tobytes()
                self._latest_result = result
                self._last_frame_time = time.time()

            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_rendered_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_rendered_jpeg

    def get_latest_original_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_original_jpeg

    def start_recording(self, fps: float = 30.0, resolution: Tuple[int, int] = (1280, 720)) -> Dict[str, Any]:
        """Starts asynchronous MP4 video recording."""
        with self._lock:
            if self.recorder.is_recording():
                return {"success": False, "message": "Recording already in progress."}
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"studio_recording_{timestamp}.mp4"
            filepath = os.path.join(RECORDINGS_DIR, filename)
            active_target = self.pipeline.get_selected_target()
            meta = {
                "target_id": active_target.person_id if active_target else None,
                "target_name": active_target.display_name if active_target else "None",
                "category": active_target.category if active_target else None,
            }
            success = self.recorder.start_recording(filepath, fps=fps, resolution=resolution, extra_metadata=meta)
            return {"success": success, "filename": filename, "filepath": filepath, "url": f"/recordings/{filename}"}

    def stop_recording(self) -> Dict[str, Any]:
        """Stops active MP4 video recording and finalizes container."""
        with self._lock:
            if not self.recorder.is_recording():
                return {"success": False, "message": "No active recording."}
            meta = self.recorder.stop_recording()
            filename = os.path.basename(meta.get("filepath", ""))
            meta["filename"] = filename
            meta["url"] = f"/recordings/{filename}" if filename else ""
            meta["success"] = True
            return meta

    def get_recording_status(self) -> Dict[str, Any]:
        """Returns current recording status and frame statistics."""
        with self._lock:
            rec = self.recorder
            is_rec = rec.is_recording()
            dur = round(time.time() - rec._start_time, 1) if is_rec else 0.0
            return {
                "is_recording": is_rec,
                "duration_seconds": dur,
                "frames_written": getattr(rec, "_frames_written", 0),
                "frames_dropped": getattr(rec, "_frames_dropped", 0),
                "filename": os.path.basename(getattr(rec, "_filepath", "")) if is_rec else "",
            }

    def get_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            res = self._latest_result
            if res is None:
                return {
                    "status": "initializing",
                    "fps": 0.0,
                    "target_fps": 30,
                    "is_swapped": False,
                    "message": "Engine starting...",
                    "active_clients": self._active_clients,
                    "is_synthetic": self._use_synthetic_source,
                    "recording": self.get_recording_status(),
                    "hardware": {
                        "device": self.pipeline.gpu_manager.device_name,
                        "has_gpu": self.pipeline.gpu_manager.has_gpu,
                        "vram": self.pipeline.gpu_manager.get_vram_info(),
                    },
                }

            metrics = res.metrics_summary
            stage_timings = metrics.get("timings", {})
            active_target = res.target

            return {
                "status": "running",
                "fps": metrics.get("current_fps", 0.0),
                "target_fps": getattr(self.app_cfg.performance, "target_fps", 30),
                "is_swapped": res.is_swapped,
                "message": res.status_message,
                "face_count": len(res.all_faces) if res.all_faces else 0,
                "stage_timings": stage_timings,
                "hardware": {
                    "device": self.pipeline.gpu_manager.device_name,
                    "has_gpu": self.pipeline.gpu_manager.has_gpu,
                    "vram": self.pipeline.gpu_manager.get_vram_info(),
                },
                "governor_state": self.pipeline.governor.current_state,
                "active_target": {
                    "id": active_target.person_id if active_target else None,
                    "name": active_target.display_name if active_target else "None",
                    "category": active_target.category if active_target else "",
                } if active_target else None,
                "active_clients": self._active_clients,
                "is_synthetic": self._use_synthetic_source,
                "recording": self.get_recording_status(),
                "buffer_pool": self.pipeline.buffer_pool.get_stats(),
                "model_status": self.pipeline.model_manager.get_status_summary(),
            }

    def register_client(self) -> None:
        with self._lock:
            self._active_clients += 1

    def unregister_client(self) -> None:
        with self._lock:
            self._active_clients = max(0, self._active_clients - 1)


# Global Runner Instance
_runner: Optional[WebPipelineRunner] = None
_runner_lock = threading.Lock()


def get_web_runner() -> WebPipelineRunner:
    global _runner
    with _runner_lock:
        if _runner is None:
            a_cfg, m_cfg, t_cfg = load_all_configs()
            _runner = WebPipelineRunner(a_cfg, m_cfg, t_cfg)
            _runner.start()
        return _runner


def create_app() -> FastAPI:
    """Builds and configures the FastAPI application."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        get_web_runner()
        yield
        global _runner
        if _runner:
            _runner.stop()
            _runner = None

    app = FastAPI(
        title="Real-Time AI Face Swap — Web Studio",
        description="Low-Latency Real-Time AI Face Swap Studio & Live Broadcast",
        version="10.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static assets & media
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
    os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")
    app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")
    app.mount("/faces", StaticFiles(directory=FACES_DIR), name="faces")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.isfile(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>Real-Time AI Face Swap Web Studio Initializing...</h1>"

    @app.get("/stream")
    async def stream_rendered():
        """MJPEG video feed of swapped/transformed output."""
        runner = get_web_runner()

        def frame_generator():
            runner.register_client()
            try:
                while True:
                    frame_bytes = runner.get_latest_rendered_jpeg()
                    if frame_bytes is not None:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame_bytes)).encode("utf-8") + b"\r\n\r\n"
                            + frame_bytes + b"\r\n"
                        )
                    time.sleep(0.033)  # ~30 FPS
            finally:
                runner.unregister_client()

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/original")
    async def stream_original():
        """MJPEG video feed of original camera feed for split comparison."""
        runner = get_web_runner()

        def frame_generator():
            try:
                while True:
                    frame_bytes = runner.get_latest_original_jpeg()
                    if frame_bytes is not None:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame_bytes)).encode("utf-8") + b"\r\n\r\n"
                            + frame_bytes + b"\r\n"
                        )
                    time.sleep(0.033)
            finally:
                pass

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/telemetry")
    async def get_telemetry():
        runner = get_web_runner()
        return runner.get_telemetry()

    @app.get("/api/targets")
    async def get_targets():
        runner = get_web_runner()
        categories = runner.pipeline.target_manager.get_categories()
        selected = runner.pipeline.target_manager.get_selected_target()
        selected_id = selected.person_id if selected else None

        result: Dict[str, Any] = {
            "categories": categories,
            "selected_id": selected_id,
            "targets": [],
        }

        all_targets = runner.pipeline.target_manager.get_all_targets()
        for t in all_targets:
            # Build relative URL for reference image
            img_path = getattr(t, "reference_image_path", getattr(t, "image_path", ""))
            rel_path = os.path.relpath(img_path, FACES_DIR).replace("\\", "/") if img_path else ""
            img_url = f"/faces/{rel_path}" if rel_path else "/static/icons/default_avatar.png"

            result["targets"].append({
                "id": t.person_id,
                "name": t.display_name,
                "category": t.category,
                "image_url": img_url,
                "has_embedding": t.embedding is not None,
                "is_selected": (t.person_id == selected_id),
            })

        return result

    @app.post("/api/target/select")
    async def select_target(payload: Dict[str, str]):
        target_id = payload.get("target_id")
        runner = get_web_runner()
        success = runner.pipeline.select_target(target_id)
        if not success and target_id is not None:
            raise HTTPException(status_code=404, detail=f"Target '{target_id}' not found.")
        return {"success": True, "selected_id": target_id}

    @app.post("/api/target/upload")
    async def upload_target(
        file: UploadFile = File(...),
        name: str = Form(...),
        category: str = Form("custom"),
    ):
        """Uploads custom portrait, saves reference, computes embedding."""
        runner = get_web_runner()
        clean_name = "".join(c for c in name.lower().replace(" ", "_") if c.isalnum() or c == "_")
        if not clean_name:
            clean_name = f"user_{int(time.time())}"

        target_dir = os.path.join(FACES_DIR, category, clean_name)
        os.makedirs(target_dir, exist_ok=True)
        ref_path = os.path.join(target_dir, "reference.jpg")

        content = await file.read()
        with open(ref_path, "wb") as f:
            f.write(content)

        # Write metadata.json
        meta = {
            "name": name,
            "category": category,
            "created_at": datetime.datetime.now().isoformat(),
            "notes": "Uploaded via Web Studio",
        }
        with open(os.path.join(target_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Refresh target manager
        runner.pipeline.target_manager.reload_all_targets()
        runner.pipeline.select_target(clean_name)

        return {"success": True, "target_id": clean_name, "name": name}

    @app.get("/api/pipeline/config")
    async def get_pipeline_config():
        runner = get_web_runner()
        p = runner.pipeline.app_config.processing
        return {
            "enhancement_strength": getattr(p, "enhancement_strength", 0.40),
            "eye_realism_strength": getattr(p, "eye_realism_strength", 0.70),
            "mouth_preservation_strength": getattr(p, "mouth_preservation_strength", 0.65),
            "specular_lighting_strength": getattr(p, "specular_lighting_strength", 0.50),
            "occlusion_sensitivity": getattr(p, "occlusion_sensitivity", 0.50),
            "color_grading_preset": getattr(p, "color_grading_preset", "neutral"),
            "color_correction": getattr(p, "color_correction", "reinhard"),
            "enable_stabilization": getattr(p, "enable_stabilization", True),
            "enable_swapping": runner.pipeline.enable_swapping,
            "multi_face_mode": runner.pipeline.multi_face_mode,
            "expression_transfer_strength": getattr(p, "expression_transfer_strength", 0.65),
            "enable_expression_transfer": getattr(p, "enable_expression_transfer", True),
            "enable_turbo_spatial_caching": getattr(p, "enable_turbo_spatial_caching", True),
            "enable_spectacles_preservation": getattr(p, "enable_spectacles_preservation", True),
            "spectacles_preservation_strength": getattr(p, "spectacles_preservation_strength", 0.75),
            "enable_hand_occlusion": getattr(p, "enable_hand_occlusion", True),
            "hand_occlusion_strength": getattr(p, "hand_occlusion_strength", 0.60),
            "visualize_occlusion_hud": getattr(p, "visualize_occlusion_hud", False),
            "enable_dense_mesh_mask": getattr(p, "enable_dense_mesh_mask", True),
            "enable_edge_snapping": getattr(p, "enable_edge_snapping", True),
            "edge_snapping_strength": getattr(p, "edge_snapping_strength", 0.65),
            "enable_curvature_feathering": getattr(p, "enable_curvature_feathering", True),
            "enable_hairline_carving": getattr(p, "enable_hairline_carving", True),
            "hairline_carving_strength": getattr(p, "hairline_carving_strength", 0.50),
            "visualize_mask_hud": getattr(p, "visualize_mask_hud", False),
            "show_fps_hud": getattr(p, "show_fps_hud", True),
            # Phase 15
            "enable_volumetric_relighting": getattr(p, "enable_volumetric_relighting", True),
            "volumetric_shadow_strength": getattr(p, "volumetric_shadow_strength", 0.50),
            "subsurface_scattering_warmth": getattr(p, "subsurface_scattering_warmth", 0.45),
            "enable_identity_morphing": getattr(p, "enable_identity_morphing", True),
            "identity_morph_duration": getattr(p, "identity_morph_duration", 0.50),
            "dual_target_fusion_ratio": getattr(p, "dual_target_fusion_ratio", 0.0),
            "secondary_target_id": runner.pipeline.secondary_target_id,
            "split_screen_mode": getattr(p, "split_screen_mode", "off"),
            "split_screen_position": getattr(p, "split_screen_position", 0.50),
            # Phase 16
            "enable_cinematic_bokeh": getattr(p, "enable_cinematic_bokeh", False),
            "bokeh_blur_radius": getattr(p, "bokeh_blur_radius", 12.0),
            "bokeh_focus_depth": getattr(p, "bokeh_focus_depth", 0.35),
            "bokeh_specular_boost": getattr(p, "bokeh_specular_boost", 0.25),
            "enable_micro_expressions": getattr(p, "enable_micro_expressions", True),
            "micro_expression_strength": getattr(p, "micro_expression_strength", 0.60),
            "micro_expression_sensitivity": getattr(p, "micro_expression_sensitivity", 0.75),
            "enable_motion_blur_matching": getattr(p, "enable_motion_blur_matching", True),
            "motion_blur_sensitivity": getattr(p, "motion_blur_sensitivity", 0.50),
            "visualize_depth_hud": getattr(p, "visualize_depth_hud", False),
            # Phase 17
            "enable_studio_lighting": getattr(p, "enable_studio_lighting", False),
            "studio_rim_intensity": getattr(p, "studio_rim_intensity", 0.50),
            "studio_rim_thickness": getattr(p, "studio_rim_thickness", 3),
            "studio_rim_color": getattr(p, "studio_rim_color", "warm_gold"),
            "studio_backdrop_dim": getattr(p, "studio_backdrop_dim", 0.15),
            "enable_temporal_coherence": getattr(p, "enable_temporal_coherence", True),
            "identity_coherence_factor": getattr(p, "identity_coherence_factor", 0.85),
            "temporal_jitter_suppression": getattr(p, "temporal_jitter_suppression", 0.40),
            "multi_identity_mode": getattr(p, "multi_identity_mode", "primary"),
            "spatial_target_ids": getattr(p, "spatial_target_ids", []),
        }

    @app.post("/api/pipeline/config")
    async def update_pipeline_config(config: ConfigUpdateModel):
        runner = get_web_runner()
        p = runner.pipeline.app_config.processing

        if config.enhancement_strength is not None:
            p.enhancement_strength = float(np.clip(config.enhancement_strength, 0.0, 1.0))
            runner.pipeline.enhancer.default_strength = p.enhancement_strength

        if config.eye_realism_strength is not None:
            p.eye_realism_strength = float(np.clip(config.eye_realism_strength, 0.0, 1.0))
            runner.pipeline.eye_preserver.default_strength = p.eye_realism_strength

        if config.mouth_preservation_strength is not None:
            p.mouth_preservation_strength = float(np.clip(config.mouth_preservation_strength, 0.0, 1.0))
            runner.pipeline.mouth_preserver.default_strength = p.mouth_preservation_strength

        if config.specular_lighting_strength is not None:
            p.specular_lighting_strength = float(np.clip(config.specular_lighting_strength, 0.0, 1.0))
            runner.pipeline.lighting_harmonizer.default_strength = p.specular_lighting_strength

        if config.occlusion_sensitivity is not None:
            p.occlusion_sensitivity = float(np.clip(config.occlusion_sensitivity, 0.0, 1.0))
            runner.pipeline.occlusion_detector.sensitivity = p.occlusion_sensitivity

        if config.color_grading_preset is not None:
            p.color_grading_preset = config.color_grading_preset

        if config.color_correction is not None:
            p.color_correction = config.color_correction

        if config.enable_stabilization is not None:
            p.enable_stabilization = bool(config.enable_stabilization)

        if config.enable_swapping is not None:
            runner.pipeline.enable_swapping = bool(config.enable_swapping)

        if config.multi_face_mode is not None:
            runner.pipeline.set_multi_face_mode(config.multi_face_mode)

        if config.expression_transfer_strength is not None:
            p.expression_transfer_strength = float(np.clip(config.expression_transfer_strength, 0.0, 1.0))
            runner.pipeline.expression_engine.default_strength = p.expression_transfer_strength

        if config.enable_expression_transfer is not None:
            p.enable_expression_transfer = bool(config.enable_expression_transfer)

        if config.enable_turbo_spatial_caching is not None:
            p.enable_turbo_spatial_caching = bool(config.enable_turbo_spatial_caching)

        # Phase 12 Hand Carving & Spectacles Settings
        if config.enable_spectacles_preservation is not None:
            p.enable_spectacles_preservation = bool(config.enable_spectacles_preservation)
            runner.pipeline.occlusion_detector.enable_spectacles = p.enable_spectacles_preservation

        if config.spectacles_preservation_strength is not None:
            p.spectacles_preservation_strength = float(np.clip(config.spectacles_preservation_strength, 0.0, 1.0))
            runner.pipeline.occlusion_detector.spectacles_strength = p.spectacles_preservation_strength
            runner.pipeline.occlusion_detector.spectacles_engine.default_strength = p.spectacles_preservation_strength

        if config.enable_hand_occlusion is not None:
            p.enable_hand_occlusion = bool(config.enable_hand_occlusion)
            runner.pipeline.occlusion_detector.enable_hand_carving = p.enable_hand_occlusion

        if config.hand_occlusion_strength is not None:
            p.hand_occlusion_strength = float(np.clip(config.hand_occlusion_strength, 0.0, 1.0))
            runner.pipeline.occlusion_detector.hand_carving_strength = p.hand_occlusion_strength
            runner.pipeline.occlusion_detector.hand_engine.sensitivity = p.hand_occlusion_strength

        if config.visualize_occlusion_hud is not None:
            p.visualize_occlusion_hud = bool(config.visualize_occlusion_hud)
            runner.pipeline.occlusion_detector.visualize_hud = p.visualize_occlusion_hud

        # Phase 13 Mask Precision & FPS HUD Settings
        if config.enable_dense_mesh_mask is not None:
            p.enable_dense_mesh_mask = bool(config.enable_dense_mesh_mask)
        if config.enable_edge_snapping is not None:
            p.enable_edge_snapping = bool(config.enable_edge_snapping)
        if config.edge_snapping_strength is not None:
            p.edge_snapping_strength = float(np.clip(config.edge_snapping_strength, 0.0, 1.0))
        if config.enable_curvature_feathering is not None:
            p.enable_curvature_feathering = bool(config.enable_curvature_feathering)
        if config.enable_hairline_carving is not None:
            p.enable_hairline_carving = bool(config.enable_hairline_carving)
        if config.hairline_carving_strength is not None:
            p.hairline_carving_strength = float(np.clip(config.hairline_carving_strength, 0.0, 1.0))
        if config.visualize_mask_hud is not None:
            p.visualize_mask_hud = bool(config.visualize_mask_hud)
        if config.show_fps_hud is not None:
            p.show_fps_hud = bool(config.show_fps_hud)

        # Phase 15 Volumetric Relighting, Identity Morphing & Split Screen Settings
        if config.enable_volumetric_relighting is not None:
            runner.pipeline.set_volumetric_relighting_config(enabled=config.enable_volumetric_relighting)
        if config.volumetric_shadow_strength is not None:
            runner.pipeline.set_volumetric_relighting_config(shadow_strength=config.volumetric_shadow_strength)
        if config.subsurface_scattering_warmth is not None:
            runner.pipeline.set_volumetric_relighting_config(sss_warmth=config.subsurface_scattering_warmth)
        if config.enable_identity_morphing is not None:
            p.enable_identity_morphing = bool(config.enable_identity_morphing)
        if config.identity_morph_duration is not None:
            runner.pipeline.set_identity_morph_config(duration=config.identity_morph_duration)
        if config.dual_target_fusion_ratio is not None:
            runner.pipeline.set_identity_morph_config(fusion_ratio=config.dual_target_fusion_ratio)
        if config.secondary_target_id is not None:
            runner.pipeline.set_identity_morph_config(secondary_target_id=config.secondary_target_id)
        if config.split_screen_mode is not None:
            runner.pipeline.set_split_screen_config(mode=config.split_screen_mode)
        if config.split_screen_position is not None:
            runner.pipeline.set_split_screen_config(position=config.split_screen_position)

        # Phase 16 Cinematic Depth, Micro-Expressions & Motion Blur Settings
        if (
            config.enable_cinematic_bokeh is not None
            or config.bokeh_blur_radius is not None
            or config.bokeh_focus_depth is not None
            or config.bokeh_specular_boost is not None
            or config.visualize_depth_hud is not None
        ):
            runner.pipeline.set_cinematic_bokeh_config(
                enabled=config.enable_cinematic_bokeh,
                bokeh_radius=config.bokeh_blur_radius,
                focus_depth=config.bokeh_focus_depth,
                specular_boost=config.bokeh_specular_boost,
                visualize_depth_hud=config.visualize_depth_hud,
            )
        if (
            config.enable_micro_expressions is not None
            or config.micro_expression_strength is not None
            or config.micro_expression_sensitivity is not None
        ):
            runner.pipeline.set_micro_expression_config(
                enabled=config.enable_micro_expressions,
                strength=config.micro_expression_strength,
                sensitivity=config.micro_expression_sensitivity,
            )
        if config.enable_motion_blur_matching is not None or config.motion_blur_sensitivity is not None:
            runner.pipeline.set_motion_blur_config(
                enabled=config.enable_motion_blur_matching,
                sensitivity=config.motion_blur_sensitivity,
            )

        # Phase 17 Studio Rim Lighting, Multi-Identity Matrix & Temporal Coherence
        if (
            config.enable_studio_lighting is not None
            or config.studio_rim_intensity is not None
            or config.studio_rim_thickness is not None
            or config.studio_rim_color is not None
            or config.studio_backdrop_dim is not None
        ):
            runner.pipeline.set_studio_lighting_config(
                enabled=config.enable_studio_lighting,
                rim_intensity=config.studio_rim_intensity,
                rim_thickness=config.studio_rim_thickness,
                rim_color=config.studio_rim_color,
                backdrop_dim=config.studio_backdrop_dim,
            )
        if (
            config.enable_temporal_coherence is not None
            or config.identity_coherence_factor is not None
            or config.temporal_jitter_suppression is not None
        ):
            runner.pipeline.set_temporal_coherence_config(
                enabled=config.enable_temporal_coherence,
                smoothing_factor=config.identity_coherence_factor,
                jitter_suppression=config.temporal_jitter_suppression,
            )
        if config.multi_identity_mode is not None or config.spatial_target_ids is not None:
            runner.pipeline.set_multi_identity_matrix_config(
                mode=config.multi_identity_mode,
                spatial_target_ids=config.spatial_target_ids,
            )

        dump_dict = config.model_dump(exclude_unset=True) if hasattr(config, "model_dump") else config.dict(exclude_unset=True)
        return {"success": True, "updated": dump_dict}

    @app.post("/api/settings/morph")
    async def update_morph_settings(
        secondary_target_id: Optional[str] = None,
        fusion_ratio: Optional[float] = None,
        duration: Optional[float] = None,
        trigger_target_id: Optional[str] = None,
    ):
        """Dedicated endpoint for Phase 15 Identity Morphing and Dual-Target Fusion."""
        runner = get_web_runner()
        if trigger_target_id:
            ok = runner.pipeline.trigger_morph_transition(trigger_target_id, duration=duration)
            if not ok:
                raise HTTPException(status_code=404, detail=f"Target '{trigger_target_id}' not found.")
        else:
            runner.pipeline.set_identity_morph_config(
                secondary_target_id=secondary_target_id,
                fusion_ratio=fusion_ratio,
                duration=duration,
            )
        return {"success": True, "morph": runner.pipeline.morph_engine.get_telemetry()}

    @app.post("/api/settings/relighting")
    async def update_relighting_settings(
        enabled: Optional[bool] = None,
        shadow_strength: Optional[float] = None,
        sss_warmth: Optional[float] = None,
    ):
        """Dedicated endpoint for Phase 15 Volumetric Relighting & SSS warmth."""
        runner = get_web_runner()
        runner.pipeline.set_volumetric_relighting_config(
            enabled=enabled,
            shadow_strength=shadow_strength,
            sss_warmth=sss_warmth,
        )
        return {
            "success": True,
            "relighting": {
                "enabled": runner.pipeline.app_config.processing.enable_volumetric_relighting,
                "shadow_strength": runner.pipeline.relighting_engine.default_shadow_strength,
                "sss_warmth": runner.pipeline.relighting_engine.default_warmth_strength,
                "light_vector": runner.pipeline.relighting_engine._last_light_vector,
            },
        }

    @app.post("/api/settings/split_screen")
    async def update_split_screen_settings(
        mode: Optional[str] = None,
        position: Optional[float] = None,
    ):
        """Dedicated endpoint for Phase 15 Live Split-Screen Comparison."""
        runner = get_web_runner()
        runner.pipeline.set_split_screen_config(mode=mode, position=position)
        return {
            "success": True,
            "split_screen": {
                "mode": runner.pipeline.app_config.processing.split_screen_mode,
                "position": runner.pipeline.app_config.processing.split_screen_position,
            },
        }

    @app.get("/api/telemetry/phase15")
    async def get_phase15_telemetry():
        """Returns consolidated Phase 15 morph, relighting, and comparison telemetry."""
        runner = get_web_runner()
        p = runner.pipeline.app_config.processing
        return {
            "morph": runner.pipeline.morph_engine.get_telemetry(),
            "relighting": {
                "enabled": getattr(p, "enable_volumetric_relighting", True),
                "light_vector": runner.pipeline.relighting_engine._last_light_vector,
                "shadow_strength": getattr(p, "volumetric_shadow_strength", 0.50),
                "sss_warmth": getattr(p, "subsurface_scattering_warmth", 0.45),
            },
            "split_screen": {
                "mode": getattr(p, "split_screen_mode", "off"),
                "position": getattr(p, "split_screen_position", 0.50),
            },
            "dual_fusion": {
                "secondary_target_id": runner.pipeline.secondary_target_id,
                "fusion_ratio": getattr(p, "dual_target_fusion_ratio", 0.0),
            },
        }

    @app.post("/api/settings/depth_bokeh")
    async def update_depth_bokeh_settings(
        enabled: Optional[bool] = None,
        bokeh_radius: Optional[float] = None,
        focus_depth: Optional[float] = None,
        specular_boost: Optional[float] = None,
        visualize_depth_hud: Optional[bool] = None,
    ):
        """Dedicated endpoint for Phase 16 Cinematic Depth-of-Field Portrait Bokeh."""
        runner = get_web_runner()
        runner.pipeline.set_cinematic_bokeh_config(
            enabled=enabled,
            bokeh_radius=bokeh_radius,
            focus_depth=focus_depth,
            specular_boost=specular_boost,
            visualize_depth_hud=visualize_depth_hud,
        )
        return {
            "success": True,
            "depth_bokeh": {
                "enabled": runner.pipeline.app_config.processing.enable_cinematic_bokeh,
                "bokeh_radius": runner.pipeline.depth_engine.bokeh_radius,
                "focus_depth": runner.pipeline.depth_engine.focus_depth,
                "specular_boost": runner.pipeline.depth_engine.specular_boost,
                "visualize_depth_hud": runner.pipeline.app_config.processing.visualize_depth_hud,
                "mean_depth": runner.pipeline.depth_engine._last_metrics.get("mean_depth", 0.50),
            },
        }

    @app.post("/api/settings/micro_expression")
    async def update_micro_expression_settings(
        enabled: Optional[bool] = None,
        strength: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ):
        """Dedicated endpoint for Phase 16 Dynamic Micro-Expression Wrinkle Transfer."""
        runner = get_web_runner()
        runner.pipeline.set_micro_expression_config(
            enabled=enabled,
            strength=strength,
            sensitivity=sensitivity,
        )
        return {
            "success": True,
            "micro_expression": runner.pipeline.micro_expression_engine.get_telemetry(),
        }

    @app.post("/api/settings/motion_blur")
    async def update_motion_blur_settings(
        enabled: Optional[bool] = None,
        sensitivity: Optional[float] = None,
    ):
        """Dedicated endpoint for Phase 16 Directional Motion Blur Harmonization."""
        runner = get_web_runner()
        runner.pipeline.set_motion_blur_config(
            enabled=enabled,
            sensitivity=sensitivity,
        )
        return {
            "success": True,
            "motion_blur": runner.pipeline.motion_blur_harmonizer.get_telemetry(),
        }

    @app.get("/api/telemetry/phase16")
    async def get_phase16_telemetry():
        """Returns consolidated Phase 16 depth bokeh, micro-expression, and motion blur telemetry."""
        runner = get_web_runner()
        p = runner.pipeline.app_config.processing
        return {
            "depth_bokeh": {
                "enabled": getattr(p, "enable_cinematic_bokeh", False),
                "bokeh_radius": getattr(p, "bokeh_blur_radius", 12.0),
                "focus_depth": getattr(p, "bokeh_focus_depth", 0.35),
                "specular_boost": getattr(p, "bokeh_specular_boost", 0.25),
                "visualize_depth_hud": getattr(p, "visualize_depth_hud", False),
                **runner.pipeline.depth_engine.get_telemetry(),
            },
            "micro_expression": {
                "enabled": getattr(p, "enable_micro_expressions", True),
                "strength": getattr(p, "micro_expression_strength", 0.60),
                "sensitivity": getattr(p, "micro_expression_sensitivity", 0.75),
                **runner.pipeline.micro_expression_engine.get_telemetry(),
            },
            "motion_blur": {
                "enabled": getattr(p, "enable_motion_blur_matching", True),
                "sensitivity": getattr(p, "motion_blur_sensitivity", 0.50),
                **runner.pipeline.motion_blur_harmonizer.get_telemetry(),
            },
        }

    @app.post("/api/settings/studio_lighting")
    async def update_studio_lighting_settings(
        payload: Optional[StudioLightingUpdateModel] = None,
        enabled: Optional[bool] = Query(None),
        rim_intensity: Optional[float] = Query(None),
        rim_thickness: Optional[int] = Query(None),
        rim_color: Optional[str] = Query(None),
        backdrop_dim: Optional[float] = Query(None),
    ):
        """Dedicated endpoint for Phase 17 Virtual Portrait Studio Lighting & Edge Rim Glow."""
        runner = get_web_runner()
        eff_enabled = payload.enabled if payload and payload.enabled is not None else enabled
        eff_intensity = payload.rim_intensity if payload and payload.rim_intensity is not None else rim_intensity
        eff_thickness = payload.rim_thickness if payload and payload.rim_thickness is not None else rim_thickness
        eff_color = payload.rim_color if payload and payload.rim_color is not None else rim_color
        eff_dim = payload.backdrop_dim if payload and payload.backdrop_dim is not None else backdrop_dim

        runner.pipeline.set_studio_lighting_config(
            enabled=eff_enabled,
            rim_intensity=eff_intensity,
            rim_thickness=eff_thickness,
            rim_color=eff_color,
            backdrop_dim=eff_dim,
        )
        return {
            "success": True,
            "studio_lighting": runner.pipeline.studio_lighting_engine.get_telemetry(),
        }

    @app.post("/api/settings/multi_identity")
    async def update_multi_identity_settings(
        payload: Optional[MultiIdentityUpdateModel] = None,
        mode: Optional[str] = Query(None),
    ):
        """Dedicated endpoint for Phase 17 Multi-Subject Identity Binding Matrix."""
        runner = get_web_runner()
        eff_mode = payload.mode if payload and payload.mode is not None else mode
        eff_mappings = payload.custom_mappings if payload and payload.custom_mappings is not None else None
        eff_spatial = payload.spatial_target_ids if payload and payload.spatial_target_ids is not None else None

        int_mappings = {int(k): v for k, v in eff_mappings.items()} if eff_mappings else None
        runner.pipeline.set_multi_identity_matrix_config(
            mode=eff_mode,
            custom_mappings=int_mappings,
            spatial_target_ids=eff_spatial,
        )
        return {
            "success": True,
            "multi_identity": runner.pipeline.multi_identity_manager.get_telemetry(),
        }

    @app.post("/api/settings/temporal_coherence")
    async def update_temporal_coherence_settings(
        payload: Optional[TemporalCoherenceUpdateModel] = None,
        enabled: Optional[bool] = Query(None),
        smoothing_factor: Optional[float] = Query(None),
        jitter_suppression: Optional[float] = Query(None),
    ):
        """Dedicated endpoint for Phase 17 Temporal Identity Coherence & Micro-Jitter Suppression."""
        runner = get_web_runner()
        eff_enabled = payload.enabled if payload and payload.enabled is not None else enabled
        eff_smooth = payload.smoothing_factor if payload and payload.smoothing_factor is not None else smoothing_factor
        eff_jitter = payload.jitter_suppression if payload and payload.jitter_suppression is not None else jitter_suppression

        runner.pipeline.set_temporal_coherence_config(
            enabled=eff_enabled,
            smoothing_factor=eff_smooth,
            jitter_suppression=eff_jitter,
        )
        return {
            "success": True,
            "temporal_coherence": runner.pipeline.temporal_identity_smoother.get_telemetry(),
        }

    @app.get("/api/telemetry/phase17")
    async def get_phase17_telemetry():
        """Returns consolidated Phase 17 studio lighting, multi-identity, and temporal coherence telemetry."""
        runner = get_web_runner()
        p = runner.pipeline.app_config.processing
        return {
            "studio_lighting": {
                "enabled": getattr(p, "enable_studio_lighting", False),
                **runner.pipeline.studio_lighting_engine.get_telemetry(),
            },
            "multi_identity": {
                "mode": getattr(p, "multi_identity_mode", "primary"),
                **runner.pipeline.multi_identity_manager.get_telemetry(),
            },
            "temporal_coherence": {
                "enabled": getattr(p, "enable_temporal_coherence", True),
                **runner.pipeline.temporal_identity_smoother.get_telemetry(),
            },
        }

    @app.post("/api/showcase/generate")
    async def generate_showcase(target_id: Optional[str] = "prabhas"):
        """Triggers dynamic generation of Phase 17 multi-stage showcase and split comparison."""
        runner = get_web_runner()
        from src.processing.showcase_generator import ShowcaseGenerator
        with runner._lock:
            frame_bytes = runner.get_latest_original_jpeg()
        
        if frame_bytes:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            frame = None

        if frame is None:
            # Fallback to capture
            candidates = [
                os.path.join(CAPTURES_DIR, "capture_2026-08-24_21-47-58_hero_001.jpg"),
                os.path.join(CAPTURES_DIR, "test_swapped_phase7.jpg"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    frame = cv2.imread(c)
                    break

        if frame is None:
            raise HTTPException(status_code=400, detail="No source frame available to generate showcase.")

        generator = ShowcaseGenerator(pipeline=runner.pipeline)
        grid_file = os.path.join(CAPTURES_DIR, f"showcase_grid_{int(time.time())}.jpg")
        split_file = os.path.join(CAPTURES_DIR, f"showcase_split_{int(time.time())}.jpg")

        generator.generate_comparison_grid(frame, target_id=target_id or "prabhas", output_path=grid_file)
        generator.generate_split_showcase(frame, target_id=target_id or "prabhas", output_path=split_file)

        return {
            "success": True,
            "grid_url": f"/captures/{os.path.basename(grid_file)}",
            "split_url": f"/captures/{os.path.basename(split_file)}",
        }

    @app.get("/api/telemetry/fps")
    async def get_fps_telemetry():
        """Returns dedicated real-time FPS and latency performance metrics."""
        runner = get_web_runner()
        return runner.pipeline.get_fps_telemetry()

    @app.post("/api/settings/mask")
    async def update_mask_settings(
        enable_dense_mesh: Optional[bool] = None,
        enable_edge_snapping: Optional[bool] = None,
        edge_snapping_strength: Optional[float] = None,
        enable_curvature_feathering: Optional[bool] = None,
        enable_hairline_carving: Optional[bool] = None,
        hairline_carving_strength: Optional[float] = None,
        visualize_mask_hud: Optional[bool] = None,
        show_fps_hud: Optional[bool] = None,
    ):
        """Dedicated endpoint for Phase 13 mask precision and FPS HUD tuning."""
        runner = get_web_runner()
        runner.pipeline.set_mask_fitting_config(
            enable_dense_mesh=enable_dense_mesh,
            enable_edge_snapping=enable_edge_snapping,
            edge_snapping_strength=edge_snapping_strength,
            enable_curvature_feathering=enable_curvature_feathering,
            enable_hairline_carving=enable_hairline_carving,
            hairline_carving_strength=hairline_carving_strength,
            visualize_mask_hud=visualize_mask_hud,
        )
        if show_fps_hud is not None:
            runner.pipeline.set_fps_hud(show_fps_hud)
        return {"success": True}

    @app.post("/api/settings/occlusion")
    async def update_occlusion_settings(
        sensitivity: Optional[float] = None,
        enable_spectacles: Optional[bool] = None,
        spectacles_strength: Optional[float] = None,
        enable_hand_carving: Optional[bool] = None,
        hand_strength: Optional[float] = None,
        visualize_hud: Optional[bool] = None,
    ):
        """Dedicated high-level endpoint for Phase 12 Occlusion & Spectacles tuning."""
        runner = get_web_runner()
        det = runner.pipeline.occlusion_detector
        p = runner.pipeline.app_config.processing

        if sensitivity is not None:
            p.occlusion_sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
            det.sensitivity = p.occlusion_sensitivity
        if enable_spectacles is not None:
            p.enable_spectacles_preservation = bool(enable_spectacles)
            det.enable_spectacles = p.enable_spectacles_preservation
        if spectacles_strength is not None:
            p.spectacles_preservation_strength = float(np.clip(spectacles_strength, 0.0, 1.0))
            det.spectacles_strength = p.spectacles_preservation_strength
            det.spectacles_engine.default_strength = p.spectacles_preservation_strength
        if enable_hand_carving is not None:
            p.enable_hand_occlusion = bool(enable_hand_carving)
            det.enable_hand_carving = p.enable_hand_occlusion
        if hand_strength is not None:
            p.hand_occlusion_strength = float(np.clip(hand_strength, 0.0, 1.0))
            det.hand_carving_strength = p.hand_occlusion_strength
            det.hand_engine.sensitivity = p.hand_occlusion_strength
        if visualize_hud is not None:
            p.visualize_occlusion_hud = bool(visualize_hud)
            det.visualize_hud = p.visualize_occlusion_hud

        return {
            "success": True,
            "occlusion": {
                "sensitivity": det.sensitivity,
                "enable_spectacles": det.enable_spectacles,
                "spectacles_strength": det.spectacles_strength,
                "enable_hand_carving": det.enable_hand_carving,
                "hand_carving_strength": det.hand_carving_strength,
                "visualize_hud": det.visualize_hud,
            }
        }

    @app.post("/api/capture")
    async def capture_frame():
        """Saves current transformed frame to disk as a high-quality JPEG snapshot."""
        runner = get_web_runner()
        with runner._lock:
            res = runner._latest_result

        if res is None or res.rendered_frame is None:
            raise HTTPException(status_code=503, detail="No active frame available to capture.")

        target_id = res.target.person_id if res.target else "preview"
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"studio_capture_{ts}_{target_id}.jpg"
        file_path = os.path.join(CAPTURES_DIR, filename)

        cv2.imwrite(file_path, res.rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return {
            "success": True,
            "filename": filename,
            "url": f"/captures/{filename}",
            "timestamp": ts,
        }

    @app.post("/api/recording/start")
    async def start_recording(fps: float = Query(30.0), width: int = Query(1280), height: int = Query(720)):
        """Starts asynchronous recording of live swapped video stream."""
        runner = get_web_runner()
        res = runner.start_recording(fps=fps, resolution=(width, height))
        return res

    @app.post("/api/recording/stop")
    async def stop_recording():
        """Finalizes active video recording and returns metadata & download link."""
        runner = get_web_runner()
        meta = runner.stop_recording()
        return meta

    @app.get("/api/recording/status")
    async def get_recording_status():
        """Returns live frame counts and duration of active recording."""
        runner = get_web_runner()
        return runner.get_recording_status()

    @app.get("/api/recordings")
    async def list_recordings():
        """Lists all completed MP4 video recordings."""
        items = []
        if os.path.isdir(RECORDINGS_DIR):
            for fn in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
                if fn.lower().endswith(".mp4"):
                    fp = os.path.join(RECORDINGS_DIR, fn)
                    meta_path = os.path.splitext(fp)[0] + ".json"
                    meta = {}
                    if os.path.isfile(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                        except Exception:
                            pass
                    items.append({
                        "filename": fn,
                        "url": f"/recordings/{fn}",
                        "size_mb": round(os.path.getsize(fp) / (1024 * 1024), 2),
                        "modified_at": datetime.datetime.fromtimestamp(os.path.getmtime(fp)).isoformat(),
                        "metadata": meta,
                    })
        return {"recordings": items}

    @app.post("/api/source/toggle")
    async def toggle_source():
        """Toggles between hardware webcam and animated synthetic test mode."""
        runner = get_web_runner()
        runner._use_synthetic_source = not runner._use_synthetic_source
        mode = "Synthetic Video Loop" if runner._use_synthetic_source else "Physical Webcam"
        logger.info(f"Source switched to {mode}")
        return {"success": True, "is_synthetic": runner._use_synthetic_source, "mode": mode}

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        """Zero-polling WebSocket telemetry stream for real-time Web Studio HUD."""
        await websocket.accept()
        runner = get_web_runner()
        runner.register_client()
        try:
            import asyncio
            while True:
                telemetry = runner.get_telemetry()
                await websocket.send_json(telemetry)
                await asyncio.sleep(0.066)
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            runner.unregister_client()

    return app


def run_web_studio(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launches the Uvicorn ASGI server."""
    import uvicorn
    app = create_app()
    logger.info(f"Starting Real-Time Web Studio at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
