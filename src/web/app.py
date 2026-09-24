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

        while self._running:
            t0 = time.perf_counter()
            frame = None

            if not self._use_synthetic_source and self.camera_manager.is_running():
                pkt = self.camera_manager.get_latest_frame()
                frame = pkt.frame if pkt is not None else None
                if frame is None:
                    # Camera dropped frame, retry or fall back
                    time.sleep(0.005)
                    continue
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

        dump_dict = config.model_dump(exclude_unset=True) if hasattr(config, "model_dump") else config.dict(exclude_unset=True)
        return {"success": True, "updated": dump_dict}

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
