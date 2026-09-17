"""
Virtual Camera and Zero-Config Live Network Streaming Broadcaster.
Enables broadcasting transformed face-swap frames directly to virtual camera devices
(OBS VirtualCam / DirectShow via pyvirtualcam) and network endpoints (HTTP MJPEG stream).
"""

import time
import threading
import io
from typing import Optional, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("VirtualCamera")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server allowing concurrent stream client connections."""
    daemon_threads = True
    allow_reuse_address = True


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving live MJPEG stream and status dashboard."""

    server_broadcaster: Any = None  # Reference injected by VirtualCameraBroadcaster

    def log_message(self, format, *args):
        """Suppress standard HTTP request logging spam."""
        pass

    def do_GET(self):
        broadcaster = getattr(self.server, "broadcaster", None) or MJPEGStreamHandler.server_broadcaster
        if broadcaster is None:
            self.send_error(500, "Broadcaster Not Initialized")
            return

        if self.path in ("/stream", "/video_feed"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            broadcaster.register_client()
            try:
                while broadcaster.is_active():
                    jpeg_bytes = broadcaster.get_latest_jpeg()
                    if jpeg_bytes is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg_bytes)))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(1.0 / max(1, broadcaster.fps))
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                broadcaster.unregister_client()

        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status_json = (
                f'{{"status": "broadcasting", "clients": {broadcaster.active_clients}, '
                f'"fps": {broadcaster.fps}, "width": {broadcaster.width}, "height": {broadcaster.height}}}'
            )
            self.wfile.write(status_json.encode("utf-8"))

        else:
            # Clean embedded HTML Web Previewer
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Face Swap Live Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ margin: 0; background: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; text-align: center; }}
        header {{ padding: 18px 20px; background: #1e293b; border-bottom: 2px solid #3b82f6; }}
        h1 {{ margin: 0; font-size: 20px; font-weight: 600; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; background: #22c55e; color: black; font-size: 11px; font-weight: bold; margin-left: 8px; }}
        .container {{ max-width: 1280px; margin: 24px auto; padding: 0 16px; }}
        .feed-card {{ background: #1e293b; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }}
        img {{ width: 100%; max-height: 80vh; object-fit: contain; display: block; }}
        .info {{ padding: 12px; font-size: 13px; color: #94a3b8; display: flex; justify-content: space-around; }}
    </style>
</head>
<body>
    <header>
        <h1>⚡ Real-Time AI Face Swap <span class="badge">LIVE BROADCAST</span></h1>
    </header>
    <div class="container">
        <div class="feed-card">
            <img src="/stream" alt="Live Stream Feed" />
            <div class="info">
                <span>Resolution: {broadcaster.width}x{broadcaster.height}</span>
                <span>Target: {broadcaster.fps} FPS</span>
                <span>Active Viewers: {broadcaster.active_clients}</span>
                <span>OBS Source: <code>http://localhost:{broadcaster.port}/stream</code></span>
            </div>
        </div>
    </div>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))


class VirtualCameraBroadcaster:
    """
    Coordinates hardware virtual camera streaming (via pyvirtualcam / DirectShow)
    and network MJPEG server streaming.
    """

    def __init__(
        self,
        port: int = 8080,
        enable_virtualcam: bool = True,
        enable_mjpeg: bool = True,
    ):
        self.port = port
        self.enable_virtualcam = enable_virtualcam
        self.enable_mjpeg = enable_mjpeg

        self.width = 1280
        self.height = 720
        self.fps = 30

        self._is_active = False
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._active_clients = 0

        # Virtual camera device handle (pyvirtualcam)
        self._vcam_device = None
        self._vcam_available = False

        # HTTP MJPEG server
        self._httpd: Optional[ThreadedHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

    @property
    def active_clients(self) -> int:
        return self._active_clients

    def register_client(self) -> None:
        with self._lock:
            self._active_clients += 1

    def unregister_client(self) -> None:
        with self._lock:
            self._active_clients = max(0, self._active_clients - 1)

    def is_active(self) -> bool:
        return self._is_active

    def get_stream_url(self) -> str:
        """Returns the local network URL for accessing the MJPEG video stream."""
        return f"http://127.0.0.1:{self.port}/stream"

    def get_web_url(self) -> str:
        """Returns the browser preview URL."""
        return f"http://127.0.0.1:{self.port}/"

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def start(self, width: int = 1280, height: int = 720, fps: int = 30) -> bool:
        """Initializes virtual camera broadcaster and HTTP streaming server."""
        if self._is_active:
            return True

        self.width = width
        self.height = height
        self.fps = fps
        self._is_active = True

        # 1. Initialize pyvirtualcam if requested and available
        if self.enable_virtualcam:
            try:
                import pyvirtualcam
                self._vcam_device = pyvirtualcam.Camera(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    fmt=pyvirtualcam.PixelFormat.BGR,
                )
                self._vcam_available = True
                logger.info(f"Initialized Virtual Camera device ({self._vcam_device.device}).")
            except ImportError:
                logger.info("pyvirtualcam module not found. DirectShow virtual camera skipped.")
                self._vcam_available = False
            except Exception as e:
                logger.warning(f"Could not open pyvirtualcam device: {e}")
                self._vcam_available = False

        # 2. Initialize HTTP MJPEG server
        if self.enable_mjpeg:
            try:
                MJPEGStreamHandler.server_broadcaster = self
                self._httpd = ThreadedHTTPServer(("0.0.0.0", self.port), MJPEGStreamHandler)
                self._httpd.broadcaster = self
                self._http_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
                self._http_thread.start()
                logger.info(f"MJPEG Live Stream Server active at {self.get_stream_url()}")
            except Exception as e:
                logger.warning(f"Could not start HTTP stream server on port {self.port}: {e}")

        return True

    def send_frame(self, frame: np.ndarray) -> None:
        """Broadcasts a newly transformed video frame to virtual camera and network stream."""
        if not self._is_active or frame is None or frame.size == 0:
            return

        # 1. Update pyvirtualcam device
        if self._vcam_device is not None and self._vcam_available:
            try:
                # Resize if frame resolution differs from device negotiation
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    v_frame = cv2.resize(frame, (self.width, self.height))
                else:
                    v_frame = frame
                self._vcam_device.send(v_frame)
                self._vcam_device.sleep_until_next_frame()
            except Exception as e:
                logger.debug(f"Virtual camera send error: {e}")

        # 2. Encode to JPEG for HTTP MJPEG stream (only encode if clients are listening or for buffer)
        if self.enable_mjpeg:
            try:
                # Encode with optimized quality/speed trade-off (quality 80)
                success, encoded_img = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    jpeg_bytes = encoded_img.tobytes()
                    with self._lock:
                        self._latest_jpeg = jpeg_bytes
            except Exception as e:
                logger.debug(f"JPEG encode error: {e}")

    def stop(self) -> None:
        """Safely terminates all streaming threads and releases hardware handles."""
        if not self._is_active:
            return

        self._is_active = False

        # Close pyvirtualcam
        if self._vcam_device is not None:
            try:
                self._vcam_device.close()
            except Exception:
                pass
            self._vcam_device = None

        # Shutdown HTTP server
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None

        self._http_thread = None
        logger.info("Virtual camera broadcaster stopped.")
