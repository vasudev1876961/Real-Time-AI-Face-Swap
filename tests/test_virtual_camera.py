"""
Unit and Integration Tests for Virtual Camera and HTTP MJPEG Live Stream Broadcaster.
"""

import time
import urllib.request
import json
import pytest
import numpy as np
import cv2

from src.camera.virtual_camera import VirtualCameraBroadcaster


def test_broadcaster_lifecycle_and_http_endpoints():
    # Use ephemeral port to avoid conflict
    test_port = 18088
    broadcaster = VirtualCameraBroadcaster(
        port=test_port,
        enable_virtualcam=False,  # Skip directshow virtualcam in headless test
        enable_mjpeg=True,
    )

    assert not broadcaster.is_active()
    assert broadcaster.start(width=320, height=240, fps=30)
    assert broadcaster.is_active()
    assert f":{test_port}/stream" in broadcaster.get_stream_url()

    # Send test frame
    test_frame = np.full((240, 320, 3), 180, dtype=np.uint8)
    broadcaster.send_frame(test_frame)

    time.sleep(0.2)  # Allow HTTP server to bind

    try:
        # Test status endpoint
        status_url = f"http://127.0.0.1:{test_port}/status"
        req = urllib.request.urlopen(status_url, timeout=3.0)
        assert req.status == 200
        data = json.loads(req.read().decode("utf-8"))
        assert data.get("status") == "broadcasting"
        assert data.get("width") == 320
        assert data.get("height") == 240

        # Test HTML preview page
        web_url = f"http://127.0.0.1:{test_port}/"
        web_req = urllib.request.urlopen(web_url, timeout=3.0)
        assert web_req.status == 200
        html_content = web_req.read().decode("utf-8")
        assert "AI Face Swap" in html_content

    finally:
        broadcaster.stop()
        assert not broadcaster.is_active()


def test_broadcaster_frame_handling():
    broadcaster = VirtualCameraBroadcaster(enable_virtualcam=False, enable_mjpeg=True)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Sending frame when inactive should not raise errors
    broadcaster.send_frame(frame)
    assert broadcaster.get_latest_jpeg() is None
