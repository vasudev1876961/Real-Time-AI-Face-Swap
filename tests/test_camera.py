"""
Unit Tests for Camera Subsystem and Device Manager.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.camera.camera_backend import CameraBackend, FramePacket
from src.camera.camera_manager import CameraManager, CameraDeviceInfo
from src.core.config_loader import CameraConfig
from src.core.device import get_device_manager, get_execution_providers


def test_device_manager_initialization():
    dm = get_device_manager()
    assert dm is not None
    assert dm.active_provider in ["CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider"]
    stats = dm.get_system_stats()
    assert "cpu_percent" in stats
    assert "ram_percent" in stats
    assert "active_provider" in stats


def test_frame_packet_creation():
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(frame=dummy, frame_index=1, timestamp=123456.78, width=640, height=480)
    assert packet.width == 640
    assert packet.height == 480
    assert packet.frame_index == 1
    assert packet.frame.shape == (480, 640, 3)


def test_camera_discovery():
    cams = CameraManager.discover_cameras(max_probe=2)
    assert isinstance(cams, list)
    for c in cams:
        assert isinstance(c, CameraDeviceInfo)
        assert isinstance(c.index, int)
        assert isinstance(c.name, str)


def test_camera_manager_callback():
    mgr = CameraManager()
    callback_called = []

    def on_switched(idx):
        callback_called.append(idx)

    mgr.register_switch_callback(on_switched)

    # Trigger switch callback
    for cb in mgr._on_camera_switched_callbacks:
        cb(1)

    assert callback_called == [1]
