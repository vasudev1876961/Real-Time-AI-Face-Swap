"""
Camera subsystem for multi-backend frame acquisition and device management.
"""

from src.camera.camera_backend import CameraBackend, FramePacket
from src.camera.camera_manager import CameraManager, CameraDeviceInfo

__all__ = [
    "CameraBackend",
    "FramePacket",
    "CameraManager",
    "CameraDeviceInfo",
]
