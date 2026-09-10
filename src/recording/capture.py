"""
Snapshot and Image Capture Subsystem.
Handles high-resolution frame snapshots, safe image serialization, and metadata sidecar logging.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

from src.utils.image_utils import write_image_safe
from src.utils.logger import get_logger

logger = get_logger("SnapshotCapture")


class SnapshotCaptureManager:
    """
    Manages structured frame snapshot captures with metadata sidecars.
    """

    def __init__(self, default_captures_dir: str = "outputs/captures"):
        self.captures_dir = default_captures_dir
        os.makedirs(self.captures_dir, exist_ok=True)

    def capture(
        self,
        frame: np.ndarray,
        target_name: Optional[str] = None,
        target_id: Optional[str] = None,
        category: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Saves the current rendered frame to the captures directory with timestamp and JSON metadata.
        Returns the absolute filepath of the saved image, or None if failed.
        """
        if frame is None or frame.size == 0:
            logger.error("Cannot capture: Frame is empty.")
            return None

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_target = (target_id or "raw").replace(" ", "_").lower()
        filename = f"capture_{timestamp}_{safe_target}.jpg"
        meta_filename = f"capture_{timestamp}_{safe_target}.json"

        img_path = os.path.join(self.captures_dir, filename)
        meta_path = os.path.join(self.captures_dir, meta_filename)

        if not write_image_safe(img_path, frame):
            logger.error(f"Failed to write capture image to: {img_path}")
            return None

        h, w = frame.shape[:2]
        meta = {
            "captured_at": datetime.now().isoformat(),
            "filename": filename,
            "resolution": f"{w}x{h}",
            "target_id": target_id or "none",
            "target_name": target_name or "none",
            "category": category or "none",
        }
        if extra_metadata:
            meta.update(extra_metadata)

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save snapshot metadata: {e}")

        logger.info(f"Snapshot saved: {img_path}")
        return img_path
