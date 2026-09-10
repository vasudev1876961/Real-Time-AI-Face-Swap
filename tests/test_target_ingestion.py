"""
Unit Tests for Custom Target Face Ingestion.
"""

import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest

from src.targets.target_manager import TargetManager
from src.core.config_loader import TargetsConfig


def test_add_custom_target_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_faces_root:
        cfg = TargetsConfig(target_root=tmp_faces_root)
        tm = TargetManager(cfg)

        # Create a test image with a recognizable face-like structure
        test_img_path = "outputs/captures/capture_2026-08-24_21-47-58_hero_001.jpg"
        if not os.path.isfile(test_img_path):
            pytest.skip("Test portrait capture image not found.")

        success, msg, target = tm.add_custom_target(
            image_or_path=test_img_path,
            display_name="Mahesh Babu Custom",
            category="telugu_heroes",
            person_id="mahesh_babu_custom",
        )

        assert success, f"Import failed: {msg}"
        assert target is not None
        assert target.target_id == "mahesh_babu_custom"
        assert target.category == "telugu_heroes"
        assert target.embedding.shape == (512,)

        # Verify filesystem layout
        expected_dir = os.path.join(tmp_faces_root, "telugu_heroes", "mahesh_babu_custom")
        assert os.path.isdir(expected_dir)
        assert os.path.isfile(os.path.join(expected_dir, "reference.jpg"))
        assert os.path.isfile(os.path.join(expected_dir, "face.npy"))
        assert os.path.isfile(os.path.join(expected_dir, "metadata.json"))

        # Verify listed in target manager
        found = tm.get_target("mahesh_babu_custom")
        assert found is not None
        assert found.display_name == "Mahesh Babu Custom"
