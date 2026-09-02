"""
Unit Tests for Dataset Tools, Validation, Deduplication, and Metadata Generation.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import cv2

from dataset_tools.scan_dataset import scan_raw_dataset
from dataset_tools.validate_faces import validate_and_filter_images
from dataset_tools.remove_duplicates import detect_and_flag_duplicates
from dataset_tools.create_metadata import generate_metadata_csvs
from dataset_tools.migrate_dataset import migrate_legacy_dataset
from src.utils.image_utils import calculate_blur_score, compute_phash


@pytest.fixture
def temp_dataset_dir():
    temp_dir = tempfile.mkdtemp()
    raw_dir = os.path.join(temp_dir, "raw", "telugu_heroes", "hero_01")
    os.makedirs(raw_dir, exist_ok=True)

    # Create sharp test image with face
    img_sharp = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img_sharp, (100, 100), 50, (200, 200, 200), -1)
    cv2.imwrite(os.path.join(raw_dir, "sharp.jpg"), img_sharp)

    # Create blurry image
    img_blurry = cv2.GaussianBlur(img_sharp, (25, 25), 10)
    cv2.imwrite(os.path.join(raw_dir, "blurry.jpg"), img_blurry)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_scan_dataset(temp_dataset_dir):
    raw_path = os.path.join(temp_dataset_dir, "raw")
    rep = scan_raw_dataset(raw_path, min_dimension=64)
    assert rep.valid_images == 2
    assert rep.total_files_scanned == 2


def test_blur_calculation():
    img_sharp = np.zeros((100, 100, 3), dtype=np.uint8)
    img_sharp[25:75, 25:75] = 255
    score_sharp = calculate_blur_score(img_sharp)

    img_blurry = cv2.GaussianBlur(img_sharp, (15, 15), 5)
    score_blurry = calculate_blur_score(img_blurry)

    assert score_sharp > score_blurry


def test_duplicate_detection(temp_dataset_dir):
    raw_path = os.path.join(temp_dataset_dir, "raw")
    report_csv = os.path.join(temp_dataset_dir, "dup_report.csv")
    dups = detect_and_flag_duplicates(input_dir=raw_path, report_csv=report_csv, dry_run=True)
    assert os.path.exists(report_csv)


def test_metadata_generation(temp_dataset_dir):
    raw_path = os.path.join(temp_dataset_dir, "raw")
    meta_dir = os.path.join(temp_dataset_dir, "metadata")
    counts = generate_metadata_csvs(raw_dir=raw_path, metadata_dir=meta_dir)
    assert "telugu_heroes" in counts
    assert os.path.exists(os.path.join(meta_dir, "telugu_heroes.csv"))
