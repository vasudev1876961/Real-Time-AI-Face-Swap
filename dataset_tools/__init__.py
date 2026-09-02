"""
Dataset Management, Face Extraction, Migration, and Metadata Generation Tools.
"""

from dataset_tools.scan_dataset import scan_raw_dataset, ScanReport
from dataset_tools.validate_faces import validate_and_filter_images, ValidationReport
from dataset_tools.extract_faces import extract_dataset_faces
from dataset_tools.remove_duplicates import detect_and_flag_duplicates
from dataset_tools.create_metadata import generate_metadata_csvs
from dataset_tools.migrate_dataset import migrate_legacy_dataset
from dataset_tools.prepare_dataset import run_dataset_pipeline

__all__ = [
    "scan_raw_dataset",
    "ScanReport",
    "validate_and_filter_images",
    "ValidationReport",
    "extract_dataset_faces",
    "detect_and_flag_duplicates",
    "generate_metadata_csvs",
    "migrate_legacy_dataset",
    "run_dataset_pipeline",
]
