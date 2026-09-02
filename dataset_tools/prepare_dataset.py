"""
Comprehensive Dataset Preparation Pipeline.
Coordinates scanning, validation, extraction, deduplication, and metadata generation.
"""

import os
import argparse
from typing import Dict, Any

from dataset_tools.scan_dataset import scan_raw_dataset
from dataset_tools.validate_faces import validate_and_filter_images
from dataset_tools.extract_faces import extract_dataset_faces
from dataset_tools.remove_duplicates import detect_and_flag_duplicates
from dataset_tools.create_metadata import generate_metadata_csvs
from src.utils.logger import get_logger, setup_logging

logger = get_logger("PrepareDataset")


def run_dataset_pipeline(
    raw_dir: str = "datasets/raw",
    processed_dir: str = "datasets/processed",
    rejected_dir: str = "datasets/rejected",
    metadata_dir: str = "datasets/metadata",
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Executes all dataset preparation stages in sequence.
    """
    logger.info("=" * 60)
    logger.info(f"RUNNING DATASET PREPARATION PIPELINE (Mode: {'DRY RUN' if dry_run else 'APPLY'})")
    logger.info("=" * 60)

    # Step 1: Scan
    logger.info("\n--- STEP 1: SCANNING DATASET ---")
    scan_rep = scan_raw_dataset(raw_dir)

    # Step 2: Validate & Filter
    logger.info("\n--- STEP 2: QUALITY VALIDATION ---")
    val_rep = validate_and_filter_images(
        input_dir=raw_dir,
        rejected_dir=rejected_dir,
        metadata_csv=os.path.join(metadata_dir, "rejection_report.csv"),
        dry_run=dry_run,
    )

    # Step 3: Extract & Align (Only during apply mode)
    extracted_count = 0
    if not dry_run:
        logger.info("\n--- STEP 3: FACE EXTRACTION & ALIGNMENT ---")
        extracted_count = extract_dataset_faces(
            input_dir=raw_dir,
            output_dir=processed_dir,
            crop_size=(512, 512),
        )
    else:
        logger.info("\n--- STEP 3: FACE EXTRACTION (Skipped in Dry-Run) ---")

    # Step 4: Duplicate Detection
    logger.info("\n--- STEP 4: PERCEPTUAL DUPLICATE DETECTION ---")
    dup_rep = detect_and_flag_duplicates(
        input_dir=raw_dir,
        rejected_dir=os.path.join(rejected_dir, "duplicates"),
        report_csv=os.path.join(metadata_dir, "duplicate_report.csv"),
        dry_run=dry_run,
    )

    # Step 5: Metadata Generation
    logger.info("\n--- STEP 5: METADATA CATALOG GENERATION ---")
    meta_rep = generate_metadata_csvs(raw_dir=raw_dir, metadata_dir=metadata_dir)

    logger.info("\n" + "=" * 60)
    logger.info("DATASET PIPELINE COMPLETE")
    logger.info(f"Valid Raw Images: {scan_rep.valid_images}")
    logger.info(f"Rejected / Flagged: {len(val_rep.rejections)}")
    logger.info(f"Extracted Faces: {extracted_count}")
    logger.info(f"Duplicates Flagged: {len(dup_rep)}")
    logger.info(f"Metadata Catalogs: {meta_rep}")
    logger.info("=" * 60)

    return {
        "scan": scan_rep,
        "validation": val_rep,
        "extracted_count": extracted_count,
        "duplicates_count": len(dup_rep),
        "metadata_counts": meta_rep,
    }


def main():
    parser = argparse.ArgumentParser(description="Run complete dataset preparation pipeline.")
    parser.add_argument("--input", "-i", type=str, default="datasets/raw", help="Raw dataset input directory")
    parser.add_argument("--output", "-o", type=str, default="datasets/processed", help="Processed output directory")
    parser.add_argument("--rejected", "-r", type=str, default="datasets/rejected", help="Rejected directory")
    parser.add_argument("--metadata", "-m", type=str, default="datasets/metadata", help="Metadata directory")
    parser.add_argument("--apply", action="store_true", help="Execute changes (Default is dry-run)")
    args = parser.parse_args()

    setup_logging()
    run_dataset_pipeline(
        raw_dir=args.input,
        processed_dir=args.output,
        rejected_dir=args.rejected,
        metadata_dir=args.metadata,
        dry_run=not args.apply,
    )


if __name__ == "__main__":
    main()
