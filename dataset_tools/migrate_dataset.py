"""
Legacy Dataset Migration and Categorization Tool.
Audits existing dataset folders (e.g. TFI-Faces) and prepares non-destructive migrations.
"""

import os
import csv
import shutil
import argparse
from typing import List, Dict
from tqdm import tqdm

from src.utils.logger import get_logger, setup_logging

logger = get_logger("DatasetMigration")

MIGRATION_FIELDS = [
    "original_path",
    "detected_folder",
    "proposed_category",
    "proposed_person_id",
    "action",
    "reason",
    "confidence",
]

# Known Telugu Cinema Heroes list for category mapping assistance without appearance guessing
TELUGU_HERO_NAMES = {
    "chiranjeevi", "nagarjuna", "venkatesh", "balakrishna", "prabhas",
    "mahesh_babu", "pawan_kalyan", "allu_arjun", "ram_charan", "jr_ntr",
    "ntr", "nani", "vijay_deverakonda", "ram_pothineni", "ravi_teja",
    "rana_daggubati", "naga_chaitanya", "akhil_akkineni", "varun_tej",
    "sai_dharam_tej", "sharwanand", "gopichand", "allari_naresh"
}

# Known Actress name hints for category separation assistance
ACTRESS_NAME_HINTS = {
    "anushka_shetty", "deepika_padukone", "deepti_naval", "dimple_kapadia",
    "farida_jalal", "huma_qureshi", "kajol", "kalki_koechlin", "kangana_ranaut",
    "kareena_kapoor", "karisma_kapoor", "katrina_kaif", "madhuri_dixit",
    "preity_zinta", "priyanka_chopra", "rani_mukerji", "rekha", "sridevi",
    "vidya_balan", "tabu", "zeenat_aman", "samantha", "rashmika", "pooja_hegde",
    "sai_pallavi", "keerthy_suresh", "tamannaah", "nayanthara", "kajal_aggarwal",
    "aruna_irani"
}


def migrate_legacy_dataset(
    source_root: str = "TFI-Faces",
    target_raw_root: str = "datasets/raw",
    report_csv: str = "datasets/metadata/migration_report.csv",
    dry_run: bool = True,
) -> List[Dict[str, str]]:
    """
    Scans legacy folder structure and creates non-destructive migration plan.
    """
    os.makedirs(os.path.dirname(report_csv), exist_ok=True)
    report_rows: List[Dict[str, str]] = []

    if not os.path.exists(source_root):
        logger.warning(f"Source directory '{source_root}' not found.")
        return report_rows

    logger.info(f"Auditing source directory '{source_root}' (dry-run: {dry_run})...")

    # Discover person folders
    discovered_folders = []
    for root, dirs, files in os.walk(source_root):
        # If folder contains images directly, it's a person directory
        has_images = any(f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) for f in files)
        if has_images:
            discovered_folders.append((root, files))

    logger.info(f"Discovered {len(discovered_folders)} identity folders in '{source_root}'.")

    for folder_path, files in tqdm(discovered_folders, desc="Analyzing folders"):
        folder_name = os.path.basename(folder_path.rstrip("\\/")).lower()
        person_id = folder_name.replace(" ", "_").replace("-", "_")

        # Determine proposed category using explicit name matching rules
        if person_id in TELUGU_HERO_NAMES:
            category = "telugu_heroes"
            reason = "Matches verified Telugu Hero list entry"
            confidence = "HIGH"
        elif person_id in ACTRESS_NAME_HINTS:
            category = "actresses"
            reason = "Matches verified Actress list entry"
            confidence = "HIGH"
        else:
            category = "actors"
            reason = "Standard actor classification (requires manual verification)"
            confidence = "MEDIUM"

        action = "REVIEW" if dry_run else "COPY"

        report_rows.append({
            "original_path": folder_path,
            "detected_folder": folder_name,
            "proposed_category": category,
            "proposed_person_id": person_id,
            "action": action,
            "reason": reason,
            "confidence": confidence,
        })

        if not dry_run:
            dest_dir = os.path.join(target_raw_root, category, person_id)
            os.makedirs(dest_dir, exist_ok=True)
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    src_f = os.path.join(folder_path, f)
                    dst_f = os.path.join(dest_dir, f)
                    if not os.path.exists(dst_f):
                        shutil.copy2(src_f, dst_f)

    # Save Migration Report CSV
    try:
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MIGRATION_FIELDS)
            writer.writeheader()
            writer.writerows(report_rows)
        logger.info(f"Saved migration report to '{report_csv}'.")
    except Exception as e:
        logger.error(f"Failed to write migration report: {e}")

    logger.info(f"Migration Audit Complete: Processed {len(report_rows)} folders.")
    return report_rows


def main():
    parser = argparse.ArgumentParser(description="Audit and safely migrate legacy datasets.")
    parser.add_argument("--source", "-s", type=str, default="TFI-Faces", help="Source legacy dataset directory")
    parser.add_argument("--target", "-t", type=str, default="datasets/raw", help="Target datasets/raw directory")
    parser.add_argument("--report", "-r", type=str, default="datasets/metadata/migration_report.csv", help="Report CSV")
    parser.add_argument("--apply", action="store_true", help="Execute copy operations (Default is dry-run)")
    args = parser.parse_args()

    setup_logging()
    migrate_legacy_dataset(
        source_root=args.source,
        target_raw_root=args.target,
        report_csv=args.report,
        dry_run=not args.apply,
    )


if __name__ == "__main__":
    main()
