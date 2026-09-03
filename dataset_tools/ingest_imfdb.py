"""
IMFDB Dataset Ingestion Tool.
Transfers and categorizes images from the downloaded Kaggle IMFDB dataset
into the structured datasets/raw directory across telugu_heroes, actresses, and actors.
"""

import os
import sys
import shutil
import argparse
from typing import Dict, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger, setup_logging

logger = get_logger("IMFDBIngestion")

DEFAULT_KAGGLE_PATH = os.path.expanduser(
    r"~/.cache/kagglehub/datasets/anirudhsimhachalam/indian-movie-faces-datasetimfdb-face-recognition/versions/1/IMFDB FR dataset/IMFDB FR dataset"
)

# Canonical mapping for all 100 identities in IMFDB
IMFDB_IDENTITY_MAP: Dict[str, Tuple[str, str]] = {
    # --- TELUGU HEROES (Tollywood Leads) ---
    "ANR": ("telugu_heroes", "anr"),
    "NTR": ("telugu_heroes", "ntr"),
    "Balakrishna": ("telugu_heroes", "balakrishna"),
    "Nagarjuna": ("telugu_heroes", "nagarjuna"),
    "Venkatesh": ("telugu_heroes", "venkatesh"),
    "JagapathiBabu": ("telugu_heroes", "jagapathi_babu"),
    "Srihari": ("telugu_heroes", "srihari"),

    # --- ACTRESSES ---
    "Aarthi": ("actresses", "aarthi"),
    "Annapoorna": ("actresses", "annapoorna"),
    "Bhanupriya": ("actresses", "bhanupriya"),
    "Bharathi": ("actresses", "bharathi"),
    "FaridaJalal": ("actresses", "farida_jalal"),
    "Jamuna": ("actresses", "jamuna"),
    "JayaBhaduri": ("actresses", "jaya_bhaduri"),
    "KajalAgarwal": ("actresses", "kajal_aggarwal"),
    "Kajol": ("actresses", "kajol"),
    "KareenaKapoor": ("actresses", "kareena_kapoor"),
    "KarunaBenerjee": ("actresses", "karuna_banerjee"),
    "KatrinaKaif": ("actresses", "katrina_kaif"),
    "KavyaMadhavan": ("actresses", "kavya_madhavan"),
    "Lakshmidevi": ("actresses", "lakshmidevi"),
    "Leelavathi": ("actresses", "leelavathi"),
    "MadhabiMukherjee": ("actresses", "madhabi_mukherjee"),
    "MadhuriDixit": ("actresses", "madhuri_dixit"),
    "Pavithralokesh": ("actresses", "pavithra_lokesh"),
    "Prema": ("actresses", "prema"),
    "RakheeGulzar": ("actresses", "rakhee_gulzar"),
    "Ramaprabha": ("actresses", "ramaprabha"),
    "RamyaKrishna": ("actresses", "ramya_krishnan"),
    "Ranimukherji": ("actresses", "rani_mukerji"),
    "Rimisen": ("actresses", "rimi_sen"),
    "Savithri": ("actresses", "savithri"),
    "SharmilaTagore": ("actresses", "sharmila_tagore"),
    "ShilpaShetty": ("actresses", "shilpa_shetty"),
    "Shobana": ("actresses", "shobana"),
    "Simran": ("actresses", "simran"),
    "Soundarya": ("actresses", "soundarya"),
    "Suryakantham": ("actresses", "suryakantham"),
    "Trisha": ("actresses", "trisha"),
    "Umashri": ("actresses", "umashri"),

    # --- ACTORS ---
    "AamairKhan": ("actors", "aamir_khan"),
    "AkshayKumar": ("actors", "akshay_kumar"),
    "Ali": ("actors", "ali"),
    "Ambresh": ("actors", "ambareesh"),
    "AmitabhBachchan": ("actors", "amitabh_bachchan"),
    "AmrishPuri": ("actors", "amrish_puri"),
    "AnilKapoor": ("actors", "anil_kapoor"),
    "AnupamKher": ("actors", "anupam_kher"),
    "Ashwath": ("actors", "ks_ashwath"),
    "Avinash": ("actors", "avinash"),
    "BabuMohan": ("actors", "babu_mohan"),
    "BomanIrani": ("actors", "boman_irani"),
    "Brahmanandam": ("actors", "brahmanandam"),
    "Cochinhaneefa": ("actors", "cochin_haneefa"),
    "Dileep": ("actors", "dileep"),
    "Dr.Rajkumar": ("actors", "dr_rajkumar"),
    "Dwarkish": ("actors", "dwarakish"),
    "HrithikRoshan": ("actors", "hrithik_roshan"),
    "Innocent": ("actors", "innocent"),
    "Jagadeesh": ("actors", "jagadeesh"),
    "Jagathi": ("actors", "jagathy_sreekumar"),
    "Jayaprakash": ("actors", "jayaprakash"),
    "Jayram": ("actors", "jayaram"),
    "JosePrakash": ("actors", "jose_prakash"),
    "K.Viswanath": ("actors", "k_viswanath"),
    "KotaSrinivasarao": ("actors", "kota_srinivasa_rao"),
    "Loknath": ("actors", "loknath"),
    "M.S.Narayana": ("actors", "ms_narayana"),
    "Madhavan": ("actors", "madhavan"),
    "Madhu": ("actors", "madhu"),
    "Mallikarjunrao": ("actors", "mallikarjuna_rao"),
    "Mammukoya": ("actors", "mammukoya"),
    "Mamootty": ("actors", "mammootty"),
    "Mohanlal": ("actors", "mohanlal"),
    "Mukesh": ("actors", "mukesh"),
    "NedumudiVenu": ("actors", "nedumudi_venu"),
    "PareshRaval": ("actors", "paresh_rawal"),
    "PrakashRaj": ("actors", "prakash_raj"),
    "PremaNazir": ("actors", "prem_nazir"),
    "RajeshKhanna": ("actors", "rajesh_khanna"),
    "RamanaReddy": ("actors", "ramana_reddy"),
    "RameshArvind": ("actors", "ramesh_aravind"),
    "Rameshbhatt": ("actors", "ramesh_bhat"),
    "Relangi": ("actors", "relangi"),
    "RishiKapoor": ("actors", "rishi_kapoor"),
    "SVR": ("actors", "sv_ranga_rao"),
    "SailendraMukherjee": ("actors", "sailen_mukherjee"),
    "SalmanKhan": ("actors", "salman_khan"),
    "SharukhKhan": ("actors", "shah_rukh_khan"),
    "Shashikumar": ("actors", "shashikumar"),
    "Shivaram": ("actors", "shivaram"),
    "Siddique": ("actors", "siddique"),
    "SoumithraChatterjee": ("actors", "soumitra_chatterjee"),
    "SureshGopi": ("actors", "suresh_gopi"),
    "TanikellaBharani": ("actors", "tanikella_bharani"),
    "Tenniskrishna": ("actors", "tennis_krishna"),
    "Thilakan": ("actors", "thilakan"),
    "Vajramuni": ("actors", "vajramuni"),
    "VinodKhanna": ("actors", "vinod_khanna"),
    "Vishnuvardhan": ("actors", "vishnuvardhan"),
}


def ingest_imfdb(
    source_root: str = DEFAULT_KAGGLE_PATH,
    target_root: str = "datasets/raw",
    dry_run: bool = True,
) -> Dict[str, int]:
    """
    Ingests IMFDB images into target_root categorized as telugu_heroes, actresses, or actors.
    """
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"Source directory '{source_root}' not found.")

    logger.info(f"Scanning source directory '{source_root}' (Dry-Run: {dry_run})...")
    folders = [f for f in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, f))]
    logger.info(f"Found {len(folders)} identity folders in IMFDB source.")

    stats = {
        "telugu_heroes": 0,
        "actresses": 0,
        "actors": 0,
        "total_copied": 0,
        "skipped_existing": 0,
    }

    for folder_name in tqdm(folders, desc="Ingesting identities"):
        if folder_name in IMFDB_IDENTITY_MAP:
            category, person_id = IMFDB_IDENTITY_MAP[folder_name]
        else:
            # Fallback
            category, person_id = "actors", folder_name.lower()

        src_folder = os.path.join(source_root, folder_name)
        dst_folder = os.path.join(target_root, category, person_id)

        if not dry_run:
            os.makedirs(dst_folder, exist_ok=True)

        files = [f for f in os.listdir(src_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

        for fname in files:
            src_file = os.path.join(src_folder, fname)
            # Prefix to avoid collision with any existing raw files
            dest_fname = f"imfdb_{fname}"
            dst_file = os.path.join(dst_folder, dest_fname)

            if os.path.exists(dst_file):
                stats["skipped_existing"] += 1
                continue

            if not dry_run:
                shutil.copy2(src_file, dst_file)

            stats[category] += 1
            stats["total_copied"] += 1

    logger.info("=" * 60)
    logger.info(f"IMFDB INGESTION COMPLETED (Dry-Run: {dry_run})")
    logger.info(f"Telugu Heroes images: {stats['telugu_heroes']}")
    logger.info(f"Actresses images:     {stats['actresses']}")
    logger.info(f"Actors images:        {stats['actors']}")
    logger.info(f"Total images copied:  {stats['total_copied']}")
    logger.info(f"Skipped existing:     {stats['skipped_existing']}")
    logger.info("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Ingest IMFDB face dataset into datasets/raw.")
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default=DEFAULT_KAGGLE_PATH,
        help="Source IMFDB dataset directory",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="datasets/raw",
        help="Target datasets/raw directory",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (Default is dry-run)",
    )
    args = parser.parse_args()

    setup_logging()
    ingest_imfdb(
        source_root=args.source,
        target_root=args.target,
        dry_run=not args.apply,
    )


if __name__ == "__main__":
    main()
