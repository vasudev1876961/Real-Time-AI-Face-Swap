"""
Command Line Tool to Import a New Target Face into the Database.
Usage:
    python scripts/add_target.py --image path/to/photo.jpg --name "Mahesh Babu" --category telugu_heroes
"""

import os
import sys
import argparse

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.targets.target_manager import get_target_manager
from src.core.config_loader import load_all_configs


def main():
    parser = argparse.ArgumentParser(description="Import a New Target Face into the Active Database")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to reference portrait photo (.jpg, .png)")
    parser.add_argument("--name", "-n", type=str, required=True, help="Display name of the target (e.g., 'Mahesh Babu')")
    parser.add_argument("--category", "-c", type=str, default="custom", help="Target category (telugu_heroes, actors, actresses, custom)")
    parser.add_argument("--id", type=str, default=None, help="Optional unique person_id slug")

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        sys.exit(1)

    print("=" * 60)
    print(f"Importing target face: '{args.name}'")
    print(f"Reference image: {args.image}")
    print(f"Category: {args.category}")
    print("=" * 60)

    _, _, t_cfg = load_all_configs()
    tm = get_target_manager(t_cfg)

    success, msg, target = tm.add_custom_target(
        image_or_path=args.image,
        display_name=args.name,
        category=args.category,
        person_id=args.id,
    )

    if success and target:
        print(f"\n[SUCCESS] {msg}")
        print(f"  Target ID: {target.target_id}")
        print(f"  Saved Image: {target.reference_image_path}")
        print(f"  Embedding Shape: {target.embedding.shape}")
        print("=" * 60)
    else:
        print(f"\n[FAILURE] {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
