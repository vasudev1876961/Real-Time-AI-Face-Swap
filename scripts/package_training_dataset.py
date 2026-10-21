"""
Helper script to package face dataset for Google Colab GPU training.
Compresses aligned face crops into a clean zip archive ready for 1-click upload to Colab.
"""

import os
import sys
import argparse
import zipfile
import glob


def package_dataset(input_dir: str, output_zip: str):
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Directory '{input_dir}' not found.")
        return

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    if not image_paths:
        print(f"[ERROR] No face images found in '{input_dir}'")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_zip)), exist_ok=True)

    print("=" * 65)
    print("Packaging Hero Dataset for Google Colab GPU Training")
    print("=" * 65)
    print(f"Source Directory: {input_dir}")
    print(f"Found Images:     {len(image_paths)}")
    print(f"Output Zip:       {output_zip}")
    print("-" * 65)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, img_path in enumerate(image_paths, 1):
            arcname = os.path.basename(img_path)
            zipf.write(img_path, arcname=arcname)
            if idx % 20 == 0 or idx == len(image_paths):
                print(f"  Added {idx}/{len(image_paths)} images...")

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print("=" * 65)
    print(f"[SUCCESS] Dataset packaged successfully: {output_zip} ({size_mb:.2f} MB)")
    print("Next step: Upload this zip file to your Google Colab notebook:")
    print("  notebooks/07_train_hero_model_colab.ipynb")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package face dataset for Google Colab")
    parser.add_argument("--input", type=str, default="datasets/processed/telugu_heroes", help="Path to face images directory")
    parser.add_argument("--output", type=str, default="outputs/hero_dataset.zip", help="Destination zip path")
    args = parser.parse_args()

    package_dataset(args.input, args.output)
