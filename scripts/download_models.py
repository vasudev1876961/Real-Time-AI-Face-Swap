"""
Model Downloader Utility for Face Restoration and Enhancement Models.
Downloads GFPGANv1.4 ONNX and other optional models directly into the models/ directory.
"""

import os
import sys
import argparse
import urllib.request
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

MODEL_REGISTRY = {
    "gfpgan": {
        "filename": "face_enhancer.onnx",
        "subdir": "models/enhancement",
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx",
        "description": "GFPGAN v1.4 Face Restoration ONNX Model (512x512 super-resolution)",
        "expected_min_bytes": 300 * 1024 * 1024,  # ~340 MB
    },
    "codeformer": {
        "filename": "codeformer.onnx",
        "subdir": "models/enhancement",
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
        "description": "CodeFormer Face Restoration ONNX Model (512x512 codebook-guided)",
        "expected_min_bytes": 350 * 1024 * 1024,  # ~376 MB
    },
}


def download_file_with_progress(url: str, dest_path: str, description: str = ""):
    """Downloads a file with dynamic console progress and resume/overwrite safety."""
    print("=" * 65)
    print(f"Downloading: {description or os.path.basename(dest_path)}")
    print(f"  Source URL: {url}")
    print(f"  Destination: {dest_path}")
    print("=" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = dest_path + ".tmp"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceSwapEngine/2.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get("content-length", 0))
            chunk_size = 1024 * 1024  # 1MB chunks
            downloaded = 0
            start_time = time.perf_counter()

            with open(temp_path, "wb") as f_out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)

                    elapsed = max(0.1, time.perf_counter() - start_time)
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed

                    if total_size > 0:
                        pct = (downloaded / total_size) * 100.0
                        eta = (total_size - downloaded) / max(1, (downloaded / elapsed))
                        sys.stdout.write(
                            f"\r  [{downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB] "
                            f"{pct:.1f}% | Speed: {speed_mb:.1f} MB/s | ETA: {eta:.1f}s "
                        )
                    else:
                        sys.stdout.write(
                            f"\r  [{downloaded / (1024*1024):.1f} MB] | Speed: {speed_mb:.1f} MB/s "
                        )
                    sys.stdout.flush()

        # Atomic rename
        if os.path.isfile(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)

        final_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
        print(f"\n[SUCCESS] Download completed successfully: {dest_path} ({final_mb} MB)")
        return True

    except Exception as e:
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        print(f"\n[ERROR] Failed to download model: {e}")
        return False


def ensure_model(model_name: str = "gfpgan") -> bool:
    """Ensures model is present, downloading if necessary."""
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        print(f"[ERROR] Unknown model key: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
        return False

    info = MODEL_REGISTRY[key]
    dest = os.path.join(repo_root, info["subdir"], info["filename"])

    if os.path.isfile(dest) and os.path.getsize(dest) >= info["expected_min_bytes"]:
        mb = round(os.path.getsize(dest) / (1024 * 1024), 2)
        print(f"[INFO] Model '{key}' already exists at '{dest}' ({mb} MB).")
        return True

    return download_file_with_progress(info["url"], dest, description=info["description"])


def main():
    parser = argparse.ArgumentParser(description="Pretrained Face Swap & Enhancement Model Downloader")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gfpgan",
        choices=["gfpgan", "codeformer", "all"],
        help="Which model to download (default: gfpgan)",
    )
    args = parser.parse_args()

    if args.model == "all":
        for k in MODEL_REGISTRY:
            ensure_model(k)
    else:
        ensure_model(args.model)


if __name__ == "__main__":
    main()
