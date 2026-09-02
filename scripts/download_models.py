"""
Utility Script to Download Standard Pretrained ONNX Models.
Downloads the standard open-source ONNX face-swap and face-analysis models
directly into models/face_swap/ and models/face_analysis/.
"""

import os
import sys
import urllib.request
from tqdm import tqdm

MODELS = {
    "face_swap": {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "dest": "models/face_swap/model.onnx",
        "description": "Pretrained ONNX Face Swap Model (inswapper_128)",
    },
    "face_analysis": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/w600k_r50.onnx",
        "dest": "models/face_analysis/face_recognition.onnx",
        "description": "ArcFace / InsightFace Face Recognition Embedding Model (512-D)",
    },
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, desc: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 * 1024:
        print(f"[ALREADY EXISTS] {output_path} ({round(os.path.getsize(output_path) / (1024*1024), 1)} MB)")
        return

    print(f"\n[DOWNLOADING] {desc}...")
    print(f"  URL: {url}")
    print(f"  Destination: {output_path}")

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print(f"[SUCCESS] Saved to {output_path}")


def main():
    print("=" * 60)
    print("Pretrained ONNX Models Downloader")
    print("=" * 60)

    for key, info in MODELS.items():
        try:
            download_file(info["url"], info["dest"], info["description"])
        except Exception as e:
            print(f"\n[ERROR] Failed to download {info['description']}: {e}")
            print("You can manually place your own .onnx file at:")
            print(f"  -> {info['dest']}\n")

    print("\n" + "=" * 60)
    print("Model setup complete. You can now launch: python scripts/run_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
