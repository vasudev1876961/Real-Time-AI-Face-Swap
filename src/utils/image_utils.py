"""
Image Utility Functions for File I/O, Quality Assessment, and Visual Formatting.
"""

import os
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


def read_image_safe(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Safely reads an image from disk, supporting Windows Unicode paths and special characters.
    Returns BGR numpy array or None if reading fails.
    """
    if not os.path.isfile(path):
        return None
    try:
        # np.fromfile handles Unicode file paths cleanly on Windows
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        # Fallback to standard OpenCV
        try:
            return cv2.imread(path, flags)
        except Exception:
            return None


def write_image_safe(path: str, img: np.ndarray, quality: int = 95) -> bool:
    """
    Safely writes an image to disk, creating parent directories and supporting Unicode paths.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if not ext:
            ext = ".jpg"
            path += ext

        params = []
        if ext in [".jpg", ".jpeg"]:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif ext == ".png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        elif ext == ".webp":
            params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]

        success, encoded = cv2.imencode(ext, img, params)
        if success:
            with open(path, "wb") as f:
                encoded.tofile(f)
            return True
        return False
    except Exception:
        return False


def calculate_blur_score(img: np.ndarray) -> float:
    """
    Calculates the blur score of an image using the Variance of the Laplacian.
    Higher values indicate a sharper image (> 100 is typically sharp).
    """
    if img is None or img.size == 0:
        return 0.0
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_phash(img: Union[np.ndarray, Image.Image], hash_size: int = 8) -> Optional[str]:
    """
    Computes a Perceptual Hash (pHash) for an image to identify near-duplicates.
    Returns hexadecimal string hash.
    """
    try:
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb)
        else:
            pil_img = img

        if HAS_IMAGEHASH:
            return str(imagehash.phash(pil_img, hash_size=hash_size))
        else:
            # Fallback simple 8x8 average hash if imagehash not installed
            resized = pil_img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(resized.getdata(), dtype=np.float32)
            avg = pixels.mean()
            diff = pixels > avg
            decimal_value = 0
            hex_str = []
            for i, v in enumerate(diff):
                if v:
                    decimal_value += 2 ** (i % 8)
                if (i % 8) == 7:
                    hex_str.append(hex(decimal_value)[2:].rjust(2, "0"))
                    decimal_value = 0
            return "".join(hex_str)
    except Exception:
        return None


def resize_aspect_ratio(
    img: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_AREA
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resizes an image preserving aspect ratio with letterboxing.
    Returns (padded_image, scale_factor, (pad_w, pad_h)).
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return canvas, scale, (pad_w, pad_h)


def normalize_image_shape(img: np.ndarray, required_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """Resizes an image directly to standard required dimensions."""
    if img is None or img.size == 0:
        return np.zeros((required_size[1], required_size[0], 3), dtype=np.uint8)
    if (img.shape[1], img.shape[0]) == required_size:
        return img
    return cv2.resize(img, required_size, interpolation=cv2.INTER_LINEAR)
