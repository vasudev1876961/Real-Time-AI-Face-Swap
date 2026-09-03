"""
Color and Illumination Adaptation to Eliminate Skin Tone Seams.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("ColorCorrection")


def reinhard_color_transfer(
    source_img: np.ndarray,
    target_img: np.ndarray,
    blend_ratio: float = 0.65,
) -> np.ndarray:
    """
    Applies Reinhard color transfer in CIE-Lab color space from source (original face)
    to target (swapped face crop) to match skin illumination and tone.
    Samples color distribution strictly from the central skin region to avoid hair/background bias.
    """
    if source_img is None or target_img is None:
        return target_img

    # Convert BGR to Lab float32
    src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Sample central facial skin area (approx 70% inner region)
    h, w = source_img.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.85)
    x1, x2 = int(w * 0.20), int(w * 0.80)

    src_skin = src_lab[y1:y2, x1:x2]
    tgt_skin = tgt_lab[y1:y2, x1:x2]

    src_mean, src_std = src_skin.mean(axis=(0, 1)), src_skin.std(axis=(0, 1))
    tgt_mean, tgt_std = tgt_skin.mean(axis=(0, 1)), tgt_skin.std(axis=(0, 1))

    # Avoid division by zero and clamp extreme contrast scaling
    tgt_std = np.maximum(tgt_std, 1e-3)
    scale = np.clip(src_std / tgt_std, 0.70, 1.40)

    # Scale and shift channels
    matched_lab = (tgt_lab - tgt_mean) * scale + src_mean
    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)

    # Convert back to BGR
    result_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    if blend_ratio < 1.0:
        result_bgr = cv2.addWeighted(result_bgr, blend_ratio, target_img, 1.0 - blend_ratio, 0)

    return result_bgr


def gain_color_match(
    source_img: np.ndarray,
    target_img: np.ndarray,
    blend_ratio: float = 0.65,
) -> np.ndarray:
    """
    Linear channel-wise gain matching in BGR space sampled on facial skin.
    """
    h, w = source_img.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.85)
    x1, x2 = int(w * 0.20), int(w * 0.80)

    src_mean = source_img[y1:y2, x1:x2].mean(axis=(0, 1)) + 1e-4
    tgt_mean = target_img[y1:y2, x1:x2].mean(axis=(0, 1)) + 1e-4

    gain = np.clip(src_mean / tgt_mean, 0.6, 1.5)
    matched = np.clip(target_img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    if blend_ratio < 1.0:
        matched = cv2.addWeighted(matched, blend_ratio, target_img, 1.0 - blend_ratio, 0)
    return matched


def match_histograms(
    source_img: np.ndarray,
    target_img: np.ndarray,
    blend_ratio: float = 0.85,
) -> np.ndarray:
    """
    Channel-wise histogram matching.
    """
    matched = np.zeros_like(target_img)
    for ch in range(3):
        src_ch = source_img[:, :, ch]
        tgt_ch = target_img[:, :, ch]

        src_hist, _ = np.histogram(src_ch.flatten(), 256, [0, 256])
        tgt_hist, _ = np.histogram(tgt_ch.flatten(), 256, [0, 256])

        src_cdf = src_hist.cumsum().astype(np.float32) / (src_ch.size + 1e-6)
        tgt_cdf = tgt_hist.cumsum().astype(np.float32) / (tgt_ch.size + 1e-6)

        lookup = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lookup[i] = int(np.argmin(np.abs(tgt_cdf[i] - src_cdf)))

        matched[:, :, ch] = cv2.LUT(tgt_ch, lookup)

    if blend_ratio < 1.0:
        matched = cv2.addWeighted(matched, blend_ratio, target_img, 1.0 - blend_ratio, 0)
    return matched


def apply_color_correction(
    original_crop: np.ndarray,
    swapped_crop: np.ndarray,
    method: str = "reinhard",
    blend_ratio: float = 0.85,
) -> np.ndarray:
    """
    Applies the configured color correction algorithm.
    """
    m = method.strip().lower()
    if m == "reinhard":
        return reinhard_color_transfer(original_crop, swapped_crop, blend_ratio)
    elif m == "gain_matching":
        return gain_color_match(original_crop, swapped_crop, blend_ratio)
    elif m == "histogram":
        return match_histograms(original_crop, swapped_crop, blend_ratio)
    return swapped_crop
