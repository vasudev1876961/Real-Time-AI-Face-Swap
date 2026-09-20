"""
Color and Illumination Adaptation with Mask-Weighted Statistics and Temporal Stabilization.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("ColorCorrection")


class TemporalColorStabilizer:
    """
    Exponential Moving Average (EMA) stabilizer for color transfer parameters
    to eliminate frame-to-frame skin tone flicker caused by camera auto-exposure.
    """

    def __init__(self, alpha: float = 0.70):
        self.alpha = alpha
        self.smoothed_scale: Optional[np.ndarray] = None
        self.smoothed_offset: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.smoothed_scale = None
        self.smoothed_offset = None

    def update(self, scale: np.ndarray, offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.smoothed_scale is None or self.smoothed_offset is None:
            self.smoothed_scale = scale.copy()
            self.smoothed_offset = offset.copy()
        else:
            self.smoothed_scale = self.alpha * scale + (1.0 - self.alpha) * self.smoothed_scale
            self.smoothed_offset = self.alpha * offset + (1.0 - self.alpha) * self.smoothed_offset
        return self.smoothed_scale, self.smoothed_offset


def _compute_channel_stats(
    img: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes mean and standard deviation, optionally weighted by face mask."""
    h, w = img.shape[:2]
    if weights is not None and weights.size > 0:
        if weights.shape[:2] != (h, w):
            w_resized = cv2.resize(weights, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            w_resized = weights
        w_arr = w_resized.astype(np.float32)
        total_w = np.sum(w_arr)
        if total_w > 10.0:
            w_3ch = w_arr[:, :, np.newaxis] if w_arr.ndim == 2 else w_arr
            mean = np.sum(img * w_3ch, axis=(0, 1)) / total_w
            variance = np.sum(w_3ch * ((img - mean) ** 2), axis=(0, 1)) / total_w
            std = np.sqrt(np.maximum(variance, 1e-4))
            return mean, std

    # Fallback to central region
    y1, y2 = int(h * 0.25), int(h * 0.85)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    sub = img[y1:y2, x1:x2]
    return sub.mean(axis=(0, 1)), np.maximum(sub.std(axis=(0, 1)), 1e-4)


def reinhard_color_transfer(
    source_img: np.ndarray,
    target_img: np.ndarray,
    blend_ratio: float = 0.65,
    mask: Optional[np.ndarray] = None,
    stabilizer: Optional[TemporalColorStabilizer] = None,
) -> np.ndarray:
    """
    Applies Reinhard color transfer in CIE-Lab color space from source (original face)
    to target (swapped face crop) to match skin illumination and tone.
    Uses mask-weighted sampling to isolate facial skin and eliminate hair/background bias.
    """
    if source_img is None or target_img is None:
        return target_img

    src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = _compute_channel_stats(src_lab, weights=mask)
    tgt_mean, tgt_std = _compute_channel_stats(tgt_lab, weights=mask)

    tgt_std = np.maximum(tgt_std, 1e-3)
    raw_scale = np.clip(src_std / tgt_std, 0.70, 1.40)
    raw_offset = src_mean - tgt_mean * raw_scale

    if stabilizer is not None:
        scale, offset = stabilizer.update(raw_scale, raw_offset)
    else:
        scale, offset = raw_scale, raw_offset

    # Harmonize LAB channels
    matched_lab = tgt_lab * scale + offset
    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)

    result_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    if blend_ratio < 1.0:
        result_bgr = cv2.addWeighted(result_bgr, blend_ratio, target_img, 1.0 - blend_ratio, 0)

    return result_bgr


def gain_color_match(
    source_img: np.ndarray,
    target_img: np.ndarray,
    blend_ratio: float = 0.65,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Linear channel-wise gain matching in BGR space with optional mask weighting.
    """
    src_mean, _ = _compute_channel_stats(source_img.astype(np.float32), weights=mask)
    tgt_mean, _ = _compute_channel_stats(target_img.astype(np.float32), weights=mask)

    gain = np.clip((src_mean + 1e-4) / (tgt_mean + 1e-4), 0.6, 1.5)
    matched = np.clip(target_img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    if blend_ratio < 1.0:
        matched = cv2.addWeighted(matched, blend_ratio, target_img, 1.0 - blend_ratio, 0)
    return matched


def match_histograms(
    source_img: np.ndarray,
    target_img: np.ndarray,
    blend_ratio: float = 0.85,
) -> np.ndarray:
    """Channel-wise histogram matching."""
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


def apply_directional_illumination_transfer(
    source_img: np.ndarray,
    target_img: np.ndarray,
    strength: float = 0.50,
    sigma: float = 16.0,
) -> np.ndarray:
    """
    Applies Retinex-based low-frequency illumination ratio transfer to harmonize
    directional lighting, key-light highlights, and cheek shadows between subject and swap.

    Args:
        source_img: Original camera face crop.
        target_img: Swapped face crop.
        strength: Illumination adaptation factor [0.0, 1.0].
        sigma: Gaussian kernel radius for spatial illumination decomposition.

    Returns:
        Illumination-harmonized BGR image.
    """
    if source_img is None or target_img is None or strength <= 0.01:
        return target_img

    h, w = target_img.shape[:2]
    sh, sw = source_img.shape[:2]
    if (sw, sh) != (w, h):
        src_resized = cv2.resize(source_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        src_resized = source_img

    src_lab = cv2.cvtColor(src_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_src = src_lab[:, :, 0]
    l_tgt = tgt_lab[:, :, 0]

    ksize = int(sigma * 3) | 1
    illum_src = cv2.GaussianBlur(l_src, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    illum_tgt = cv2.GaussianBlur(l_tgt, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    ratio = np.clip((illum_src + 1e-3) / (illum_tgt + 1e-3), 0.65, 1.45)
    new_l = np.clip(l_tgt * (1.0 + (ratio - 1.0) * strength), 0, 255)
    tgt_lab[:, :, 0] = new_l

    return cv2.cvtColor(tgt_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_color_correction(
    original_crop: np.ndarray,
    swapped_crop: np.ndarray,
    method: str = "reinhard",
    blend_ratio: float = 0.85,
    mask: Optional[np.ndarray] = None,
    stabilizer: Optional[TemporalColorStabilizer] = None,
    illumination_match: bool = True,
    illumination_strength: float = 0.45,
) -> np.ndarray:
    """Applies the configured color correction and optional directional illumination transfer."""
    m = method.strip().lower()
    if m == "reinhard":
        result = reinhard_color_transfer(
            original_crop,
            swapped_crop,
            blend_ratio=blend_ratio,
            mask=mask,
            stabilizer=stabilizer,
        )
    elif m == "gain_matching":
        result = gain_color_match(
            original_crop,
            swapped_crop,
            blend_ratio=blend_ratio,
            mask=mask,
        )
    elif m == "histogram":
        result = match_histograms(
            original_crop,
            swapped_crop,
            blend_ratio=blend_ratio,
        )
    else:
        result = swapped_crop

    if illumination_match and original_crop is not None:
        result = apply_directional_illumination_transfer(
            source_img=original_crop,
            target_img=result,
            strength=illumination_strength,
        )

    return result

