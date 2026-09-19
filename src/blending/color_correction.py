"""
Color Correction Module (Re-export and Interface Adapter).
Bridges to src.processing.color_correction for Reinhard Lab, Gain, and Histogram matching.
"""

from src.processing.color_correction import (
    apply_color_correction,
    reinhard_color_transfer,
    gain_matching,
    histogram_matching,
    TemporalColorStabilizer,
)

__all__ = [
    "apply_color_correction",
    "reinhard_color_transfer",
    "gain_matching",
    "histogram_matching",
    "TemporalColorStabilizer",
]
